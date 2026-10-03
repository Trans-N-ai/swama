import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

// MARK: - DecisionScoringError

enum DecisionScoringError: Error, LocalizedError {
    case missingChatTemplate
    case lossyTemplate
    case reasoningOpen
    case reasoningPredicted
    case invalidLabel(String)
    case contextLimitExceeded
    case invalidLogits
    case pairLabelsUnavailable
    case tooManyOptions(requested: Int, available: Int)
    case imagesNotSupported
    case invalidImage

    var errorDescription: String? {
        switch self {
        case .missingChatTemplate: "Decision scoring requires a chat template."

        case .lossyTemplate: "The rendered chat prompt does not round-trip through this tokenizer."

        case .reasoningOpen: "The chat template leaves a reasoning block open at the answer position."

        case .reasoningPredicted: "The model predicts a reasoning opener instead of a direct answer."

        case let .invalidLabel(label): "The answer label '\(label)' is not one distinct token after the chat prompt."

        case .contextLimitExceeded: "The decision prompt exceeds the configured context limit."

        case .invalidLogits: "The model returned invalid next-token logits."

        case .pairLabelsUnavailable:
            "More than 26 options per choice needs an added token before the answer position, " +
                "which this tokenizer and chat template do not provide."

        case let .tooManyOptions(requested, available):
            "This model supports at most \(available) options per choice; the question has \(requested)."

        case .imagesNotSupported: "Images are not supported by this model; it has no vision processor."

        case .invalidImage: "The image data is invalid."
        }
    }
}

// MARK: - DecisionLogits

struct DecisionLogits: Sendable {
    let promptTokenIDs: [Int]
    let labelTokenIDs: [Int]
    let labelLogProbs: [Double]
}

// MARK: - PreparedDecisionPrompt

/// A decision prompt tokenized before the exclusive model slot is acquired, so this CPU work can
/// overlap another request's model work. It is valid only for the container whose tokenizer
/// produced it; `scoreDecision` recomputes the prompt inside the slot for any other container.
struct PreparedDecisionPrompt: @unchecked Sendable {
    private weak var container: ModelContainer?
    let content: String
    let contextLimit: Int
    let prompt: String
    let promptIDs: [Int]

    init(container: ModelContainer, content: String, contextLimit: Int, prompt: String, promptIDs: [Int]) {
        self.container = container
        self.content = content
        self.contextLimit = contextLimit
        self.prompt = prompt
        self.promptIDs = promptIDs
    }

    func matches(container other: ModelContainer, content: String, contextLimit: Int) -> Bool {
        container === other && self.content == content && self.contextLimit == contextLimit
    }
}

/// Tokenize a decision prompt outside the exclusive model slot, with the container's own tokenizer.
/// Throws the same errors, in the same order, as the in-slot path.
func prepareDecisionPrompt(
    container: ModelContainer, content: String, contextLimit: Int
) async throws -> PreparedDecisionPrompt {
    try Task.checkCancellation()
    let tokenizer = await DecisionTokenizerCache.shared.tokenizer(for: container)
    let (prompt, promptIDs) = try decisionPrompt(content: content, contextLimit: contextLimit, tokenizer: tokenizer)
    return PreparedDecisionPrompt(
        container: container, content: content, contextLimit: contextLimit, prompt: prompt, promptIDs: promptIDs
    )
}

/// Two-letter answer labels usable after this container's chat prompt, computed once per cached container
/// outside the exclusive model slot. `nil` when this tokenizer and chat template cannot provide them.
func cachedDecisionPairLabels(container: ModelContainer) async throws -> [String]? {
    try Task.checkCancellation()
    let tokenizer = await DecisionTokenizerCache.shared.tokenizer(for: container)
    let boundaryTokens = await DecisionBoundaryCache.shared.tokens(for: container)
    return try await DecisionPairLabelCache.shared.labels(for: container) {
        let probes = try decisionPairLabelProbes {
            try decisionPrompt(content: $0, contextLimit: .max, tokenizer: tokenizer)
        }
        guard let probes else {
            return nil
        }

        return decisionPairLabels(probes: probes, boundaryTokens: boundaryTokens) {
            tokenizer.encode(text: $0, addSpecialTokens: false)
        }
    }
}

/// The "x" and "y" probe prompts for pair-label discovery. A chat template that cannot give a direct
/// answer position (missing, lossy, or leaving reasoning open) cannot provide the labels: `nil`.
/// Every other error, such as cancellation, keeps its own category.
func decisionPairLabelProbes(
    render: (String) throws -> (prompt: String, promptIDs: [Int])
) throws -> [(prompt: String, promptIDs: [Int])]? {
    do {
        return try ["x", "y"].map(render)
    }
    catch DecisionScoringError.missingChatTemplate, DecisionScoringError.lossyTemplate,
        DecisionScoringError.reasoningOpen
    {
        return nil
    }
}

/// The labels for a choice with more than 26 options: the first `count` usable pair labels.
/// Refuses rather than truncating.
func decisionPairLabelPrefix(count: Int, usable: [String]?) throws -> [String] {
    guard let usable else {
        throw DecisionScoringError.pairLabelsUnavailable
    }
    guard count <= usable.count else {
        throw DecisionScoringError.tooManyOptions(requested: count, available: usable.count)
    }

    return Array(usable.prefix(count))
}

/// Render the chat prompt, check it fits, check it round-trips through the tokenizer and that the
/// template does not leave a reasoning block open. Shared by the pre-slot and in-slot paths.
func decisionPrompt(
    content: String, contextLimit: Int, tokenizer: any MLXLMCommon.Tokenizer
) throws -> (prompt: String, promptIDs: [Int]) {
    let promptIDs: [Int]
    do {
        promptIDs = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": content]],
            tools: nil,
            additionalContext: ["enable_thinking": false]
        )
    }
    catch TokenizerError.missingChatTemplate {
        throw DecisionScoringError.missingChatTemplate
    }
    guard !promptIDs.isEmpty, promptIDs.count < contextLimit else {
        throw DecisionScoringError.contextLimitExceeded
    }

    let prompt = tokenizer.decode(tokenIds: promptIDs, skipSpecialTokens: false)
    guard tokenizer.encode(text: prompt, addSpecialTokens: false) == promptIDs else {
        throw DecisionScoringError.lossyTemplate
    }

    // Inspect only the generated assistant prefix. User text may quote reasoning tags.
    let closingLine = content.components(separatedBy: "\n").last ?? content
    let generationPrefix = prompt.range(of: closingLine, options: .backwards)
        .map { String(prompt[$0.upperBound...]) } ?? prompt
    guard !decisionHasOpenReasoning(generationPrefix) else {
        throw DecisionScoringError.reasoningOpen
    }

    return (prompt, promptIDs)
}

extension ModelRunner {
    /// Score the raw next-token distribution after one chat prompt. A new KV cache is allocated
    /// for every question and discarded here; the chat PromptCacheStore is never touched.
    /// `prepared` is the prompt tokenized before the slot was acquired; it is used only when it came
    /// from this runner's container, and recomputed here otherwise.
    func scoreDecision(
        content: String,
        labels: [String],
        contextLimit: Int,
        prepared: PreparedDecisionPrompt? = nil,
        images: [Data] = [],
        imageProcessing: MLXLMCommon.UserInput.Processing = .init()
    ) async throws -> DecisionLogits {
        try Task.checkCancellation()
        let boundaryTokens = await DecisionBoundaryCache.shared.tokens(for: container)
        let knownOpeners = await DecisionBoundaryCache.shared.reasoningOpeners(for: container)
        let (result, openers) = try await container.perform { [container] context in
            let tokenizer = context.tokenizer
            let prompt: String
            let promptIDs: [Int]
            // A prepared prompt is text-only: never reuse it for a request with images.
            if images.isEmpty, let prepared,
               prepared.matches(container: container, content: content, contextLimit: contextLimit)
            {
                prompt = prepared.prompt
                promptIDs = prepared.promptIDs
            }
            else {
                // With images, these checks run on the text-only render of the same message:
                // template, round-trip and an open reasoning block at the answer position.
                (prompt, promptIDs) = try decisionPrompt(
                    content: content, contextLimit: contextLimit, tokenizer: tokenizer
                )
            }

            try Task.checkCancellation()
            let cache = context.model.newCache(parameters: nil)
            let input: LMInput
            let scoredIDs: [Int]
            if images.isEmpty {
                let rawTokens = MLXArray(promptIDs)
                // LLM prefill consumes a one-dimensional sequence; VLM processors use [batch, tokens].
                input = LMInput(tokens: context.model is any LLMModel
                    ? rawTokens : rawTokens.expandedDimensions(axis: 0)
                )
                scoredIDs = promptIDs
            }
            else {
                (input, scoredIDs) = try await decisionImageInput(
                    content: content, images: images, processing: imageProcessing,
                    contextLimit: contextLimit, context: context
                )
            }
            let output: LMOutput =
                switch try context.model.prepare(input, cache: cache, state: nil, windowSize: nil) {
                case let .tokens(tokens):
                    withPreparedCache(cache, lengths: tokens.sequenceLengths) {
                        context.model(tokens[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: nil)
                    }

                case let .logits(value):
                    value
                }
            guard output.logits.ndim == 3, output.logits.dim(0) == 1, output.logits.dim(1) > 0 else {
                throw DecisionScoringError.invalidLogits
            }

            let last = output.logits[0, -1, 0...].asType(.float32)
            // Start the unchanged MLX graph before CPU label validation. The final
            // synchronization is required on both success and failure before leaving
            // ModelContainer.perform; no MLXArray escapes this isolated operation.
            asyncEval(last)
            let labelIDs: [Int]
            do {
                let labelContext = decisionLabelContext(
                    prompt: prompt, promptIDs: promptIDs, labels: labels, boundaryTokens: boundaryTokens
                ) { tokenizer.encode(text: $0, addSpecialTokens: false) }
                labelIDs = try decisionLabelIDs(promptIDs: labelContext.tokenIDs, labels: labels) { label in
                    tokenizer.encode(text: labelContext.text + label, addSpecialTokens: false)
                }
            }
            catch {
                eval(last)
                throw error
            }
            let reasoningIDs = knownOpeners ?? decisionReasoningOpenerIDs(tokenizer)
            let labelLogProbs = try decisionLabelLogProbs(
                finalLogits: last, labelIDs: labelIDs, reasoningOpenerIDs: reasoningIDs
            )
            return (
                DecisionLogits(promptTokenIDs: scoredIDs, labelTokenIDs: labelIDs, labelLogProbs: labelLogProbs),
                reasoningIDs
            )
        }
        if knownOpeners == nil {
            await DecisionBoundaryCache.shared.storeReasoningOpeners(openers, for: container)
        }
        try Task.checkCancellation()
        return result
    }

    /// The pair labels of this runner's container, for a model that was not loaded before the slot.
    func decisionPairLabels() async throws -> [String]? {
        try await cachedDecisionPairLabels(container: container)
    }
}

/// The prepared model input for a decision with images: the same single user message the chat path
/// builds (images before the text), processed by the model's own vision processor.
/// Images require a vision model; capability is decided by the loaded container, not the model name.
func decisionImageInput(
    content: String,
    images: [Data],
    processing: MLXLMCommon.UserInput.Processing,
    contextLimit: Int,
    context: ModelContext
) async throws -> (input: LMInput, tokenIDs: [Int]) {
    guard context.model is any VLMModel else {
        throw DecisionScoringError.imagesNotSupported
    }

    let userInput = try decisionImageUserInput(content: content, images: images, processing: processing)
    let input = try await context.processor.prepare(input: userInput)
    let tokenIDs = input.text.tokens.flattened().asArray(Int.self)
    // The same multimodal safety limit the chat path applies to requests with media.
    let limit = min(contextLimit, InferenceSafetyLimits.multimodalContextLimit)
    guard !tokenIDs.isEmpty, tokenIDs.count < limit else {
        throw DecisionScoringError.contextLimitExceeded
    }

    return (input, tokenIDs)
}

/// The chat user input a decision with images is scored on. Shared with tests that compare the
/// decision prompt with what the chat path prepares for the same message.
func decisionImageUserInput(
    content: String, images: [Data], processing: MLXLMCommon.UserInput.Processing
) throws -> MLXLMCommon.UserInput {
    let decoded = try images.map { data -> MLXLMCommon.UserInput.Image in
        guard let image = CIImage(data: data) else {
            throw DecisionScoringError.invalidImage
        }

        return .ciImage(image)
    }
    return MLXLMCommon.UserInput(
        chat: [.user(content, images: decoded)],
        processing: processing,
        additionalContext: ["enable_thinking": false]
    )
}

/// Check the actual answer boundary, including prefix stability and distinct label IDs.
package func decisionLabelIDs(
    promptIDs: [Int], labels: [String], encodeAppendedLabel: (String) -> [Int]
) throws -> [Int] {
    var labelIDs = [Int]()
    for label in labels {
        let encoded = encodeAppendedLabel(label)
        guard encoded.count == promptIDs.count + 1,
              Array(encoded.dropLast()) == promptIDs,
              let last = encoded.last,
              !labelIDs.contains(last)
        else {
            throw DecisionScoringError.invalidLabel(label)
        }

        labelIDs.append(last)
    }
    return labelIDs
}

package func decisionHasOpenReasoning(_ prefix: String) -> Bool {
    for (start, end) in [("<think>", "</think>"), ("[THINK]", "[/THINK]")] {
        if let opened = prefix.range(of: start, options: .backwards) {
            guard let closed = prefix.range(of: end, options: .backwards),
                  closed.lowerBound > opened.lowerBound
            else {
                return true
            }
        }
    }
    // Harmony-style templates must place the answer in their final channel.
    for marker in ["<|channel|>", "<|channel>"] {
        if let channel = prefix.range(of: marker, options: .backwards) {
            let value = prefix[channel.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
            if value.hasPrefix("analysis") || value.hasPrefix("thought") { return true }
        }
    }
    return false
}

/// Reject a recognized reasoning opener tied for the vocabulary's highest logit.
/// Low label mass by itself is not a proof of reasoning or an application accuracy threshold.
/// Token ids of the single-token reasoning openers this tokenizer has. A clean template prefix
/// does not prove the model obeyed enable_thinking=false, so a decision whose most likely next
/// token is one of these is rejected. Only exact single-token openers count, never a tokenizer's
/// unknown-token fallback.
func decisionReasoningOpenerIDs(_ tokenizer: any MLXLMCommon.Tokenizer) -> [Int] {
    ["<think>", "[THINK]"].compactMap { marker -> Int? in
        let encoded = tokenizer.encode(text: marker, addSpecialTokens: false)
        guard encoded.count == 1,
              tokenizer.decode(tokenIds: encoded, skipSpecialTokens: false) == marker
        else {
            return nil
        }

        return encoded[0]
    }
}

/// Label log-probabilities from the final-position logits, computed on the device: the maximum and
/// the log-sum-exp are reduced there, and only the label and reasoning-opener logits are copied
/// back, instead of the whole vocabulary as `[Double]`. NaN or +infinity anywhere is invalid: MLX's
/// max propagates both, so a non-finite maximum covers them. A reasoning opener at the maximum is
/// `reasoningPredicted`. The normaliser is float32 (it used to be summed in Double).
package func decisionLabelLogProbs(
    finalLogits last: MLXArray,
    labelIDs: [Int],
    reasoningOpenerIDs: [Int]
) throws -> [Double] {
    let vocabulary = 0 ..< last.dim(0)
    guard labelIDs.allSatisfy({ vocabulary.contains($0) }) else {
        eval(last)
        throw DecisionScoringError.invalidLogits
    }

    let openers = reasoningOpenerIDs.filter { vocabulary.contains($0) }
    let gathered = last.take(MLXArray((labelIDs + openers).map { Int32($0) }))
    let maximum = last.max()
    let logNormalizer = last.logSumExp()
    eval(gathered, maximum, logNormalizer)

    let maximumValue = Double(maximum.item(Float.self))
    guard maximumValue.isFinite else {
        throw DecisionScoringError.invalidLogits
    }

    let picked = gathered.asArray(Float.self).map(Double.init)
    guard picked.suffix(openers.count).contains(maximumValue) == false else {
        throw DecisionScoringError.reasoningPredicted
    }

    let normalizer = Double(logNormalizer.item(Float.self))
    guard normalizer.isFinite else {
        throw DecisionScoringError.invalidLogits
    }

    return picked.prefix(labelIDs.count).map { $0 - normalizer }
}

/// Use an added-token boundary only for tokenizer layouts whose final text segment is
/// independent of the preceding message. All unsupported or inconsistent cases retain
/// the original full-prompt validation. The first label also cross-checks the shortcut.
func decisionLabelContext(
    prompt: String, promptIDs: [Int], labels: [String], boundaryTokens: [Int: String],
    encode: (String) -> [Int]
) -> (text: String, tokenIDs: [Int]) {
    guard let position = promptIDs.lastIndex(where: { boundaryTokens[$0] != nil }),
          let marker = boundaryTokens[promptIDs[position]],
          !boundaryTokens.values.contains(where: { $0 != marker && $0.contains(marker) }),
          let range = prompt.range(of: marker, options: .backwards),
          let first = labels.first
    else {
        return (prompt, promptIDs)
    }

    let suffix = String(prompt[range.upperBound...])
    let suffixIDs = Array(promptIDs.dropFirst(position + 1))
    guard encode(suffix) == suffixIDs,
          encode(prompt + first) == Array(promptIDs.prefix(position + 1)) + encode(suffix + first)
    else {
        return (prompt, promptIDs)
    }

    return (suffix, suffixIDs)
}

/// SGLang `_pair_labels`, pinned to eb9c9ee9: candidates AA to ZZ in order, each kept only when it adds
/// exactly one token, distinct from those already kept, after the text that follows the last added
/// token. That text must not depend on the message, so every probe prompt must take the added-token
/// shortcut and end with the same text; otherwise `nil`. Each final prompt is still checked as a whole.
func decisionPairLabels(
    probes: [(prompt: String, promptIDs: [Int])], boundaryTokens: [Int: String], encode: (String) -> [Int]
) -> [String]? {
    let contexts = probes.map {
        decisionLabelContext(
            prompt: $0.prompt, promptIDs: $0.promptIDs, labels: ["AA"], boundaryTokens: boundaryTokens, encode: encode
        )
    }
    guard let context = contexts.first,
          zip(contexts, probes).allSatisfy({ $0.tokenIDs.count < $1.promptIDs.count }),
          contexts.allSatisfy({ $0.text == context.text })
    else {
        return nil
    }

    let letters = (UInt8(ascii: "A") ... UInt8(ascii: "Z")).map { String(UnicodeScalar($0)) }
    var labels = [String]()
    var labelIDs = Set<Int>()
    for first in letters {
        for second in letters {
            let encoded = encode(context.text + first + second)
            guard encoded.count == context.tokenIDs.count + 1,
                  encoded.dropLast().elementsEqual(context.tokenIDs),
                  let last = encoded.last,
                  labelIDs.insert(last).inserted
            else {
                continue
            }

            labels.append(first + second)
        }
    }
    return labels
}

/// Added tokens are split off before NFC and the Split/ByteLevel pre-tokenizers run.
/// Reject stripping, first-segment prefix insertion, ambiguous metadata and unknown layouts.
func decisionBoundaryTokens(from data: Data) -> [Int: String] {
    guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let normalizer = object["normalizer"] as? [String: Any], normalizer["type"] as? String == "NFC",
          let pre = object["pre_tokenizer"] as? [String: Any], pre["type"] as? String == "Sequence",
          let parts = pre["pretokenizers"] as? [[String: Any]], parts.count == 2,
          parts[0]["type"] as? String == "Split", parts[1]["type"] as? String == "ByteLevel",
          parts[1]["add_prefix_space"] as? Bool == false,
          let tokens = object["added_tokens"] as? [[String: Any]]
    else {
        return [:]
    }

    var result = [Int: String]()
    var texts = Set<String>()
    for token in tokens {
        guard token["lstrip"] as? Bool == false, token["rstrip"] as? Bool == false,
              let id = token["id"] as? Int, let text = token["content"] as? String,
              !text.isEmpty, result[id] == nil, texts.insert(text).inserted
        else {
            return [:]
        }

        result[id] = text
    }
    return result
}

private func decisionBoundaryMetadata(_ configuration: ModelConfiguration) -> [Int: String] {
    let directory: URL
    if let source = configuration.tokenizerSource {
        guard case let .directory(value) = source else {
            return [:]
        }

        directory = value
    }
    else {
        guard case let .directory(value) = configuration.id else {
            return [:]
        }

        directory = value
    }
    guard let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer.json")) else { return [:] }

    return decisionBoundaryTokens(from: data)
}

/// Runners are created per request; cache metadata against the shared model container.
/// Weak ownership never keeps model weights alive after the model pool evicts them.
/// The tokenizer of each live container, fetched once: reading it through the container waits for
/// any in-flight `perform`, which would serialize the pre-slot work again. Weak, like the boundary cache.
private actor DecisionTokenizerCache {
    static let shared: DecisionTokenizerCache = .init()
    private struct Entry {
        weak var container: ModelContainer?
        let tokenizer: any MLXLMCommon.Tokenizer
    }

    private var entries: [ObjectIdentifier: Entry] = [:]
    func tokenizer(for container: ModelContainer) async -> any MLXLMCommon.Tokenizer {
        let key = ObjectIdentifier(container)
        if let entry = entries[key], entry.container === container {
            return entry.tokenizer
        }
        let tokenizer = await container.tokenizer
        if let entry = entries[key], entry.container === container {
            return entry.tokenizer
        }
        entries = entries.filter { $0.value.container != nil }
        if entries.count >= 8, let evictedKey = entries.keys.first {
            entries.removeValue(forKey: evictedKey)
        }
        entries[key] = Entry(container: container, tokenizer: tokenizer)
        return tokenizer
    }
}

actor DecisionBoundaryCache {
    static let shared: DecisionBoundaryCache = .init()
    private struct Entry {
        weak var container: ModelContainer?
        let tokens: [Int: String]
    }

    private var entries: [ObjectIdentifier: Entry] = [:]
    private struct OpenerEntry {
        weak var container: ModelContainer?
        let ids: [Int]
    }

    private var openers: [ObjectIdentifier: OpenerEntry] = [:]

    func reasoningOpeners(for container: ModelContainer) -> [Int]? {
        let key = ObjectIdentifier(container)
        guard let entry = openers[key], entry.container === container else {
            return nil
        }

        return entry.ids
    }

    func storeReasoningOpeners(_ ids: [Int], for container: ModelContainer) {
        openers = openers.filter { $0.value.container != nil }
        if openers.count >= 8, let evictedKey = openers.keys.first {
            openers.removeValue(forKey: evictedKey)
        }
        openers[ObjectIdentifier(container)] = OpenerEntry(container: container, ids: ids)
    }

    func tokens(for container: ModelContainer) async -> [Int: String] {
        let key = ObjectIdentifier(container)
        if let entry = entries[key], entry.container === container {
            return entry.tokens
        }
        let configuration = await container.configuration
        if let entry = entries[key], entry.container === container {
            return entry.tokens
        }
        let tokens = decisionBoundaryMetadata(configuration)
        entries = entries.filter { $0.value.container != nil }
        if entries.count >= 8, let evictedKey = entries.keys.first {
            entries.removeValue(forKey: evictedKey)
        }
        entries[key] = Entry(container: container, tokens: tokens)
        return tokens
    }
}

/// The pair labels of each live container, computed once while the container is in the cache;
/// `nil` results are kept too. Weak and bounded to 8 entries, like the boundary cache: an evicted
/// container is computed again, with the same result.
actor DecisionPairLabelCache {
    static let shared: DecisionPairLabelCache = .init()
    private struct Entry {
        weak var container: ModelContainer?
        let labels: [String]?
    }

    private var entries: [ObjectIdentifier: Entry] = [:]
    func labels(
        for container: ModelContainer, compute: @Sendable () throws -> [String]?
    ) throws -> [String]? {
        let key = ObjectIdentifier(container)
        if let entry = entries[key], entry.container === container {
            return entry.labels
        }
        let labels = try compute()
        entries = entries.filter { $0.value.container != nil }
        if entries.count >= 8, let evictedKey = entries.keys.first {
            entries.removeValue(forKey: evictedKey)
        }
        entries[key] = Entry(container: container, labels: labels)
        return labels
    }
}

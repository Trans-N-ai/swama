import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - DecisionScoringError

enum DecisionScoringError: Error, LocalizedError {
    case missingChatTemplate
    case lossyTemplate
    case reasoningOpen
    case reasoningPredicted
    case invalidLabel(String)
    case contextLimitExceeded
    case invalidLogits

    var errorDescription: String? {
        switch self {
        case .missingChatTemplate: "Decision scoring requires a chat template."
        case .lossyTemplate: "The rendered chat prompt does not round-trip through this tokenizer."
        case .reasoningOpen: "The chat template leaves a reasoning block open at the answer position."
        case .reasoningPredicted: "The model predicts a reasoning opener instead of a direct answer."
        case let .invalidLabel(label): "The answer label '\(label)' is not one distinct token after the chat prompt."
        case .contextLimitExceeded: "The decision prompt exceeds the configured context limit."
        case .invalidLogits: "The model returned invalid next-token logits."
        }
    }
}

// MARK: - DecisionLogits

struct DecisionLogits: Sendable {
    let promptTokenIDs: [Int]
    let labelTokenIDs: [Int]
    let labelLogProbs: [Double]
}

extension ModelRunner {
    /// Score the raw next-token distribution after one chat prompt. A new KV cache is allocated
    /// for every question and discarded here; the chat PromptCacheStore is never touched.
    func scoreDecision(content: String, labels: [String], contextLimit: Int) async throws -> DecisionLogits {
        try Task.checkCancellation()
        let boundaryTokens = await DecisionBoundaryCache.shared.tokens(for: container)
        let result = try await container.perform { context in
            let tokenizer = context.tokenizer
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

            try Task.checkCancellation()
            let cache = context.model.newCache(parameters: nil)
            let rawTokens = MLXArray(promptIDs)
            // LLM prefill consumes a one-dimensional sequence; VLM processors use [batch, tokens].
            let input = LMInput(tokens: context.model is any LLMModel
                ? rawTokens : rawTokens.expandedDimensions(axis: 0)
            )
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
            eval(last)
            let values = last.asArray(Float.self).map(Double.init)
            guard !values.isEmpty, values.allSatisfy({ !$0.isNaN && $0 != .infinity }),
                  labelIDs.allSatisfy({ values.indices.contains($0) })
            else {
                throw DecisionScoringError.invalidLogits
            }

            let maximum = values.max()!
            guard maximum.isFinite else {
                throw DecisionScoringError.invalidLogits
            }

            // A clean template prefix does not prove the model obeyed enable_thinking=false.
            // Only recognize exact single-token openers, never a tokenizer's unknown-token fallback.
            let reasoningIDs = ["<think>", "[THINK]"].compactMap { marker -> Int? in
                let encoded = tokenizer.encode(text: marker, addSpecialTokens: false)
                guard encoded.count == 1,
                      tokenizer.decode(tokenIds: encoded, skipSpecialTokens: false) == marker
                else {
                    return nil
                }

                return encoded[0]
            }
            guard !decisionPredictsReasoning(logits: values, reasoningTokenIDs: reasoningIDs) else {
                throw DecisionScoringError.reasoningPredicted
            }

            let total = values.reduce(0) { $0 + exp($1 - maximum) }
            let logNormalizer = maximum + log(total)
            guard logNormalizer.isFinite else {
                throw DecisionScoringError.invalidLogits
            }

            return DecisionLogits(
                promptTokenIDs: promptIDs,
                labelTokenIDs: labelIDs,
                labelLogProbs: labelIDs.map { values[$0] - logNormalizer }
            )
        }
        try Task.checkCancellation()
        return result
    }
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
package func decisionPredictsReasoning(logits: [Double], reasoningTokenIDs: [Int]) -> Bool {
    guard let maximum = logits.max(), maximum.isFinite else {
        return false
    }

    return reasoningTokenIDs.contains { logits.indices.contains($0) && logits[$0] == maximum }
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
private actor DecisionBoundaryCache {
    static let shared: DecisionBoundaryCache = .init()
    private struct Entry {
        weak var container: ModelContainer?
        let tokens: [Int: String]
    }

    private var entries: [ObjectIdentifier: Entry] = [:]
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

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

// MARK: - DecisionScoringError

package enum DecisionScoringError: Error, LocalizedError {
    case missingChatTemplate
    case lossyTemplate
    case reasoningOpen
    case invalidLabel(String)
    case contextLimitExceeded
    case invalidLogits

    package var errorDescription: String? {
        switch self {
        case .missingChatTemplate: "Decision scoring requires a chat template."
        case .lossyTemplate: "The rendered chat prompt does not round-trip through this tokenizer."
        case .reasoningOpen: "The chat template leaves a reasoning block open at the answer position."
        case let .invalidLabel(label): "The answer label '\(label)' is not one distinct token after the chat prompt."
        case .contextLimitExceeded: "The decision prompt exceeds the configured context limit."
        case .invalidLogits: "The model returned invalid next-token logits."
        }
    }
}

// MARK: - DecisionLogits

package struct DecisionLogits: Sendable {
    package let promptTokenIDs: [Int]
    package let labelTokenIDs: [Int]
    package let labelLogProbs: [Double]
}

package extension ModelRunner {
    /// Score the raw next-token distribution after one chat prompt. A new KV cache is allocated
    /// for every question and discarded here; the chat PromptCacheStore is never touched.
    func scoreDecision(content: String, labels: [String], contextLimit: Int) async throws -> DecisionLogits {
        try Task.checkCancellation()
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

            let labelIDs = try decisionLabelIDs(promptIDs: promptIDs, labels: labels) { label in
                tokenizer.encode(text: prompt + label, addSpecialTokens: false)
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

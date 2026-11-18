import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import SwamaCore
@testable import SwamaKit
@testable import SwamaServer
import Testing

// MARK: - DecisionRawLogitTests

@Suite(
    "Decision raw logit control",
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["SWAMA_TEST_DECISIONS_MODEL"] == "1")
)
struct DecisionRawLogitTests {
    @Test func matchesUpstreamIteratorAndUncachedForward() async throws {
        if let path = ProcessInfo.processInfo.environment["SWAMA_TEST_METALLIB"] {
            try #require(FileManager.default.fileExists(atPath: path))
            GPU.metallib = URL(fileURLWithPath: path)
        }
        let baselineModel = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
        // An explicit other checkpoint runs only the native Swift controls; the Python golden
        // values below belong exclusively to the baseline checkpoint and must never be reused.
        let name = ProcessInfo.processInfo.environment["SWAMA_DECISION_CONTROL_MODEL"] ?? baselineModel
        let hasBaselinePythonReference = name == baselineModel
        let path = FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent(".swama/models/\(name)/config.json")
        try #require(FileManager.default.fileExists(atPath: path.path))
        // Independent Python mlx_lm full-prefill raw logits on this same checkpoint.
        let cases: [(String, String, DecisionQuestion, [Double])] = [
            (
                "yes_no",
                "The meeting ended ten minutes early because everyone agreed.",
                .yesNo(id: "q", question: "The meeting finished before its scheduled time."),
                [24.25, 22.875]
            ),
            (
                "choice",
                "My invoice charged me twice for the same subscription this month.",
                .choice(id: "q", question: "Which team should handle this?", options: [
                    .init(name: "billing"), .init(name: "technical"), .init(name: "sales")
                ]),
                [29.5, 26.375, 23.5]
            ),
            (
                "score",
                "The food was cold and the waiter ignored us for twenty minutes.",
                .score(
                    id: "q",
                    question: "How satisfied is the customer?",
                    levels: ["very unhappy", "unhappy", "neutral", "happy", "very happy"]
                ),
                [21.75, 23.25, 22.375, 21.75, 20.125]
            ),
            (
                "long_choice",
                String(repeating: "The service log contains another routine entry. ", count: 20)
                    + "My invoice charged me twice for the same subscription this month.",
                .choice(id: "q", question: "Which team should handle this?", options: [
                    .init(name: "billing"), .init(name: "technical"), .init(name: "sales")
                ]),
                []
            )
        ]
        for (kind, inputText, question, pythonLogits) in cases {
            let hasPythonReference = hasBaselinePythonReference && !pythonLogits.isEmpty
            let content = question.decisionPrompt(input: inputText)
            let labels = question.decisionLabels
            let scored = try await ModelPool.shared.run(modelName: name) { runner in
                try await runner.scoreDecision(content: content, labels: labels, contextLimit: 4096)
            }
            let control = try await ModelPool.shared.run(modelName: name) { runner in
                let container = await runner.container
                return try await container.perform { context in
                    let tokens = MLXArray(scored.promptTokenIDs)
                    let input = LMInput(tokens: context.model is any LLMModel ? tokens : tokens
                        .expandedDimensions(axis: 0)
                    )
                    let capture = DecisionLogitCapture(ids: scored.labelTokenIDs)
                    _ = try TokenIterator(
                        input: input,
                        model: context.model,
                        processor: capture,
                        sampler: ArgMaxSampler(),
                        maxTokens: 1
                    )
                    let direct = context.model(
                        .init(tokens: tokens.expandedDimensions(axis: 0)),
                        cache: nil,
                        state: nil
                    )
                    let uncached = DecisionLogitCapture(ids: scored.labelTokenIDs)
                    _ = uncached.process(logits: direct.logits[0..., -1, 0...])
                    let freshCache = try context.model.newCache(parameters: nil)
                    let fullCached = withPreparedCache(freshCache, lengths: input.text.sequenceLengths) {
                        context.model(
                            .init(tokens: tokens.expandedDimensions(axis: 0)),
                            cache: freshCache, state: nil
                        )
                    }
                    let cachedCapture = DecisionLogitCapture(ids: scored.labelTokenIDs)
                    _ = cachedCapture.process(logits: fullCached.logits[0..., -1, 0...])
                    let nextCache = try context.model.newCache(parameters: nil)
                    let next = withPreparedCache(nextCache, lengths: input.text.sequenceLengths) {
                        context.model.nextTokenLogits(
                            .init(tokens: tokens.expandedDimensions(axis: 0)),
                            cache: nextCache, state: nil
                        )
                    }
                    let nextCapture = DecisionLogitCapture(ids: scored.labelTokenIDs)
                    _ = nextCapture.process(logits: next.logits[0..., -1, 0...])
                    let preparedCache = try context.model.newCache(parameters: nil)
                    let preparedOutput: LMOutput =
                        switch try context.model.prepare(
                            input,
                            cache: preparedCache,
                            state: nil,
                            prefill: .init()
                        ) {
                        case let .logits(output):
                            output
                        case let .tokens(text):
                            withPreparedCache(preparedCache, lengths: text.sequenceLengths) {
                                context.model.nextTokenLogits(text[text: .newAxis], cache: preparedCache, state: nil)
                            }
                        }
                    let preparedCapture = DecisionLogitCapture(ids: scored.labelTokenIDs)
                    _ = preparedCapture.process(logits: preparedOutput.logits[0..., -1, 0...])
                    return (
                        capture.values, uncached.values, capture.dtype, capture.rawValues,
                        uncached.rawValues, cachedCapture.values, cachedCapture.rawValues,
                        nextCapture.values, nextCapture.rawValues, String(reflecting: type(of: context.model)),
                        preparedCapture.values, preparedCapture.rawValues,
                        fullCached.logits.dim(1), next.logits.dim(1), direct.logits.dim(1),
                        preparedOutput.logits.dim(1)
                    )
                }
            }
            let evidence: [String: Any] = [
                "case": kind,
                "model": name,
                "dtype": control.2,
                "score_logprobs": scored.labelLogProbs,
                "iterator_logprobs": control.0,
                "uncached_logprobs": control.1,
                "raw_logits": control.3,
                "uncached_raw_logits": control.4,
                "full_new_cache_logprobs": control.5,
                "full_new_cache_raw_logits": control.6,
                "next_token_new_cache_logprobs": control.7,
                "next_token_new_cache_raw_logits": control.8,
                "runtime_model_type": control.9,
                "prepared_new_cache_logprobs": control.10,
                "prepared_new_cache_raw_logits": control.11,
                "full_new_cache_logit_rows": control.12,
                "next_token_new_cache_logit_rows": control.13,
                "uncached_logit_rows": control.14,
                "prepared_new_cache_logit_rows": control.15,
                "prompt_token_count": scored.promptTokenIDs.count,
                "metallib_override": GPU.metallib?.path ?? "automatic",
                "python_raw_logits": hasPythonReference ? pythonLogits : [],
                "python_reference_available": hasPythonReference
            ]
            if let path = ProcessInfo.processInfo.environment["SWAMA_DECISION_CONTROL_OUT"] {
                if !FileManager.default.fileExists(atPath: path) {
                    FileManager.default.createFile(atPath: path, contents: nil)
                }
                let file = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
                defer { try? file.close() }
                try file.seekToEnd()
                try file.write(contentsOf: JSONSerialization.data(withJSONObject: evidence, options: [.sortedKeys]))
                try file.write(contentsOf: Data([0x0A]))
            }
            try print("DECISION_CONTROL " +
                String(
                    decoding: JSONSerialization.data(withJSONObject: evidence, options: [.sortedKeys]),
                    as: UTF8.self
                )
            )
            try #require(control.0.count == labels.count && control.1.count == labels.count && control.3.count == labels
                .count
            )
            #expect(control.2 == "bfloat16")
            for (actual, reference) in zip(scored.labelLogProbs, control.0) {
                #expect(abs(actual - reference) < 1e-5)
            }
            for reference in [control.5, control.7, control.10] {
                try #require(reference.count == labels.count)
                for (actual, value) in zip(scored.labelLogProbs, reference) {
                    #expect(abs(actual - value) < 1e-5)
                }
            }
            if kind == "long_choice" {
                #expect((97 ... 512).contains(scored.promptTokenIDs.count))
                #expect(control.15 < scored.promptTokenIDs.count, "the VLM prefill must actually trim projection rows")
            }
            // The uncached Qwen path has different bf16 rounding. Keep cache-aware
            // references exact, and check centered raw logits against the uncached arm.
            let uncachedMean = control.4.reduce(0, +) / Double(control.4.count)
            let cachedMean = control.3.reduce(0, +) / Double(control.3.count)
            for (actual, reference) in zip(control.3, control.4) {
                let ulp = pow(2.0, floor(log2(abs(reference))) - 7)
                #expect(abs((actual - cachedMean) - (reference - uncachedMean)) <= ulp + 1e-12)
            }
            #expect(control.3.indices.max { control.3[$0] < control.3[$1] }
                == control.4.indices.max { control.4[$0] < control.4[$1] }
            )
            if hasPythonReference {
                // Conditional probabilities are invariant to a shared logit shift. Compare centered
                // logits, then separately retain the absolute shift and full-vocabulary label_mass.
                // Cache-aware and Python full-forward paths can share a raw-logit shift;
                // retain that shift in the evidence and bound the centered differences.
                let actualMean = control.3.reduce(0, +) / Double(control.3.count)
                let referenceMean = pythonLogits.reduce(0, +) / Double(pythonLogits.count)
                for (actual, reference) in zip(control.3, pythonLogits) {
                    let ulp = pow(2.0, floor(log2(abs(reference))) - 7)
                    #expect(abs((actual - actualMean) - (reference - referenceMean)) <= ulp + 1e-12)
                }
                if kind == "score" {
                    #expect(abs(expectedScore(control.3) - expectedScore(pythonLogits)) <= 0.05)
                }
                let actualWinner = control.3.indices.max { control.3[$0] < control.3[$1] }
                let referenceWinner = pythonLogits.indices.max { pythonLogits[$0] < pythonLogits[$1] }
                #expect(actualWinner == referenceWinner)
            }
        }
        await ModelPool.shared.clearCache()
    }

    @Test func invalidLabelAfterAsyncDispatchRecoversWithIdenticalAnswer() async throws {
        let model = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
        let content = "Choose between A and B. Answer with one letter only."
        let before = try await ModelPool.shared.run(modelName: model) { runner in
            try await runner.scoreDecision(content: content, labels: ["A", "B"], contextLimit: 4096)
        }
        var rejected = false
        do {
            _ = try await ModelPool.shared.run(modelName: model) { runner in
                try await runner.scoreDecision(
                    content: content, labels: ["A", "this label needs multiple tokens"], contextLimit: 4096
                )
            }
        }
        catch SwamaKit.DecisionScoringError.invalidLabel {
            rejected = true
        }
        #expect(rejected)
        let after = try await ModelPool.shared.run(modelName: model) { runner in
            try await runner.scoreDecision(content: content, labels: ["A", "B"], contextLimit: 4096)
        }
        #expect(after.promptTokenIDs == before.promptTokenIDs)
        #expect(after.labelTokenIDs == before.labelTokenIDs)
        #expect(after.labelLogProbs == before.labelLogProbs)
        await ModelPool.shared.clearCache()
    }

    private func expectedScore(_ logits: [Double]) -> Double {
        let maximum = logits.max()!
        let weights = logits.map { exp($0 - maximum) }
        return weights.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element } / weights.reduce(0, +)
    }
}

// MARK: - DecisionLogitCapture

private final class DecisionLogitCapture: LogitProcessor {
    init(ids: [Int]) { self.ids = ids }
    let ids: [Int]
    var values: [Double] = .init()
    var rawValues: [Double] = .init()
    var dtype = ""
    func prompt(_: MLXArray) {}
    func didSample(token _: MLXArray) {}
    func process(logits: MLXArray) -> MLXArray {
        dtype = String(describing: logits.dtype)
        let flat = logits.asType(.float32).reshaped(-1)
        let raw = flat.asArray(Float.self)
        rawValues = ids.map { Double(raw[$0]) }
        let normalized = flat - flat.logSumExp()
        let all = normalized.asArray(Float.self)
        values = ids.map { Double(all[$0]) }
        return logits
    }
}

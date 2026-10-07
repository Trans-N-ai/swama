import CryptoKit
import Foundation
import MLX
import SwamaCore
@testable import SwamaKit
@testable import SwamaRuntime
@testable import SwamaServer
import Testing

// MARK: - DecisionAPITests

@Suite("Decision API")
struct DecisionAPITests {
    private func parse(_ text: String) throws -> DecisionRequest {
        let value = try JSONDecoder().decode([String: JSONValue].self, from: Data(text.utf8))
        return try OpenAIDecisionRequest.parse(value).decision
    }

    @Test func parsesThreeQuestionTypesAndPromptFormat() throws {
        let request =
            try parse(
                #"{"model":"m","input":"context","questions":[{"type":"choice","instructions":"Which?","choices":[{"value":"alpha"},{"value":"beta","description":"second"}]},{"type":"score","instructions":"Rate?","levels":[{"label":"bad"},{"label":"good"}]},{"type":"predicate","instructions":"True?"}]}"#
            )
        #expect(request.questions.count == 3)
        #expect(request.returnPromptTokenIDs == false)
        #expect(request.questions[0].decisionLabels == ["A", "B"])
        #expect(request.questions[1].decisionLabels == ["0", "1"])
        #expect(request.questions[2].decisionLabels == ["yes", "no"])
        #expect(request.questions[0]
            .decisionPrompt(input: request.input) ==
            "context\n\nQuestion: Which?\nA: alpha\nB: beta - second\nAnswer with the letter of one option only."
        )
    }

    @Test func rejectsUnsupportedAndMalformedWireFields() throws {
        let base = #"{"model":"m","input":"x","questions":[{"type":"predicate","instructions":"true?"}]}"#
        #expect(throws: DecisionWireError.self) { try parse(base.replacingOccurrences(
            of: "\"model\":",
            with: "\"unknown\":1,\"model\":"
        )) }
        #expect(throws: DecisionWireError.self) { try parse(base.replacingOccurrences(
            of: "\"instructions\":\"true?\"",
            with: "\"instructions\":9"
        )) }
        #expect(throws: DecisionWireError.self) { try parse(base.replacingOccurrences(
            of: "\"type\":\"predicate\"",
            with: "\"type\":\"other\""
        )) }
        // Temperature, prompt version and thinking toggles are the removed SGLang shape.
        for field in [
            "\"prompt_format_version\":1",
            "\"chat_template_kwargs\":{\"enable_thinking\":false}",
            "\"temperature\":1"
        ] {
            #expect(throws: DecisionWireError.self) { try parse(base.replacingOccurrences(
                of: "\"model\":\"m\"",
                with: "\"model\":\"m\"," + field
            )) }
        }
    }

    @Test func computesConditionalProbabilitiesWithoutAlteringLabelMass() throws {
        let question = DecisionQuestion.yesNo(id: "y", question: "True?")
        let answer = try question.decisionAnswer(
            logProbs: [log(0.377), log(0.095)],
            temperature: 1,
            promptTokenIDs: [1, 2],
            labelTokenIDs: [9405, 2083]
        )
        #expect(abs(answer.labelMass - 0.472) < 1e-12)
        #expect(abs(answer.probabilities.values.reduce(0, +) - 1) < 1e-12)
        #expect(abs(answer.probabilities["yes"]! - 0.377 / 0.472) < 1e-12)
        #expect(answer.promptTokenIDs == [1, 2])
    }

    @Test func runtimeAndHTTPScoringStayEquivalent() throws {
        let questions: [(DecisionQuestion, RuntimeDecisionQuestion)] = [
            (
                .yesNo(id: "y", question: "True?", yes: "affirm", no: "deny"),
                .yesNo(id: "y", question: "True?", yes: "affirm", no: "deny")
            ),
            (
                .score(id: "s", question: "Rate", levels: ["bad", "good"]),
                .score(id: "s", question: "Rate", levels: ["bad", "good"])
            ),
            (
                .choice(
                    id: "c",
                    question: "Pick",
                    options: [.init(name: "a"), .init(name: "b", description: "second")]
                ),
                .choice(
                    id: "c",
                    question: "Pick",
                    options: [.init(name: "a", description: nil), .init(name: "b", description: "second")]
                )
            )
        ]
        for (core, runtime) in questions {
            #expect(core.decisionPrompt(input: "context") == runtime.content(input: "context"))
            #expect(core.decisionLabels == runtime.labels)
            for temperature in [Double.leastNonzeroMagnitude, 0.5, 1, 2] {
                let lhs = try core.decisionAnswer(
                    logProbs: [log(0.6), log(0.2)],
                    temperature: temperature,
                    promptTokenIDs: [1, 2],
                    labelTokenIDs: [10, 11]
                )
                let rhs = try SwamaRuntime.decisionAnswer(
                    question: runtime,
                    logProbs: [log(0.6), log(0.2)],
                    temperature: temperature,
                    promptTokenIDs: [1, 2],
                    labelTokenIDs: [
                        10,
                        11
                    ]
                )
                #expect(lhs.probabilities == rhs.probabilities)
                #expect(lhs.labelMass == rhs.labelMass)
                #expect(lhs.choice == rhs.choice)
                #expect(lhs.score == rhs.score)
                #expect(lhs.confidence == rhs.confidence)
            }
        }
    }

    @Test func confidenceMatchesPinnedSGLangFixtures() throws {
        // Reference values from SystemOne at eb9c9ee9, including its midpoint denominator,
        // first-maximum tie policy, and zero floor for dispersed ordinal answers.
        let fixtures: [([Double], Double, Double)] = [
            ([0.5, 0.5], 0, 0),
            ([1.0 / 3, 1.0 / 3, 1.0 / 3], 0, 0),
            ([1, 0, 0], 1, 1),
            ([0, 1, 0], 1, 1),
            ([0.7, 0.2, 0.1], 0.55, 0.4),
            ([0.1, 0.2, 0.4, 0.2, 0.1], 0.25, 1.0 / 3),
            ([0.6, 0.2, 0.1, 0.05, 0.05], 0.5, 0.375),
            ([0.4, 0.4, 0.1, 0.1], 0.2, 0.1),
            ([0.1, 0.1, 0.4, 0.4], 0.2, 0.3),
            ([0.34, 0, 0.33, 0, 0.33], 0.175, 0)
        ]
        for (probabilities, expectedChoice, expectedScore) in fixtures {
            let labels = probabilities.indices.map(String.init)
            for (question, runtimeQuestion, expected) in [
                (
                    DecisionQuestion.choice(id: "q", question: "Pick", options: labels.map { .init(name: $0) }),
                    RuntimeDecisionQuestion.choice(
                        id: "q",
                        question: "Pick",
                        options: labels.map { .init(name: $0, description: nil) }
                    ),
                    expectedChoice
                ),
                (
                    DecisionQuestion.score(id: "q", question: "Rate", levels: labels),
                    RuntimeDecisionQuestion.score(id: "q", question: "Rate", levels: labels),
                    expectedScore
                )
            ] {
                let answer = try question.decisionAnswer(
                    logProbs: probabilities.map { log($0 * 0.6) },
                    temperature: 1,
                    promptTokenIDs: nil,
                    labelTokenIDs: nil
                )
                let confidence = try #require(answer.confidence)
                #expect(abs(confidence - expected) < 1e-12)
                #expect((0 ... 1).contains(confidence))
                // Exercise the actual Core runtime path against the oracle, not only n=2 parity.
                // At n=2 absolute distance and squared distance are indistinguishable.
                let runtimeAnswer = try SwamaRuntime.decisionAnswer(
                    question: runtimeQuestion,
                    logProbs: probabilities.map { log($0 * 0.6) },
                    temperature: 1,
                    promptTokenIDs: nil,
                    labelTokenIDs: nil
                )
                let runtimeConfidence = try #require(runtimeAnswer.confidence)
                #expect(abs(runtimeConfidence - expected) < 1e-12)
                #expect(runtimeAnswer.probabilities == answer.probabilities)
                #expect(runtimeQuestion.content(input: "context") == question.decisionPrompt(input: "context"))
                #expect(runtimeQuestion.labels == question.decisionLabels)
            }
        }
    }

    @Test func decisionAdapterSourcesStayPinned() throws {
        let package = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        // These adapters have different boundary types; changes require both semantic parity
        // and an explicit reviewed source-baseline update, like RuntimeLineageTests.
        for (path, expected) in [
            (
                "Sources/SwamaRuntime/CoreBridge/RuntimeDecisions.swift",
                "20d1586a3b5a4df97421fa34d22e0a18ae36e242d6d742170a61056f9ecdcf44"
            ),
            (
                "Sources/SwamaServer/DecisionCalculation.swift",
                "876436c1de45be0f87939f87a4412dd64bfdc390abeed34b1809e6faa06e9adc"
            )
        ] {
            let bytes = try Data(contentsOf: package.appendingPathComponent(path))
            let actual = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
            #expect(actual == expected, "Decision adapter drift: \(path)")
        }
    }

    @Test func confidenceTracksTemperatureButDoesNotClaimLabelCoverage() throws {
        let question = DecisionQuestion.choice(id: "q", question: "Pick", options: [
            .init(name: "a"), .init(name: "b"), .init(name: "c")
        ])
        let logProbs = [0.7, 0.2, 0.1].map { log($0 * 0.01) }
        let normal = try question.decisionAnswer(
            logProbs: logProbs,
            temperature: 1,
            promptTokenIDs: nil,
            labelTokenIDs: nil
        )
        let sharp = try question.decisionAnswer(
            logProbs: logProbs,
            temperature: 0.5,
            promptTokenIDs: nil,
            labelTokenIDs: nil
        )
        #expect(abs(normal.confidence! - 0.55) < 1e-12)
        #expect(abs(sharp.confidence! - 0.8611111111111112) < 1e-12)
        #expect(normal.labelMass == sharp.labelMass)
        #expect(abs(normal.labelMass - 0.01) < 1e-12)
        let certain = try question.decisionAnswer(
            logProbs: [log(0.001), -.infinity, -.infinity],
            temperature: 1,
            promptTokenIDs: nil,
            labelTokenIDs: nil
        )
        #expect(certain.confidence == 1)
        #expect(abs(certain.labelMass - 0.001) < 1e-12)
        let binary = try DecisionQuestion.yesNo(id: "q", question: "True?").decisionAnswer(
            logProbs: [log(0.8), log(0.2)], temperature: 1, promptTokenIDs: nil, labelTokenIDs: nil)
        #expect(binary.confidence == nil)
    }

    @Test func labelLogProbsMatchADoubleReferenceAndRejectPredictedReasoning() throws {
        // Same rules the CPU path used: NaN or +inf anywhere is invalid, -inf is fine, a
        // reasoning opener at the maximum (including a tie) is rejected, and an opener id outside
        // the vocabulary is ignored.
        var generator = SplitMix(seed: 20_261_002)
        let random = (0 ..< 4096).map { _ in Float(generator.next() % 4000) / 100 - 20 }
        let cases: [(logits: [Float], labels: [Int], openers: [Int], expected: Expectation)] = [
            (random, [7, 70, 700], [], .valid),
            (random, [7, 70, 700], [-1, 9999], .valid),
            ([10, -30, -32], [1, 2], [0], .reasoning),
            ([10, 10, -32], [0, 2], [1], .reasoning),
            ([10, -30, -32], [0, 2], [1], .valid),
            ([10, -.infinity, -32], [0, 2], [], .valid),
            ([10, .nan, -32], [0, 2], [], .invalid),
            ([10, .infinity, -32], [0, 2], [], .invalid),
            ([10, -30, -32], [0, 3], [], .invalid)
        ]
        for (logits, labels, openers, expected) in cases {
            let reference = Self.doubleReference(logits: logits, labels: labels, openers: openers)
            for compute in [SwamaKit.decisionLabelLogProbs, SwamaRuntime.decisionLabelLogProbs] {
                let outcome: Expectation
                var values: [Double] = []
                do {
                    values = try compute(MLXArray(logits), labels, openers)
                    outcome = .valid
                }
                catch SwamaKit.DecisionScoringError.reasoningPredicted,
                    SwamaRuntime.DecisionScoringError.reasoningPredicted
                {
                    outcome = .reasoning
                }
                catch SwamaKit.DecisionScoringError.invalidLogits, SwamaRuntime.DecisionScoringError.invalidLogits {
                    outcome = .invalid
                }
                #expect(outcome == expected, "\(labels) \(openers)")
                if expected == .valid, let reference {
                    for (got, want) in zip(values, reference) {
                        #expect(abs(got - want) < 1e-5, "\(got) vs \(want)")
                    }
                }
            }
        }
    }

    private enum Expectation { case valid, invalid, reasoning }

    /// The pre-change CPU computation, kept here as the reference.
    private static func doubleReference(logits: [Float], labels: [Int], openers _: [Int]) -> [Double]? {
        let values = logits.map(Double.init)
        guard let maximum = values.max(), maximum.isFinite else {
            return nil
        }

        let total = values.reduce(0) { $0 + exp($1 - maximum) }
        let normalizer = maximum + log(total)
        return labels.map { values.indices.contains($0) ? values[$0] - normalizer : .nan }
    }

    @Test func rejectsOpenReasoningPrefixes() {
        let cases: [(String, Bool)] = [
            ("assistant\n<think>\n", true),
            ("[THINK]reasoning", true),
            ("assistant<|channel|>analysis", true),
            ("assistant\n<think>\n\n</think>\n\n", false),
            ("assistant<|channel|>final", false),
            ("assistant\n", false)
        ]
        for (prefix, expected) in cases {
            let actual = SwamaKit.decisionHasOpenReasoning(prefix)
            #expect(actual == expected)
        }
    }

    @Test func refusesMultiTokenLabelsAndChangedAnswerBoundary() throws {
        #expect(try SwamaKit.decisionLabelIDs(promptIDs: [1, 2], labels: ["yes", "no"]) {
            [1, 2, $0 == "yes" ? 10 : 11]
        } == [10, 11])
        for encoding in [[1, 2, 10, 11], [1, 3, 10], [1, 2]] {
            #expect(throws: SwamaKit.DecisionScoringError.self) {
                try SwamaKit.decisionLabelIDs(promptIDs: [1, 2], labels: ["definitely"]) { _ in encoding }
            }
        }
        #expect(throws: SwamaKit.DecisionScoringError.self) {
            try SwamaKit.decisionLabelIDs(promptIDs: [1, 2], labels: ["yes", "no"]) { _ in [1, 2, 10] }
        }
    }

    @Test func extremeTemperaturesRemainNormalized() throws {
        let question = DecisionQuestion.choice(id: "q", question: "Pick", options: [
            .init(name: "a"), .init(name: "b")
        ])
        for temperature in [Double.leastNonzeroMagnitude, 0.5, 2, Double.greatestFiniteMagnitude] {
            let answer = try question.decisionAnswer(
                logProbs: [log(0.6), log(0.2)], temperature: temperature,
                promptTokenIDs: nil, labelTokenIDs: nil
            )
            #expect(abs(answer.probabilities.values.reduce(0, +) - 1) < 1e-12)
            #expect(abs(answer.labelMass - 0.8) < 1e-12)
            #expect(answer.choice == "a")
        }
    }

    @Test func rejectsImpossibleDistributions() {
        let question = DecisionQuestion.yesNo(id: "q", question: "True?")
        for values in [[Double.nan, -1], [Double.infinity, -1], [-Double.infinity, -Double.infinity], [0, 0], [-1]] {
            #expect(throws: SwamaError.self) {
                try question.decisionAnswer(
                    logProbs: values,
                    temperature: 1,
                    promptTokenIDs: nil,
                    labelTokenIDs: nil
                )
            }
        }
    }

    @Test func scoreIsExpectedLevelAndTiesUseFirstChoice() throws {
        let score = try DecisionQuestion.score(id: "s", question: "Rate", levels: ["low", "mid", "high"])
            .decisionAnswer(
                logProbs: [log(0.1), log(0.2), log(0.1)],
                temperature: 1,
                promptTokenIDs: nil,
                labelTokenIDs: nil
            )
        #expect(abs(score.score! - 1) < 1e-12)
        let choice = try DecisionQuestion.choice(
            id: "c",
            question: "Pick",
            options: [.init(name: "a"), .init(name: "b")]
        )
        .decisionAnswer(
            logProbs: [log(0.2), log(0.2)],
            temperature: 1,
            promptTokenIDs: nil,
            labelTokenIDs: nil
        )
        #expect(choice.choice == "a")
    }

    @Test func coreRejectsInvalidQuestionsBeforeBackend() async throws {
        let backend = DecisionTestBackend()
        let engine = SwamaEngine(backend: backend)
        let invalidQuestions: [[DecisionQuestion]] = [
            [],
            [.yesNo(id: "x", question: "a"), .yesNo(id: "x", question: "b")],
            [.yesNo(id: " ", question: "a")],
            [.score(id: "s", question: "rate", levels: ["low"])],
            [.choice(id: "c", question: "pick", options: [.init(name: " A "), .init(name: "a")])],
            [.choice(id: "c", question: "pick", options: [.init(name: "a\nb"), .init(name: "c")])]
        ]
        for questions in invalidQuestions {
            do {
                _ = try await engine.decide(.init(model: .init("org/model"), input: "text", questions: questions))
                Issue.record("Invalid questions reached the backend")
            }
            catch let error as SwamaError {
                #expect(error.code == .invalidRequest)
            }
        }
        #expect(await backend.calls == 0)
        _ = try await engine.decide(.init(
            model: .init("org/model"),
            input: "text",
            questions: [.yesNo(id: "q", question: "True?")]
        ))
        #expect(await backend.calls == 1)
    }

    @Test func coreRequiresInputUnlessTheCallerAllowsBlankInput() async throws {
        let backend = DecisionTestBackend()
        let engine = SwamaEngine(backend: backend)
        let questions: [DecisionQuestion] = [.yesNo(id: "q", question: "True?")]
        #expect(DecisionRequest(model: .init("org/model"), input: "", questions: questions).allowsBlankInput == false)
        for input in ["", " \n\t"] {
            do {
                _ = try await engine.decide(.init(model: .init("org/model"), input: input, questions: questions))
                Issue.record("A blank input reached the backend by default")
            }
            catch let error as SwamaError {
                #expect(error.code == .invalidRequest)
                #expect(error.message == "Decision input, questions, and positive finite temperature are required.")
            }
        }
        #expect(await backend.calls == 0)
        for input in ["", " \n\t"] {
            _ = try await engine.decide(.init(
                model: .init("org/model"), input: input, questions: questions, allowsBlankInput: true
            ))
        }
        #expect(await backend.calls == 2)
    }
}

// MARK: - DecisionTestBackend

private actor DecisionTestBackend: SwamaEngineBackend {
    private(set) var calls = 0
    func decide(_ request: DecisionRequest) async throws -> DecisionResponse {
        calls += 1
        return .init(model: request.model, answers: [:], usage: .init(promptTokens: 0, completionTokens: 0))
    }

    func generate(
        _: GenerationRequest,
        onEvent _: (@Sendable (GenerationEvent) async throws -> Void)?
    ) async throws -> GenerationResponse {
        throw SwamaError(code: .backendFailure, message: "Unexpected generation")
    }

    func embed(_: EmbeddingRequest) async throws -> SwamaCore.EmbeddingResponse {
        throw SwamaError(code: .backendFailure, message: "Unexpected embedding")
    }

    func models() async throws -> [SwamaCore.ModelInfo] { [] }
    func fetch(_ model: ModelID) async throws -> ModelID { model }
    func remove(_: ModelID) async throws {}
    func clearCache(for _: ModelID) async {}
    func clearCache() async {}
}

// MARK: - SplitMix

/// Small deterministic generator for synthetic logits.
private struct SplitMix {
    var state: UInt64
    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

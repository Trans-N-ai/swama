import CryptoKit
import Foundation
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
        return try DecisionsHandler.parse(value)
    }

    @Test func parsesThreeQuestionTypesAndPromptFormat() throws {
        let request =
            try parse(
                #"{"model":"m","input":"context","questions":[{"id":"c","type":"choice","question":"Which?","options":[{"name":"alpha"},{"name":"beta","description":"second"}]},{"id":"s","type":"score","question":"Rate?","levels":["bad","good"]},{"id":"y","type":"yes_no","question":"True?"}],"chat_template_kwargs":{"enable_thinking":false},"prompt_format_version":1,"return_prompt_token_ids":true}"#
            )
        #expect(request.questions.count == 3)
        #expect(request.returnPromptTokenIDs)
        #expect(request.questions[0].decisionLabels == ["A", "B"])
        #expect(request.questions[1].decisionLabels == ["0", "1"])
        #expect(request.questions[2].decisionLabels == ["yes", "no"])
        #expect(request.questions[0]
            .decisionPrompt(input: request.input) ==
            "context\n\nQuestion: Which?\nA: alpha\nB: beta - second\nAnswer with the letter of one option only."
        )
    }

    @Test func rejectsUnsupportedAndMalformedWireFields() throws {
        let base = #"{"model":"m","input":"x","questions":[{"id":"q","type":"yes_no","question":"true?"}]}"#
        #expect(throws: DecisionWireError.self) { try parse(base.replacingOccurrences(
            of: "\"model\":",
            with: "\"unknown\":1,\"model\":"
        )) }
        #expect(throws: DecisionWireError.self) { try parse(base.replacingOccurrences(
            of: "\"question\":\"true?\"",
            with: "\"question\":9"
        )) }
        #expect(throws: DecisionWireError.self) { try parse(base.replacingOccurrences(
            of: "\"type\":\"yes_no\"",
            with: "\"type\":\"other\""
        )) }
        #expect(throws: DecisionWireError.self) { try parse(base.replacingOccurrences(
            of: "\"model\":\"m\"",
            with: "\"model\":\"m\",\"prompt_format_version\":2"
        )) }
        #expect(throws: DecisionWireError.self) { try parse(base.replacingOccurrences(
            of: "\"model\":\"m\"",
            with: "\"model\":\"m\",\"chat_template_kwargs\":{\"enable_thinking\":true}"
        )) }
        #expect(throws: DecisionWireError.self) { try parse(base.replacingOccurrences(
            of: "\"model\":\"m\"",
            with: "\"model\":\"m\",\"temperature\":0"
        )) }
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

    @Test func rejectsPredictedReasoningWithoutRejectingLowLabelMassAlone() {
        for (logits, ids, expected) in [
            ([10.0, -30, -32], [0], true),
            ([10.0, 10, -32], [1], true),
            ([10.0, -30, -32], [1], false),
            ([10.0, -30, -32], [], false),
            ([10.0, -30, -32], [-1, 3], false)
        ] {
            #expect(SwamaKit.decisionPredictsReasoning(logits: logits, reasoningTokenIDs: ids) == expected)
            #expect(SwamaRuntime.decisionPredictsReasoning(logits: logits, reasoningTokenIDs: ids) == expected)
        }
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

import Foundation
import SwamaCore
@testable import SwamaKit
@testable import SwamaRuntime
@testable import SwamaServer
import Synchronization
import Testing

// MARK: - DecisionPairLabelTests

/// SystemOne choices with more than 26 options take two-letter labels, built the way SGLang's
/// `_pair_labels` builds them (pinned to eb9c9ee9). Fake tokenizers keep these tests model-free.
@Suite("Decision pair labels")
struct DecisionPairLabelTests {
    private static let letters = (UInt8(ascii: "A") ... UInt8(ascii: "Z")).map { String(UnicodeScalar($0)) }
    private static let candidates = letters.flatMap { first in letters.map { first + $0 } }

    /// Encodes "<message><B><tail><label>": the message by `messages`, `<B>` as added token 50, the
    /// longest matching tail by `tails`, and the label by `labels` (each candidate one distinct token
    /// unless overridden). `probeTails` is the text the chat template puts after `<B>` for a message.
    private struct FakeTokenizer {
        var messages = ["x": [1], "y": [2]]
        var tails = ["tail": [60]]
        var probeTails = ["x": "tail", "y": "tail"]
        var labels: [String: [Int]] = .init()

        func encode(_ text: String) -> [Int] {
            if let marker = text.range(of: "<B>", options: .backwards) {
                return encode(String(text[..<marker.lowerBound])) + [50] + encode(String(text[marker.upperBound...]))
            }
            if let message = messages[text] {
                return message
            }
            guard let tail = tails.keys.filter({ text.hasPrefix($0) }).max(by: { $0.count < $1.count }) else {
                return [999]
            }

            let label = String(text.dropFirst(tail.count))
            if label.isEmpty {
                return tails[tail]!
            }
            return tails[tail]! + (labels[label] ?? DecisionPairLabelTests.candidates
                .firstIndex(of: label)
                .map { [100 + $0] } ?? [998, 997]
            )
        }

        func probe(_ message: String) -> (prompt: String, promptIDs: [Int]) {
            let prompt = message + "<B>" + probeTails[message]!
            return (prompt, encode(prompt))
        }

        var probes: [(prompt: String, promptIDs: [Int])] {
            [probe("x"), probe("y")]
        }
    }

    private func kitLabels(
        _ tokenizer: FakeTokenizer,
        boundaryTokens: [Int: String] = [50: "<B>"],
        encode: ((String) -> [Int])? = nil
    ) -> [String]? {
        SwamaKit.decisionPairLabels(
            probes: tokenizer.probes, boundaryTokens: boundaryTokens, encode: encode ?? tokenizer.encode
        )
    }

    private func runtimeLabels(
        _ tokenizer: FakeTokenizer,
        boundaryTokens: [Int: String] = [50: "<B>"],
        encode: ((String) -> [Int])? = nil
    ) -> [String]? {
        SwamaRuntime.decisionPairLabels(
            probes: tokenizer.probes, boundaryTokens: boundaryTokens, encode: encode ?? tokenizer.encode
        )
    }

    @Test func labelsFollowPairOrderAndSkipMultiTokenAndDuplicateLabels() throws {
        var tokenizer = FakeTokenizer()
        tokenizer.labels["AC"] = [7, 8]
        tokenizer.labels["BA"] = [100]
        tokenizer.labels["ZZ"] = [101]
        let labels = try #require(kitLabels(tokenizer))
        #expect(labels.count == 26 * 26 - 3)
        #expect(Array(labels.prefix(4)) == ["AA", "AB", "AD", "AE"])
        #expect(labels[labels.firstIndex(of: "AZ")! + 1] == "BB")
        #expect(!labels.contains("AC") && !labels.contains("BA") && !labels.contains("ZZ"))
        #expect(labels == Self.candidates.filter { !["AC", "BA", "ZZ"].contains($0) })
        #expect(runtimeLabels(tokenizer) == labels)
    }

    @Test func labelsAreUnavailableWithoutTheAddedTokenShortcut() {
        let tokenizer = FakeTokenizer()
        #expect(kitLabels(tokenizer)?.count == 676)
        #expect(runtimeLabels(tokenizer)?.count == 676)

        // No added token: the labels cannot be checked apart from the message.
        #expect(kitLabels(tokenizer, boundaryTokens: [:]) == nil)
        #expect(runtimeLabels(tokenizer, boundaryTokens: [:]) == nil)

        // The text after the added token does not tokenize the same on its own.
        let unstable: (String) -> [Int] = { $0 == "tail" ? [61] : tokenizer.encode($0) }
        #expect(kitLabels(tokenizer, encode: unstable) == nil)
        #expect(runtimeLabels(tokenizer, encode: unstable) == nil)

        // Both probes take the shortcut, but the text after the added token depends on the message.
        var dependent = FakeTokenizer()
        dependent.tails["tail2"] = [60, 62]
        dependent.probeTails["y"] = "tail2"
        #expect(kitLabels(dependent) == nil)
        #expect(runtimeLabels(dependent) == nil)

        // Identical probe texts are not enough without the shortcut.
        let repeated = [tokenizer.probe("x"), tokenizer.probe("x")]
        #expect(SwamaKit.decisionPairLabels(probes: repeated, boundaryTokens: [:], encode: tokenizer.encode) == nil)
        #expect(SwamaRuntime.decisionPairLabels(probes: repeated, boundaryTokens: [:], encode: tokenizer.encode) == nil)
    }

    @Test func tooFewLabelsOrNoLabelsAreRefusedWithoutTruncating() throws {
        let usable = Array(Self.candidates.prefix(30))
        #expect(try SwamaKit.decisionPairLabelPrefix(count: 30, usable: usable) == usable)
        #expect(try SwamaKit.decisionPairLabelPrefix(count: 27, usable: usable) == Array(usable.prefix(27)))
        #expect(try SwamaRuntime.decisionPairLabelPrefix(count: 27, usable: usable) == Array(usable.prefix(27)))

        let prefixes: [(Int, [String]?) throws -> [String]] = [
            SwamaKit.decisionPairLabelPrefix(count:usable:),
            SwamaRuntime.decisionPairLabelPrefix(count:usable:)
        ]
        for prefix in prefixes {
            do {
                _ = try prefix(31, usable)
                Issue.record("31 options were accepted with 30 labels")
            }
            catch {
                #expect(error is SwamaKit.DecisionScoringError || error is SwamaRuntime.DecisionScoringError)
                #expect(error.localizedDescription ==
                    "This model supports at most 30 options per choice; the question has 31."
                )
            }
            do {
                _ = try prefix(27, nil)
                Issue.record("Options were accepted without pair labels")
            }
            catch {
                #expect(error is SwamaKit.DecisionScoringError || error is SwamaRuntime.DecisionScoringError)
                #expect(error.localizedDescription ==
                    "More than 26 options per choice needs an added token before the answer position, " +
                    "which this tokenizer and chat template do not provide."
                )
            }
        }
    }

    @Test func labelsAreComputedOncePerContainer() async throws {
        let kit = SwamaKit.DecisionPairLabelCache()
        let runtime = SwamaRuntime.DecisionPairLabelCache()
        let calls = CallCounter()
        let compute: @Sendable () -> [String]? = {
            calls.increment()
            return ["AA", "AB"]
        }
        let first = makeLifetimeTestContainer()
        let second = makeLifetimeTestContainer()

        #expect(try await kit.labels(for: first, compute: compute) == ["AA", "AB"])
        #expect(try await kit.labels(for: first, compute: compute) == ["AA", "AB"])
        #expect(calls.count == 1)
        #expect(try await kit.labels(for: second, compute: compute) == ["AA", "AB"])
        #expect(calls.count == 2)

        // An unavailable result is a property of the container too.
        let unavailableCalls = CallCounter()
        let unavailable: @Sendable () -> [String]? = {
            unavailableCalls.increment()
            return nil
        }
        let third = makeLifetimeTestContainer()
        #expect(try await kit.labels(for: third, compute: unavailable) == nil)
        #expect(try await kit.labels(for: third, compute: unavailable) == nil)
        #expect(unavailableCalls.count == 1)

        let runtimeStart = calls.count
        _ = try await runtime.labels(for: first, compute: compute)
        _ = try await runtime.labels(for: first, compute: compute)
        _ = try await runtime.labels(for: second, compute: compute)
        #expect(calls.count == runtimeStart + 2)
    }

    @Test func shuffledOptionsKeepTheirLabelsAndNames() throws {
        // Option i gets label i of the usable set, with a gap at "AC", in request order.
        let names = [
            17,
            3,
            29,
            0,
            11,
            24,
            8,
            21,
            5,
            14,
            27,
            1,
            19,
            9,
            26,
            12,
            4,
            22,
            15,
            7,
            28,
            2,
            13,
            25,
            10,
            18,
            6,
            23,
            16,
            20
        ].map(String.init)
        let usable = Self.candidates.filter { $0 != "AC" }
        let labels = try SwamaKit.decisionPairLabelPrefix(count: names.count, usable: usable)
        let question = DecisionQuestion.choice(
            id: "c",
            question: "Which option is the number 0?",
            options: names.map { DecisionOption(name: $0, description: $0 == "3" ? "three" : nil) }
        )
        let runtime = RuntimeDecisionQuestion.choice(
            id: "c",
            question: "Which option is the number 0?",
            options: names.map { RuntimeDecisionOption(name: $0, description: $0 == "3" ? "three" : nil) }
        )
        #expect(question.decisionLabels.isEmpty)
        #expect(runtime.labels.isEmpty)

        let prompt = question.decisionPrompt(input: "state", labels: labels)
        let lines = prompt.components(separatedBy: "\n")
        try #require(lines.count == names.count + 4)
        #expect(lines[2] == "Question: Which option is the number 0?")
        #expect(lines[3] == "AA: 17")
        #expect(lines[4] == "AB: 3 - three")
        #expect(lines[5] == "AD: 29")
        #expect(lines[6] == "AE: 0")
        #expect(lines[3 + 29] == "BE: 20")
        #expect(lines.last == "Answer with the letter of one option only.")
        #expect(runtime.content(input: "state", labels: labels) == prompt)

        // Label i scores option i: the top label is the fourth, which is option "0".
        let logProbs = names.indices.map { $0 == 3 ? log(0.5) : log(0.01) }
        let answer = try question.decisionAnswer(
            logProbs: logProbs, temperature: 1, promptTokenIDs: nil, labelTokenIDs: nil
        )
        let runtimeAnswer = try SwamaRuntime.decisionAnswer(
            question: runtime, logProbs: logProbs, temperature: 1, promptTokenIDs: nil, labelTokenIDs: nil
        )
        #expect(answer.choice == "0")
        #expect(runtimeAnswer.choice == "0")
        #expect(Set(answer.probabilities.keys) == Set(names))
        #expect(answer.probabilities == runtimeAnswer.probabilities)
        #expect(answer.probabilities["0"]! > answer.probabilities["17"]!)
        #expect(answer.probabilities["17"] == answer.probabilities["20"])

        // A tie keeps the first maximum in request order.
        let tied = try question.decisionAnswer(
            logProbs: names.indices.map { $0 == 5 || $0 == 2 ? log(0.3) : log(0.01) },
            temperature: 1,
            promptTokenIDs: nil,
            labelTokenIDs: nil
        )
        #expect(tied.choice == "29")
    }

    @Test func onlySystemOneOptsIntoUpTo676OptionsAndDecisionsStillRefuse27() async throws {
        func systemOne(_ count: Int) throws -> SystemOneRequest {
            let criteria = (0 ..< count).map { "\"o\($0)\":null" }.joined(separator: ",")
            let body = #"{"model":"m","state":"s","questions":{"c":{"type":"choice","instructions":"Pick","criteria":{"#
                + criteria + "}}}}"
            return try SystemOneRequest.parse(Data(body.utf8))
        }
        let engine = SwamaEngine(backend: PairLabelTestBackend())
        for count in [26, 27, 151, 676] {
            let request = try systemOne(count)
            #expect(request.questions[0].names.count == count)
            #expect(request.questions[0].names.last == "o\(count - 1)")
            _ = try await engine.decide(request.decision)
        }
        do {
            _ = try systemOne(677)
            Issue.record("677 options were accepted")
        }
        catch {
            #expect(error.localizedDescription.contains("options per choice"))
        }
        #expect(try systemOne(27).decision.allowsPairLabels)
        // Library callers keep the 2-26 limit unless they opt in; the opt-in still stops at 676.
        func choice(_ count: Int, allowsPairLabels: Bool) -> DecisionRequest {
            let options = (0 ..< count).map { DecisionOption(name: "o\($0)") }
            return .init(
                model: .init("org/model"), input: "s",
                questions: [.choice(id: "c", question: "Pick", options: options)],
                allowsPairLabels: allowsPairLabels
            )
        }
        #expect(DecisionRequest(model: .init("org/model"), input: "s", questions: []).allowsPairLabels == false)
        for (count, allowsPairLabels, message) in [
            (27, false, "Choice questions require 2–26 distinct, valid options."),
            (677, true, "Choice questions require 2–676 distinct, valid options.")
        ] {
            do {
                _ = try await engine.decide(choice(count, allowsPairLabels: allowsPairLabels))
                Issue.record("\(count) options reached the backend")
            }
            catch let error as SwamaError {
                #expect(error.code == .invalidRequest)
                #expect(error.message == message)
            }
        }
        _ = try await engine.decide(choice(26, allowsPairLabels: false))

        func decisions(_ count: Int) throws -> DecisionRequest {
            let options = (0 ..< count).map { #"{"name":"o\#($0)"}"# }.joined(separator: ",")
            let body = #"{"model":"m","input":"s","questions":[{"id":"c","type":"choice","question":"Pick","options":["#
                + options + "]}]}"
            return try DecisionsHandler.parse(JSONDecoder().decode([String: JSONValue].self, from: Data(body.utf8)))
        }
        #expect(try decisions(26).questions.count == 1)
        do {
            _ = try decisions(27)
            Issue.record("/v1/decisions accepted 27 options")
        }
        catch {
            #expect(error.localizedDescription == "choice options must contain 2–26 entries.")
        }
    }
}

// MARK: - DecisionPairLabelModelTests

/// Both decision backends on a real checkpoint, starting from a cold model.
@Suite(
    "Decision pair labels on a local model",
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["SWAMA_TEST_DECISIONS_MODEL"] == "1")
)
struct DecisionPairLabelModelTests {
    private let model = ProcessInfo.processInfo.environment["SWAMA_PAIR_LABEL_MODEL"]
        ?? "mlx-community/Qwen3.5-0.8B-4bit"

    private func question(_ count: Int, target: Int) -> DecisionQuestion {
        // A fixed shuffle: 7 is coprime with every count used here.
        .choice(
            id: "c",
            question: "Which option is the number \(target)?",
            options: (0 ..< count).map { DecisionOption(name: String(($0 * 7) % count)) }
        )
    }

    @Test func bothBackendsLabelMoreThan26OptionsFromAColdModel() async throws {
        let pool = SwamaKit.ModelPool()
        let legacy = LegacyServerCoreBackend(modelPool: pool)
        let input = "Find the option whose name is exactly the number 21."
        let request = DecisionRequest(
            model: .init(model), input: input, questions: [question(30, target: 21)], returnPromptTokenIDs: true,
            allowsPairLabels: true
        )
        let cold = try await legacy.decide(request)
        let warm = try await legacy.decide(request)
        #expect(cold.answers == warm.answers)
        #expect(cold.answers["c"]?.choice == "21")

        let container = try #require(await pool.loadedContainer(modelName: model))
        let usable = try #require(try await SwamaKit.cachedDecisionPairLabels(container: container))
        #expect(usable.count > 26)
        let prompt = question(30, target: 21).decisionPrompt(input: input, labels: Array(usable.prefix(30)))
        print("pair labels for \(model): \(usable.count) usable; first \(usable.prefix(30))")
        print("30-option prompt:\n\(prompt)")

        do {
            _ = try await legacy.decide(.init(
                model: .init(model), input: input, questions: [question(usable.count + 1, target: 21)]
            ))
            Issue.record("More options than usable labels were scored")
        }
        catch let error as SwamaError {
            #expect(error.code == .invalidRequest)
            #expect(error.message.contains("options per choice"))
        }
        await pool.clearCache()

        let runtime = SwamaRuntime.RuntimeCoreEngine()
        let runtimeQuestion = RuntimeDecisionQuestion.choice(
            id: "c",
            question: "Which option is the number 21?",
            options: (0 ..< 30).map { RuntimeDecisionOption(name: String(($0 * 7) % 30), description: nil) }
        )
        let runtimeResult = try await runtime.decide(.init(
            model: model, input: input, questions: [runtimeQuestion], temperature: 1, returnPromptTokenIDs: true
        ))
        let runtimeAnswer = try #require(runtimeResult.answers["c"])
        #expect(runtimeAnswer.choice == "21")
        #expect(runtimeAnswer.probabilities == cold.answers["c"]?.probabilities)
        #expect(runtimeAnswer.promptTokenIDs == cold.answers["c"]?.promptTokenIDs)
        #expect(runtimeAnswer.labelTokenIDs == cold.answers["c"]?.labelTokenIDs)
        await runtime.clearCache()
    }
}

// MARK: - CallCounter

private final class CallCounter: Sendable {
    private let value: Mutex = .init(0)
    var count: Int { value.withLock { $0 } }
    func increment() { value.withLock { $0 += 1 } }
}

// MARK: - PairLabelTestBackend

private struct PairLabelTestBackend: SwamaEngineBackend {
    func decide(_ request: DecisionRequest) async throws -> DecisionResponse {
        .init(model: request.model, answers: [:], usage: .init(promptTokens: 0, completionTokens: 0))
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

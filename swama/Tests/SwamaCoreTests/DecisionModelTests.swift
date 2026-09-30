import Foundation
import SwamaCore
import Testing

/// Opt-in integration gate for the independently measured local Qwen checkpoint.
@Suite(
    "Decision local model",
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["SWAMA_TEST_DECISIONS_MODEL"] == "1")
)
struct DecisionModelTests {
    @Test func localQwenHasExpectedTokensAndDoesNotReuseChatCache() async throws {
        let model = ModelID("mlx-community/Qwen3.5-0.8B-MLX-4bit")
        let directory = FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent(".swama/models/\(model.rawValue)")
        try #require(FileManager.default.fileExists(atPath: directory.appendingPathComponent("config.json").path))
        let engine = SwamaEngine()
        let request = DecisionRequest(
            model: model,
            input: "The meeting ended ten minutes early because everyone agreed.",
            questions: [.yesNo(id: "early", question: "The meeting finished before its scheduled time.")],
            returnPromptTokenIDs: true
        )
        let first = try await engine.decide(request)
        let answer = try #require(first.answers["early"])
        #expect(answer.labelTokenIDs == [9405, 2083])
        #expect(answer.promptTokenIDs?.count == 44)
        // Numeric parity is checked against upstream Swift by DecisionRawLogitTests.
        // Python/Swift bf16 raw logits can differ by one representable step.
        #expect((0 ... 1).contains(answer.labelMass))
        #expect(abs(answer.probabilities.values.reduce(0, +) - 1) < 1e-12)
        #expect(answer.probabilities["yes"]! > answer.probabilities["no"]!)
        #expect(first.usage.completionTokens == 0)
        #expect(answer.confidence == nil)
        _ = try await engine.generate(.init(
            model: model, messages: [.init(role: .user, text: "Say hello.")],
            options: .init(maxTokens: 4, temperature: 0)
        ))
        let second = try await engine.decide(request)
        let repeated = try #require(second.answers["early"])
        #expect(abs(answer.labelMass - repeated.labelMass) < 1e-6)
        #expect(abs(answer.probabilities["yes"]! - repeated.probabilities["yes"]!) < 1e-6)
        let choiceResponse = try await engine.decide(.init(
            model: model,
            input: "My invoice charged me twice for the same subscription this month.",
            questions: [.choice(
                id: "team",
                question: "Which team should handle this?",
                options: [
                    .init(name: "billing"), .init(name: "technical"),
                    .init(name: "sales")
                ]
            )]
        ))
        let choice = try #require(choiceResponse.answers["team"])
        let confidence = try #require(choice.confidence)
        let maximum = try #require(choice.probabilities.values.max())
        #expect(abs(2 * confidence - (3 * maximum - 1)) < 1e-12)
        await engine.clearCache()
    }
}

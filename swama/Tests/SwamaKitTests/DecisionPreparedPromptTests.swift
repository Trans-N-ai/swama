import Foundation
@testable import SwamaKit
import Testing

// MARK: - DecisionPreparedPromptTests

/// Issue #162: the decision prompt may be tokenized before the exclusive model slot is acquired.
/// That work must be reused only for the container that produced it; anything else recomputes.
@Suite(
    "Decision prompt prepared before the model slot",
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["SWAMA_TEST_DECISIONS_MODEL"] == "1")
)
struct DecisionPreparedPromptTests {
    private let model = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
    private let labels = ["A", "B"]
    private let contentA = "Choose between A and B. The invoice was charged twice. Answer with one letter only."
    private let contentB = "Choose between A and B. The app crashes on launch. Answer with one letter only."
    /// A private pool, so clearing it never tears down another suite's work on the shared pool.
    private let pool: ModelPool = .init()

    @Test func preparedPromptMatchesTheInSlotComputation() async throws {
        let cold = try await score(contentA, prepared: nil)
        let container = try #require(await pool.loadedContainer(modelName: model))
        let prepared = try await prepareDecisionPrompt(container: container, content: contentA, contextLimit: 4096)
        #expect(prepared.promptIDs == cold.promptTokenIDs)

        let warm = try await score(contentA, prepared: prepared)
        #expect(warm.promptTokenIDs == cold.promptTokenIDs)
        #expect(warm.labelTokenIDs == cold.labelTokenIDs)
        #expect(warm.labelLogProbs == cold.labelLogProbs)

        await #expect(throws: DecisionScoringError.self) {
            _ = try await prepareDecisionPrompt(container: container, content: contentA, contextLimit: 8)
        }
        await pool.clearCache()
    }

    /// A prepared prompt carrying B's tokens under A's content shows which path ran: the reuse path
    /// returns B's tokens, the recompute path returns A's.
    @Test func preparedPromptIsUsedOnlyForTheSameContainer() async throws {
        let coldA = try await score(contentA, prepared: nil)
        let containerBefore = try #require(await pool.loadedContainer(modelName: model))
        let realB = try await prepareDecisionPrompt(container: containerBefore, content: contentB, contextLimit: 4096)
        let disguised = PreparedDecisionPrompt(
            container: containerBefore,
            content: contentA,
            contextLimit: 4096,
            prompt: realB.prompt,
            promptIDs: realB.promptIDs
        )

        let sameContainer = try await score(contentA, prepared: disguised)
        #expect(sameContainer.promptTokenIDs == realB.promptIDs)

        let otherLimit = try await score(contentA, prepared: disguised, contextLimit: 4095)
        #expect(otherLimit.promptTokenIDs == coldA.promptTokenIDs)

        // Remove and reload the model: the slot now runs on a different container object.
        await pool.clearCache()
        let reloaded = try await score(contentA, prepared: disguised)
        let containerAfter = try #require(await pool.loadedContainer(modelName: model))
        #expect(containerAfter !== containerBefore)
        #expect(reloaded.promptTokenIDs == coldA.promptTokenIDs)
        #expect(reloaded.labelLogProbs == coldA.labelLogProbs)
        await pool.clearCache()
    }

    private func score(
        _ content: String, prepared: PreparedDecisionPrompt?, contextLimit: Int = 4096
    ) async throws -> DecisionLogits {
        let labels = labels
        return try await pool.run(modelName: model) { runner in
            try await runner.scoreDecision(
                content: content, labels: labels, contextLimit: contextLimit, prepared: prepared
            )
        }
    }
}

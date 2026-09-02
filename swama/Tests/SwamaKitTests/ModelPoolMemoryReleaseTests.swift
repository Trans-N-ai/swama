import Foundation
@preconcurrency import MLXLMCommon
@testable import SwamaKit
import Testing

// MARK: - ModelPoolMemoryReleaseTests

@Suite("ModelPool releases owners before MLX cleanup")
struct ModelPoolMemoryReleaseTests {
    @Test func clearCacheDropsTheContainerBeforeMemoryCleanupRuns() async throws {
        let witness = ContainerReleaseWitness()
        let pool = makePool(witness: witness)
        var container: MLXLMCommon.ModelContainer? = makeLifetimeTestContainer()
        witness.track(container)
        try await pool.cacheContainerForTesting(#require(container), modelName: "clear-all")
        container = nil

        #expect(witness.isContainerAlive, "the pool must be the remaining owner before clear")
        await pool.clearCache()

        assertReleased(witness)
    }

    @Test func removeDropsTheNamedContainerBeforeMemoryCleanupRuns() async throws {
        let witness = ContainerReleaseWitness()
        let pool = makePool(witness: witness)
        var container: MLXLMCommon.ModelContainer? = makeLifetimeTestContainer()
        witness.track(container)
        try await pool.cacheContainerForTesting(#require(container), modelName: "one-model")
        container = nil

        #expect(witness.isContainerAlive)
        await pool.remove(modelName: "one-model")

        assertReleased(witness)
    }

    @Test func periodicEvictionDropsTheNamedContainerBeforeMemoryCleanupRuns() async throws {
        let witness = ContainerReleaseWitness()
        let pool = makePool(witness: witness)
        var container: MLXLMCommon.ModelContainer? = makeLifetimeTestContainer()
        witness.track(container)
        try await pool.cacheContainerForTesting(#require(container), modelName: "idle-model")
        container = nil

        #expect(witness.isContainerAlive)
        await pool.evictModelForTesting(modelName: "idle-model")

        assertReleased(witness)
    }

    private func makePool(witness: ContainerReleaseWitness) -> ModelPool {
        ModelPool(
            memoryHooks: .init(
                activeMemory: { 0 },
                clearCache: { witness.recordCleanup() }
            )
        )
    }

    private func assertReleased(_ witness: ContainerReleaseWitness) {
        #expect(witness.cleanupCallCount == 1)
        #expect(witness.wasReleasedAtCleanup)
        #expect(!witness.isContainerAlive)
    }
}

// MARK: - ContainerReleaseWitness

private final class ContainerReleaseWitness: @unchecked Sendable {
    private let lock: NSLock = .init()
    private weak var container: MLXLMCommon.ModelContainer?
    private var releasedAtCleanup = false
    private var cleanupCalls = 0

    func track(_ container: MLXLMCommon.ModelContainer?) {
        lock.withLock {
            self.container = container
        }
    }

    func recordCleanup() {
        lock.withLock {
            cleanupCalls += 1
            releasedAtCleanup = container == nil
        }
    }

    var isContainerAlive: Bool {
        lock.withLock { container != nil }
    }

    var wasReleasedAtCleanup: Bool {
        lock.withLock { releasedAtCleanup }
    }

    var cleanupCallCount: Int {
        lock.withLock { cleanupCalls }
    }
}

//
//  ModelPoolSlotAdmissionTests.swift
//  SwamaRuntimeTests
//

import Foundation
@preconcurrency import MLXLMCommon
@testable import SwamaRuntime
import Testing

// MARK: - ModelPoolSlotAdmissionTests

/// `ModelPool.run` admits waiting callers in arrival order. It used to poll every 50 ms, which
/// left the GPU idle between requests and let a waiter lose the race for the slot again and
/// again: with 16 concurrent decision requests the slowest one waited 4.6 s while the median
/// was 59 ms.
///
/// Every operation here is a closure that never touches the runner, so no model ever runs. The
/// container is only there so `getContainer` returns from the cache.
@Suite("ModelPool slot admission", .timeLimit(.minutes(1)))
struct ModelPoolSlotAdmissionTests {
    @Test func sameModelWaitersRunInArrivalOrder() async throws {
        let pool = await makePool(models: ["m"])
        let order = OrderRecorder()
        let holder = Gate()

        let first = Task { try await pool.run(modelName: "m") { _ in
            await order.append(0)
            await holder.wait()
        } }
        try await waitUntil { await order.values == [0] }

        var waiters: [Task<Void, Error>] = []
        for index in 1 ... 8 {
            waiters.append(Task { try await pool.run(modelName: "m") { _ in await order.append(index) } })
            try await waitUntil { await pool.slotWaiterCountForTesting() == index }
        }

        await holder.open()
        try await first.value
        for waiter in waiters {
            try await waiter.value
        }
        #expect(await order.values == Array(0 ... 8))
        #expect(await pool.runningInferenceCountForTesting() == 0)
    }

    @Test func releasingASlotHandsItToTheWaiterImmediately() async throws {
        let pool = await makePool(models: ["m"])
        let holder = Gate()
        let second = Gate()

        let first = Task { try await pool.run(modelName: "m") { _ in await holder.wait() } }
        try await waitUntil { await pool.runningInferenceCountForTesting() == 1 }
        let queued = Task { try await pool.run(modelName: "m") { _ in await second.wait() } }
        try await waitUntil { await pool.slotWaiterCountForTesting() == 1 }

        await holder.open()
        try await first.value
        // The slot is granted inside the release, before `first` returns: nothing is left waiting
        // and the slot is already taken again. The polling loop left the waiter asleep for up to
        // 50 ms here, so this held only by luck.
        #expect(await pool.slotWaiterCountForTesting() == 0)
        #expect(await pool.runningInferenceCountForTesting() == 1)

        await second.open()
        try await queued.value
        #expect(await pool.runningInferenceCountForTesting() == 0)
    }

    @Test func busyModelDoesNotBlockOtherModels() async throws {
        let pool = await makePool(models: ["busy", "free"])
        let holder = Gate()
        let order = OrderRecorder()

        let first = Task { try await pool.run(modelName: "busy") { _ in await holder.wait() } }
        try await waitUntil { await pool.runningInferenceCountForTesting() == 1 }
        let queued = Task { try await pool.run(modelName: "busy") { _ in await order.append(1) } }
        try await waitUntil { await pool.slotWaiterCountForTesting() == 1 }

        try await pool.run(modelName: "free") { _ in await order.append(2) }
        #expect(await order.values == [2])

        await holder.open()
        try await first.value
        try await queued.value
        #expect(await order.values == [2, 1])
    }

    @Test func globalLimitAdmitsTheOldestWaiterWhenASlotFrees() async throws {
        let pool = await makePool(models: ["a", "b", "c", "d", "e"])
        let holders = ["a", "b", "c", "d", "e"].reduce(into: [String: Gate]()) { $0[$1] = Gate() }
        let order = OrderRecorder()

        var tasks: [Task<Void, Error>] = []
        for (index, model) in ["a", "b", "c"].enumerated() {
            tasks.append(Task { try await pool.run(modelName: model) { _ in
                await order.append(index + 1)
                await holders[model]!.wait()
            } })
        }
        try await waitUntil { await pool.runningInferenceCountForTesting() == 3 }

        for (index, model) in ["d", "e"].enumerated() {
            tasks.append(Task { try await pool.run(modelName: model) { _ in
                await order.append(index + 4)
                await holders[model]!.wait()
            } })
            try await waitUntil { await pool.slotWaiterCountForTesting() == index + 1 }
        }
        #expect(await order.values.sorted() == [1, 2, 3])

        // One slot frees: only the older of the two waiters may take it.
        await holders["b"]!.open()
        try await waitUntil { await order.values.count == 4 }
        #expect(await order.values.last == 4)
        #expect(await pool.slotWaiterCountForTesting() == 1)
        #expect(await pool.runningInferenceCountForTesting() == 3)

        await holders["a"]!.open()
        try await waitUntil { await order.values.count == 5 }
        #expect(await order.values.last == 5)

        for gate in holders.values {
            await gate.open()
        }
        for task in tasks {
            try await task.value
        }
        #expect(await pool.runningInferenceCountForTesting() == 0)
    }

    @Test func cancelledWaiterLeavesTheQueueAndOthersStillRun() async throws {
        let pool = await makePool(models: ["m"])
        let holder = Gate()
        let order = OrderRecorder()

        let first = Task { try await pool.run(modelName: "m") { _ in await holder.wait() } }
        try await waitUntil { await pool.runningInferenceCountForTesting() == 1 }
        let cancelled = Task { try await pool.run(modelName: "m") { _ in await order.append(1) } }
        try await waitUntil { await pool.slotWaiterCountForTesting() == 1 }
        let kept = Task { try await pool.run(modelName: "m") { _ in await order.append(2) } }
        try await waitUntil { await pool.slotWaiterCountForTesting() == 2 }

        cancelled.cancel()
        await #expect(throws: CancellationError.self) { try await cancelled.value }
        #expect(await pool.slotWaiterCountForTesting() == 1)

        await holder.open()
        try await first.value
        try await kept.value
        #expect(await order.values == [2])
        #expect(await pool.runningInferenceCountForTesting() == 0)
    }

    @Test func embeddingWaitersShareTheGlobalQueue() async throws {
        let pool = await makePool(models: ["a", "b", "c"])
        let holders = [Gate(), Gate(), Gate()]

        var running: [Task<Void, Error>] = []
        for (model, holder) in zip(["a", "b", "c"], holders) {
            running.append(Task { try await pool.run(modelName: model) { _ in await holder.wait() } })
        }
        try await waitUntil { await pool.runningInferenceCountForTesting() == 3 }

        let embedding = Task {
            try await pool.runEmbeddingWithConcurrencyControl(modelName: "missing-embedding") { _ in "unused" }
        }
        try await waitUntil { await pool.slotWaiterCountForTesting() == 1 }

        await holders[0].open()
        // Admitted, then fails at the loader: the slot must still come back.
        await #expect(throws: (any Error).self) { try await embedding.value }
        await holders[1].open()
        await holders[2].open()
        for task in running {
            try await task.value
        }
        #expect(await pool.runningInferenceCountForTesting() == 0)
        #expect(await pool.slotWaiterCountForTesting() == 0)
    }

    private func makePool(models: [String]) async -> ModelPool {
        let pool = ModelPool(
            memoryHooks: .init(activeMemory: { 0 }, clearCache: {}),
            loadOverrides: .init(
                modelExistsLocally: { _ in false },
                determineIsVLM: { _ in false },
                loadLanguage: { _, _ in fatalError("unexpected language load") },
                loadEmbedding: { _ in throw ModelPoolError.modelNotFoundLocally("missing-embedding") }
            )
        )
        let container = makeTestContainer().container
        for model in models {
            await pool.cacheContainerForTesting(container, modelName: model)
        }
        return pool
    }
}

// MARK: - OrderRecorder

private actor OrderRecorder {
    private(set) var values: [Int] = []

    func append(_ value: Int) {
        values.append(value)
    }
}

// MARK: - Gate

/// Holds an operation open until the test releases it.
private actor Gate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        if isOpen {
            return
        }

        await withCheckedContinuation { waiters.append($0) }
    }

    func open() {
        isOpen = true
        waiters.forEach { $0.resume() }
        waiters.removeAll()
    }
}

// MARK: - WaitTimedOut

private struct WaitTimedOut: Error {}

private func waitUntil(
    timeout: Duration = .seconds(5),
    _ condition: @Sendable () async -> Bool
) async throws {
    let deadline = ContinuousClock.now + timeout
    while await condition() == false {
        guard ContinuousClock.now < deadline else {
            throw WaitTimedOut()
        }

        try await Task.sleep(for: .milliseconds(1))
    }
}

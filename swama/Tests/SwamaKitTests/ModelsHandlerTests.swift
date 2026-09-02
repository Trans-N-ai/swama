import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
@testable import SwamaKit
@testable import SwamaServer
import Testing

// MARK: - ModelsHandlerTests

@MainActor @Suite(.serialized)
final class ModelsHandlerTests {
    // MARK: - Bounded Scan Tests

    @Test func coalescedRunReturnsCompletedResult() async {
        let result = await ModelsHandler.coalescedRun(key: "coalesced-run-fast-\(UUID().uuidString)", seconds: 5) {
            42
        }

        #expect(result == 42)
    }

    @Test func coalescedRunReturnsNilForBlockedOperation() async {
        let key = "coalesced-run-blocked-\(UUID().uuidString)"

        let result = await ModelsHandler.coalescedRun(key: key, seconds: 0.1) { () -> Int in
            Thread.sleep(forTimeInterval: 2)
            return 42
        }

        #expect(result == nil)
    }

    // MARK: - Coalescing Tests

    /// Regression test for the review callout: the old design gave `runWithTimeout`
    /// a queue per root but still enqueued one operation per *call*, so a root
    /// stuck behind a TCC prompt accumulated an unbounded backlog of closures —
    /// one per timed-out request — that all ran, sequentially, the moment access
    /// recovered. `coalescedRun` must instead let at most one operation per key
    /// be in flight at a time: every caller that arrives while a key is already
    /// running must attach to that run instead of starting (or enqueueing)
    /// another, and a caller's own timeout must only detach that caller, never
    /// the run itself.
    ///
    /// Phase 1 below sends 8 concurrent callers against an operation blocked on
    /// a semaphore, with a short per-call timeout: since the operation cannot
    /// return until the test signals it, every caller that resolves before that
    /// signal must have done so via its own timeout, so seeing exactly one
    /// invocation proves the other 7 attached as waiters rather than each
    /// starting (or queueing) their own run.
    @Test func coalescedRunCollapsesConcurrentTimedOutCallersIntoOneInvocation() async {
        let key = "coalesced-run-coalescing-\(UUID().uuidString)"
        let invocationCount = InvocationCounter()
        let firstRunGate = DispatchSemaphore(value: 0)

        let waiterCount = 8
        let firstWaveResults = await withTaskGroup(of: Int?.self) { group in
            for _ in 0 ..< waiterCount {
                group.addTask {
                    await ModelsHandler.coalescedRun(key: key, seconds: 0.1) {
                        invocationCount.increment()
                        firstRunGate.wait()
                        return 1
                    }
                }
            }
            var collected = [Int?]()
            for await result in group {
                collected.append(result)
            }
            return collected
        }

        // Every caller's own timeout fired — the shared operation is still
        // blocked on `firstRunGate`, so none of them could have received a
        // real result yet.
        #expect(firstWaveResults.count == waiterCount)
        #expect(firstWaveResults.allSatisfy { $0 == nil })
        #expect(invocationCount.value == 1)

        // Release the one blocked operation and give its completion path a
        // moment to run: this returns the key to idle by itself, with no
        // backlog left to drain, even though every waiter above already gave
        // up on it.
        firstRunGate.signal()
        try? await Task.sleep(nanoseconds: 300_000_000)

        // Phase 2: a second wave, arriving after that run has completed, must
        // start exactly one new run and see its real result — proving there
        // is neither a permanent lockout on the key nor a leftover stale scan
        // still queued to run. Its own operation blocks on a second gate,
        // released only after a short delay, so the 3 callers that attach as
        // waiters are guaranteed to have arrived before the run finishes.
        let secondWaveCount = 4
        let secondRunGate = DispatchSemaphore(value: 0)

        async let releaseSecondRun: Void = {
            try? await Task.sleep(nanoseconds: 200_000_000)
            secondRunGate.signal()
        }()

        let secondWaveResults = await withTaskGroup(of: Int?.self) { group in
            for _ in 0 ..< secondWaveCount {
                group.addTask {
                    await ModelsHandler.coalescedRun(key: key, seconds: 5) {
                        invocationCount.increment()
                        secondRunGate.wait()
                        return 2
                    }
                }
            }
            var collected = [Int?]()
            for await result in group {
                collected.append(result)
            }
            return collected
        }
        _ = await releaseSecondRun

        #expect(secondWaveResults == Array(repeating: 2, count: secondWaveCount))
        #expect(invocationCount.value == 2)
    }

    // MARK: - Route Tests

    /// The route must answer without any synchronous work on the event loop: the
    /// response is produced by a task the handler dispatches, not by
    /// `channelRead` itself.
    @Test func modelsRouteResponds() async throws {
        let channel = NIOAsyncTestingChannel()
        try await channel.pipeline.addHandler(HTTPHandler()).get()

        let head = HTTPRequestHead(version: .http1_1, method: .GET, uri: "/v1/models")
        try await channel.writeInbound(HTTPServerRequestPart.head(head))
        try await channel.writeInbound(HTTPServerRequestPart.end(nil))
        await channel.testingEventLoop.run()

        var responseHead: HTTPServerResponsePart?
        for _ in 0 ..< 200 {
            await Task.yield()
            await channel.testingEventLoop.run()
            if let outbound = try await channel.readOutbound(as: HTTPServerResponsePart.self) {
                responseHead = outbound
                break
            }
        }

        guard case let .head(response) = responseHead else {
            Issue.record("Expected a response head, got \(String(describing: responseHead))")
            return
        }

        #expect(response.status == .ok)
        #expect(response.headers.first(name: "Content-Type") == "application/json")
    }
}

// MARK: - InvocationCounter

/// Thread-safe invocation counter shared across the coalescing tests above.
private final class InvocationCounter: @unchecked Sendable {
    func increment() {
        lock.lock()
        storedValue += 1
        lock.unlock()
    }

    var value: Int {
        lock.lock()
        defer { lock.unlock() }
        return storedValue
    }

    private let lock: NSLock = .init()
    private var storedValue = 0
}

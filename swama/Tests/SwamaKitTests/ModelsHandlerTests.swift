import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
@testable import SwamaKit
import Testing

// MARK: - ModelsHandlerTests

@MainActor @Suite(.serialized)
final class ModelsHandlerTests {
    // MARK: - Bounded Scan Tests

    @Test func runWithTimeoutReturnsCompletedResult() async {
        let queue = DispatchQueue(label: "test.models-scan.fast")

        let result = await ModelsHandler.runWithTimeout(on: queue, seconds: 5) {
            42
        }

        #expect(result == 42)
    }

    @Test func runWithTimeoutReturnsNilForBlockedOperation() async {
        let queue = DispatchQueue(label: "test.models-scan.blocked")

        let result = await ModelsHandler.runWithTimeout(on: queue, seconds: 0.1) { () -> Int in
            Thread.sleep(forTimeInterval: 2)
            return 42
        }

        #expect(result == nil)
    }

    @Test func runWithTimeoutQueuesBehindBlockedOperation() async {
        // A second operation on the same serial queue waits behind a blocked one
        // and times out rather than occupying another thread.
        let queue = DispatchQueue(label: "test.models-scan.queued")
        queue.async {
            Thread.sleep(forTimeInterval: 2)
        }

        let result = await ModelsHandler.runWithTimeout(on: queue, seconds: 0.1) {
            42
        }

        #expect(result == nil)
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

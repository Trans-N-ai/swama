import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
import NIOPosix
@testable import SwamaServer
import Testing

// MARK: - ConnectionPipelineTests

/// #158: a client disconnect must cancel a non-streaming request on a real socket. The
/// `runCancellingOnClose` tests use test channels, which cannot show whether the production
/// pipeline notices a peer close while a request is still running.
@Suite("Connection pipeline", .serialized)
struct ConnectionPipelineTests {
    @Test func realSocketDisconnectCancelsNonStreamingRequest() async throws {
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let probe = DisconnectProbe()
        let server = try await ServerBootstrap(group: group)
            .childChannelInitializer { channel in
                ServerManager.configureConnectionPipeline(channel, handler: SlowNonStreamingHandler(probe: probe))
            }
            .bind(host: "127.0.0.1", port: 0)
            .get()
        let result: Result<Void, Error>
        do {
            try await abandonSlowRequest(server: server, group: group, probe: probe)
            result = .success(())
        }
        catch {
            result = .failure(error)
        }
        try? await server.close()
        try? await group.shutdownGracefully()
        try result.get()
    }

    private func abandonSlowRequest(server: Channel, group: EventLoopGroup, probe: DisconnectProbe) async throws {
        let port = try #require(server.localAddress?.port)
        let client = try await ClientBootstrap(group: group).connect(host: "127.0.0.1", port: port).get()
        let request = "POST /slow HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}"
        try await client.writeAndFlush(ByteBuffer(string: request))
        #expect(await probe.waitUntilStarted(seconds: 5))

        let closedAt = ContinuousClock.now
        try await client.close()

        // The handler would run for 30 s if the disconnect went unnoticed.
        let outcome = await probe.waitForOutcome(seconds: 10)
        #expect(outcome == .cancelled)
        #expect(ContinuousClock.now - closedAt < .seconds(5))
    }

    @Test func sequentialKeepAliveRequestsAreAllForwarded() throws {
        let channel = try connectedGuardChannel()
        try channel.writeInbound(HTTPServerRequestPart.head(head()))
        try channel.writeInbound(HTTPServerRequestPart.end(nil))
        try channel.writeOutbound(HTTPServerResponsePart.head(.init(version: .http1_1, status: .ok)))
        try channel.writeOutbound(HTTPServerResponsePart.end(nil))

        try channel.writeInbound(HTTPServerRequestPart.head(head()))
        try channel.writeInbound(HTTPServerRequestPart.end(nil))

        #expect(try forwardedHeadCount(channel) == 2)
        #expect(channel.isActive)
        _ = try? channel.finish()
    }

    @Test func pipelinedRequestIsDroppedAndConnectionClosesAfterResponse() throws {
        let channel = try connectedGuardChannel()
        try channel.writeInbound(HTTPServerRequestPart.head(head()))
        try channel.writeInbound(HTTPServerRequestPart.end(nil))
        // Second request arrives before the first response has ended.
        try channel.writeInbound(HTTPServerRequestPart.head(head()))
        try channel.writeInbound(HTTPServerRequestPart.body(ByteBuffer(string: "ignored")))
        try channel.writeInbound(HTTPServerRequestPart.end(nil))

        #expect(try forwardedHeadCount(channel) == 1)
        #expect(channel.isActive)

        try channel.writeOutbound(HTTPServerResponsePart.head(.init(version: .http1_1, status: .ok)))
        try channel.writeOutbound(HTTPServerResponsePart.end(nil))
        channel.embeddedEventLoop.run()
        #expect(channel.isActive == false)
    }

    private func connectedGuardChannel() throws -> EmbeddedChannel {
        let channel = EmbeddedChannel(handler: HTTPPipeliningGuard())
        try channel.connect(to: SocketAddress(ipAddress: "127.0.0.1", port: 1)).wait()
        #expect(channel.isActive)
        return channel
    }

    private func head() -> HTTPRequestHead {
        HTTPRequestHead(version: .http1_1, method: .GET, uri: "/v1/models")
    }

    private func forwardedHeadCount(_ channel: EmbeddedChannel) throws -> Int {
        var heads = 0
        while let part = try channel.readInbound(as: HTTPServerRequestPart.self) {
            if case .head = part {
                heads += 1
            }
        }
        return heads
    }
}

// MARK: - DisconnectProbe

private actor DisconnectProbe {
    enum Outcome: Equatable {
        case cancelled
        case completed
    }

    private var started = false
    private var outcome: Outcome?

    func markStarted() {
        started = true
    }

    func finish(_ value: Outcome) {
        outcome = value
    }

    func waitUntilStarted(seconds: Int) async -> Bool {
        let deadline = ContinuousClock.now + .seconds(seconds)
        while started == false, ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(10))
        }
        return started
    }

    func waitForOutcome(seconds: Int) async -> Outcome? {
        let deadline = ContinuousClock.now + .seconds(seconds)
        while outcome == nil, ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(10))
        }
        return outcome
    }
}

// MARK: - SlowNonStreamingHandler

/// Behaves like a non-streaming endpoint: nothing is written until the work finishes, so the
/// only way to learn about a disconnect is to keep reading from the socket.
private final class SlowNonStreamingHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let probe: DisconnectProbe

    init(probe: DisconnectProbe) {
        self.probe = probe
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        guard case .end = unwrapInboundIn(data) else {
            return
        }

        let channel = context.channel
        let probe = probe
        Task {
            do {
                try await CompletionsHandler.runCancellingOnClose(channel: channel) {
                    await probe.markStarted()
                    try await Task.sleep(for: .seconds(30))
                }
                await probe.finish(.completed)
            }
            catch {
                await probe.finish(.cancelled)
            }
        }
    }
}

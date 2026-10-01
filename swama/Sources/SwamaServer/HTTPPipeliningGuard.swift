//
//  HTTPPipeliningGuard.swift
//  SwamaServer
//

import NIOCore
import NIOHTTP1

// MARK: - HTTPPipeliningGuard

/// Allows one request in flight per connection while still reading from the socket.
///
/// NIO's `HTTPServerPipelineHandler` keeps pipelined responses in order by not reading from the
/// socket until the current response has ended. That also hides a client disconnect from a
/// non-streaming request until its response is written, so abandoned work ran to completion and
/// held the model slot (#158). Without that handler the socket keeps being read, the peer's
/// FIN/RST closes the channel, and `channel.closeFuture` cancels the request.
///
/// HTTP/1.1 pipelining is rarely used. If a second request arrives while one is still in
/// flight, its parts are dropped and the connection is closed once the current response ends;
/// RFC 9112 section 9.3.2 lets a server close instead of answering pipelined requests, and
/// clients retry requests that were not answered. Dropping, rather than buffering, also keeps a
/// client from making the server hold unbounded pipelined data.
final class HTTPPipeliningGuard: ChannelDuplexHandler, RemovableChannelHandler {
    typealias InboundIn = HTTPServerRequestPart
    typealias InboundOut = HTTPServerRequestPart
    typealias OutboundIn = HTTPServerResponsePart
    typealias OutboundOut = HTTPServerResponsePart

    /// True from a request head until that request's response `.end` has been written.
    private var requestInFlight = false

    /// Set when a pipelined request arrives; every later request part is dropped and the
    /// connection closes after the in-flight response.
    private var closeAfterResponse = false

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        guard closeAfterResponse == false else {
            return
        }

        if case .head = unwrapInboundIn(data) {
            guard requestInFlight == false else {
                closeAfterResponse = true
                return
            }

            requestInFlight = true
        }
        context.fireChannelRead(data)
    }

    func write(context: ChannelHandlerContext, data: NIOAny, promise: EventLoopPromise<Void>?) {
        guard case .end = unwrapOutboundIn(data) else {
            context.write(data, promise: promise)
            return
        }

        requestInFlight = false
        guard closeAfterResponse else {
            context.write(data, promise: promise)
            return
        }

        let written = promise ?? context.eventLoop.makePromise(of: Void.self)
        let channel = context.channel
        written.futureResult.whenComplete { _ in
            channel.close(promise: nil)
        }
        context.write(data, promise: written)
    }
}

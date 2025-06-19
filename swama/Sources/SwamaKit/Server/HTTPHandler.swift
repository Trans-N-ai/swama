//
//  HTTPHandler.swift
//  SwamaKit
//

import Foundation
import MLXLMCommon
import NIOCore
import NIOHTTP1

/// This modelPool instance should ideally be managed and injected by ServerManager or a DI system.
/// For now, keeping it global within SwamaKit as per original Router.swift structure.
let modelPool: ModelPool = .shared // Use the same shared instance as CompletionsHandler

// MARK: - HTTPHandler

public final class HTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    // MARK: Lifecycle

    public init() {}

    // MARK: Public

    public typealias InboundIn = HTTPServerRequestPart
    public typealias OutboundOut = HTTPServerResponsePart

    public func channelActive(context: ChannelHandlerContext) {
        buffer = context.channel.allocator.buffer(capacity: 1024)
        NSLog("SwamaKit.HTTPHandler: Channel active")
    }

    public func channelInactive(context _: ChannelHandlerContext) {
        NSLog("SwamaKit.HTTPHandler: Channel inactive")
    }

    public func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let reqPart = self.unwrapInboundIn(data)
        switch reqPart {
        case let .head(request):
            self.requestHead = request
            bodyBuffer.clear()

        case var .body(chunk):
            bodyBuffer.writeBuffer(&chunk)

        case .end:
            guard let request = requestHead else {
                NSLog("SwamaKit.HTTPHandler Error: Request end received without request head.")
                // Optionally close context or send error response
                return
            }

            handleRequest(context: context, request: request, bodyBuffer: bodyBuffer)
            self.requestHead = nil // Reset for next request on persistent connection
        }
    }

    public func errorCaught(context: ChannelHandlerContext, error: Error) {
        NSLog("❌ SwamaKit.HTTPHandler Error caught: \(error)")
        context.close(promise: nil)
    }

    // MARK: Private

    private var buffer: ByteBuffer!
    private var requestHead: HTTPRequestHead?
    private var bodyBuffer: ByteBuffer = .init()

    private func handleRequest(context: ChannelHandlerContext, request: HTTPRequestHead, bodyBuffer: ByteBuffer) {
        NSLog("SwamaKit.HTTPHandler: Handling request: \(request.method) \(request.uri)")
        switch (request.method, request.uri) {
        case (.GET, "/v1/models"):
            // Use the logic similar to the old ModelsHandler
            handleModelsRequest(context: context, request: request)

        case (.POST, "/v1/chat/completions"):
            let channel = context.channel
            channel.eventLoop.execute {
                Task {
                    await CompletionsHandler.handle(
                        requestHead: request,
                        body: bodyBuffer,
                        channel: channel
                    )
                }
            }

        case (.POST, "/v1/embeddings"):
            let channel = context.channel
            channel.eventLoop.execute {
                Task {
                    await EmbeddingsHandler.handle(
                        requestHead: request,
                        body: bodyBuffer,
                        channel: channel
                    )
                }
            }

        case (.POST, "/v1/audio/transcriptions"):
            let channel = context.channel
            channel.eventLoop.execute {
                Task {
                    await TranscriptionsHandler.handle(
                        requestHead: request,
                        body: bodyBuffer,
                        channel: channel
                    )
                }
            }

        default:
            respond404(context: context, request: request)
        }
    }

    /// New method based on the old ModelsHandler logic
    private func handleModelsRequest(context: ChannelHandlerContext, request: HTTPRequestHead) {
        do {
            // Generate model list (logic from old ModelsHandler)
            let modelsList = ModelManager.models().map { model_info -> [String: Any] in
                return [
                    "id": model_info.id,
                    "object": "model",
                    "created": model_info.created,
                    "owned_by": "swama",
                    "size_in_bytes": model_info.sizeInBytes
                ]
            }

            let responsePayload: [String: Any] = [
                "object": "list",
                "data": modelsList
            ]

            let jsonData = try JSONSerialization.data(withJSONObject: responsePayload)

            var responseBuffer = context.channel.allocator.buffer(capacity: jsonData.count)
            responseBuffer.writeBytes(jsonData)

            var headers = HTTPHeaders()
            headers.add(name: "Content-Type", value: "application/json")
            headers.add(name: "Content-Length", value: "\(responseBuffer.readableBytes)")
            headers.add(name: "Connection", value: "close") // Or keep-alive based on request

            context.write(
                self.wrapOutboundOut(.head(HTTPResponseHead(version: request.version, status: .ok, headers: headers))),
                promise: nil
            )
            context.write(self.wrapOutboundOut(.body(.byteBuffer(responseBuffer))), promise: nil)
            context.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
        }
        catch {
            NSLog("SwamaKit.HTTPHandler Error (handleModelsRequest): Failed to process models request - \(error)")
            // Send an error response
            var errorBuffer = context.channel.allocator.buffer(capacity: 128)
            errorBuffer.writeString("Internal Server Error: Could not process model list.")
            var headers = HTTPHeaders()
            headers.add(name: "Content-Length", value: "\(errorBuffer.readableBytes)")
            headers.add(name: "Content-Type", value: "text/plain")
            headers.add(name: "Connection", value: "close")

            context.write(
                self.wrapOutboundOut(.head(HTTPResponseHead(
                    version: request.version,
                    status: .internalServerError,
                    headers: headers
                ))),
                promise: nil
            )
            context.write(self.wrapOutboundOut(.body(.byteBuffer(errorBuffer))), promise: nil)
            context.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
        }
    }

    private func respond404(context: ChannelHandlerContext, request: HTTPRequestHead) {
        var notFoundBuffer = context.channel.allocator.buffer(capacity: 64)
        notFoundBuffer.writeString("404 Not Found\n")
        var headers = HTTPHeaders()
        headers.add(name: "Content-Length", value: "\(notFoundBuffer.readableBytes)")
        headers.add(name: "Content-Type", value: "text/plain")
        headers.add(name: "Connection", value: "close")

        context.write(
            self.wrapOutboundOut(.head(HTTPResponseHead(
                version: request.version,
                status: .notFound,
                headers: headers
            ))),
            promise: nil
        )
        context.write(self.wrapOutboundOut(.body(.byteBuffer(notFoundBuffer))), promise: nil)
        context.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil) // Corrected syntax
    }
}

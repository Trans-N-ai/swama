import Foundation
import NIOCore
import NIOHTTP1
import SwamaCore

/// SystemOne transport only. The same engine and model pool answer both decision routes.
enum SystemOneHandler {
    static func handle(requestHead: HTTPRequestHead, body: ByteBuffer, channel: Channel) async {
        await handle(requestHead: requestHead, body: body, channel: channel, engine: ServerCoreEngine.shared)
    }

    static func handle(requestHead: HTTPRequestHead, body: ByteBuffer, channel: Channel, engine: SwamaEngine) async {
        do {
            // Clef's request body limit; swama has no global one, so this route enforces it.
            guard body.readableBytes <= SystemOneRequest.maximumBodyBytes else {
                try? await sendError(
                    "The request body is larger than 13 MiB.", status: .payloadTooLarge,
                    version: requestHead.version, channel: channel
                )
                return
            }

            var readable = body
            guard let bytes = readable.readBytes(length: body.readableBytes) else {
                throw DecisionWireError.invalid("Invalid request body.")
            }

            let request = try SystemOneRequest.parse(Data(bytes))
            let result = try await engine.decide(request.decision)
            try Task.checkCancellation()
            try await send(request.response(result), status: .ok, version: requestHead.version, channel: channel)
        }
        catch is CancellationError {
            // A disconnected peer has no response to receive.
        }
        catch let error as SwamaError {
            let status: HTTPResponseStatus =
                switch error.code {
                case .contextLimitExceeded,
                     .invalidImage,
                     .invalidRequest: .badRequest
                case .modelNotFound: .notFound
                default: .internalServerError
                }
            // HttpSystemOne treats recognized capacity refusals as unsupported rather
            // than aborting its sweep. Preserve the underlying message as well.
            let message = error.code == .contextLimitExceeded ? "maximum context length: " + error.message : error
                .message
            try? await sendError(message, status: status, version: requestHead.version, channel: channel)
        }
        catch {
            let message = error is DecisionWireError ? error.localizedDescription : "Invalid SystemOne request."
            try? await sendError(message, status: .badRequest, version: requestHead.version, channel: channel)
        }
    }

    private static func sendError(
        _ message: String,
        status: HTTPResponseStatus,
        version: HTTPVersion,
        channel: Channel
    ) async throws {
        try await send(.object([
            ("error", .object([
                ("message", .string(message)),
                ("type", .string(status.code >= 500 ? "server_error" : "invalid_request_error"))
            ]))
        ]), status: status, version: version, channel: channel)
    }

    private static func send(
        _ value: SystemOneJSON,
        status: HTTPResponseStatus,
        version: HTTPVersion,
        channel: Channel
    ) async throws {
        let data = Data(value.json.utf8)
        var buffer = channel.allocator.buffer(capacity: data.count)
        buffer.writeBytes(data)
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json")
        headers.add(name: "Content-Length", value: "\(buffer.readableBytes)")
        headers.add(name: "Connection", value: "close")
        headers.add(name: "x-typesafe-request-id", value: "systemone-\(UUID().uuidString)")
        HTTPHandler.applyCORSHeaders(&headers)
        try await channel.writeAndFlush(HTTPServerResponsePart.head(.init(
            version: version,
            status: status,
            headers: headers
        )))
        try await channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(buffer)))
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
    }
}

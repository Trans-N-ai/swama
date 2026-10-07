import Foundation
import NIOCore
import NIOHTTP1
import SwamaCore

// MARK: - DecisionWireError

enum DecisionWireError: Error, LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        if case let .invalid(message) = self {
            return message
        }
        return nil
    }
}

// MARK: - DecisionsHandler

/// `/v1/decisions` transport: the OpenAI Decisions wire format, parsed and encoded by `OpenAIDecisionRequest`.
/// The JSON contract stays out of SwamaCore; the same engine and model pool also answer `/v1/systemone`.
enum DecisionsHandler {
    static func handle(requestHead: HTTPRequestHead, body: ByteBuffer, channel: Channel) async {
        await handle(requestHead: requestHead, body: body, channel: channel, engine: ServerCoreEngine.shared)
    }

    static func handle(requestHead: HTTPRequestHead, body: ByteBuffer, channel: Channel, engine: SwamaEngine) async {
        do {
            var readable = body
            guard let bytes = readable.readBytes(length: body.readableBytes) else {
                throw DecisionWireError.invalid("Invalid request body.")
            }

            let object = try JSONDecoder().decode([String: JSONValue].self, from: Data(bytes))
            let request = try OpenAIDecisionRequest.parse(object)
            let result = try await engine.decide(request.decision)
            try Task.checkCancellation()
            try await send(request.response(result), status: .ok, version: requestHead.version, channel: channel)
        }
        catch is CancellationError {
            // The peer has disconnected; there is no response to deliver.
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
            try? await sendError(error.message, status: status, version: requestHead.version, channel: channel)
        }
        catch {
            let message = error is DecisionWireError ? error.localizedDescription : "Invalid decision request."
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

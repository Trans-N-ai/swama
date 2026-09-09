import Foundation
import NIOCore
import NIOHTTP1
import SwamaCore

// MARK: - Response object + item builders

extension ResponsesHandler {
    static func responseObject(
        id: String,
        createdAt: Int,
        model: String,
        result: GenerationResponse,
        status: String? = nil
    ) -> [String: Any] {
        var output: [[String: Any]] = []
        if !result.output.isEmpty {
            output.append(messageItem(id: newItemID(prefix: "msg"), text: result.output, status: "completed"))
        }
        for toolCall in result.toolCalls {
            output.append(functionCallItem(
                id: newItemID(prefix: "fc"),
                toolCall: toolCall,
                status: "completed"
            ))
        }
        let resolvedStatus = status ?? ((result.finishReason == .length) ? "incomplete" : "completed")
        var object: [String: Any] = [
            "id": id,
            "object": "response",
            "created_at": createdAt,
            "model": model,
            "status": resolvedStatus,
            "output": output,
            "parallel_tool_calls": true,
            "usage": usagePayload(result.usage),
        ]
        if resolvedStatus == "incomplete" {
            object["incomplete_details"] = ["reason": "max_output_tokens"]
        }
        return object
    }

    static func inProgressResponse(id: String, createdAt: Int, model: String) -> [String: Any] {
        [
            "id": id,
            "object": "response",
            "created_at": createdAt,
            "model": model,
            "status": "in_progress",
            "output": [],
            "parallel_tool_calls": true,
        ]
    }

    static func failedResponse(id: String, createdAt: Int, model: String, message: String) -> [String: Any] {
        [
            "id": id,
            "object": "response",
            "created_at": createdAt,
            "model": model,
            "status": "failed",
            "output": [],
            "error": ["code": "server_error", "message": message],
        ]
    }

    static func messageItem(id: String, text: String, status: String) -> [String: Any] {
        [
            "id": id,
            "type": "message",
            "role": "assistant",
            "status": status,
            "content": [["type": "output_text", "text": text, "annotations": []]],
        ]
    }

    static func messageItemStub(id: String) -> [String: Any] {
        ["id": id, "type": "message", "role": "assistant", "status": "in_progress", "content": []]
    }

    static func functionCallItem(id: String, toolCall: ToolCall, status: String) -> [String: Any] {
        [
            "id": id,
            "type": "function_call",
            "call_id": toolCall.id ?? id,
            "name": toolCall.name,
            "arguments": encodedArguments(toolCall.arguments),
            "status": status,
        ]
    }

    static func functionCallStub(id: String, toolCall: ToolCall) -> [String: Any] {
        [
            "id": id,
            "type": "function_call",
            "call_id": toolCall.id ?? id,
            "name": toolCall.name,
            "arguments": "",
            "status": "in_progress",
        ]
    }

    static func usagePayload(_ usage: Usage) -> [String: Any] {
        [
            "input_tokens": usage.promptTokens,
            "output_tokens": usage.completionTokens,
            "total_tokens": usage.totalTokens,
        ]
    }
}

// MARK: - SSE + HTTP writers

extension ResponsesHandler {
    static func startSSE(channel: Channel) async throws {
        let headers = HTTPHeaders([
            ("Content-Type", "text/event-stream"),
            ("Cache-Control", "no-cache"),
            ("Connection", "keep-alive"),
        ])
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        try await channel.writeAndFlush(HTTPServerResponsePart.head(head))
    }

    /// Emit one typed SSE event. OpenAI Responses uses named events plus a JSON
    /// payload that carries its own `type` and a monotonically increasing
    /// `sequence_number` (starting at 0).
    static func emit(
        _ channel: Channel,
        _ sequence: SequenceCounter,
        _ type: String,
        _ fields: [String: Any]
    ) async throws {
        var payload = fields
        payload["type"] = type
        payload["sequence_number"] = await sequence.next()
        let data = try JSONSerialization.data(withJSONObject: payload)
        let json = String(decoding: data, as: UTF8.self)
        try await writeSSELine(channel: channel, line: "event: \(type)\ndata: \(json)\n\n")
    }

    static func finishSSE(channel: Channel) async throws {
        try await writeSSELine(channel: channel, line: "data: [DONE]\n\n")
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
    }

    static func writeSSELine(channel: Channel, line: String) async throws {
        var buffer = channel.allocator.buffer(capacity: line.utf8.count)
        buffer.writeString(line)
        try await channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(buffer)))
    }

    static func writeJSON(channel: Channel, status: HTTPResponseStatus, payload: [String: Any]) async throws {
        let data = try JSONSerialization.data(withJSONObject: payload)
        try await sendFull(channel: channel, data: data, status: status)
    }

    static func respondError(
        channel: Channel,
        status: HTTPResponseStatus,
        reason: RejectionReason
    ) async throws {
        let payload: [String: Any] = [
            "error": [
                "message": reason.errorDescription ?? "invalid request",
                "type": reason.wireType,
                "code": status.code,
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: payload)
        try await sendFull(channel: channel, data: data, status: status)
    }

    private static func sendFull(channel: Channel, data: Data, status: HTTPResponseStatus) async throws {
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json")
        headers.add(name: "Content-Length", value: String(data.count))
        let head = HTTPResponseHead(version: .http1_1, status: status, headers: headers)
        try await channel.writeAndFlush(HTTPServerResponsePart.head(head))
        var buffer = channel.allocator.buffer(capacity: data.count)
        buffer.writeBytes(data)
        try await channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(buffer)))
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
    }
}

// MARK: - Concurrency helpers

extension ResponsesHandler {
    actor SequenceCounter {
        private var value = 0
        func next() -> Int {
            defer { value += 1 }
            return value
        }
    }

    actor TextAccumulator {
        private var text = ""
        private(set) var started = false
        var value: String { text }
        var isEmpty: Bool { !started }
        func append(_ chunk: String) {
            started = true
            text += chunk
        }
    }

    actor ToolCallItemState {
        private var ids: [String] = []
        private var extraIndex = 0
        func add(_: ToolCall) -> String {
            let id = ResponsesHandler.newItemID(prefix: "fc")
            ids.append(id)
            return id
        }

        func outputIndex(after base: Int) -> Int {
            let index = base + 1 + extraIndex
            extraIndex += 1
            return index
        }
    }
}

// MARK: - Small helpers

extension ResponsesHandler {
    static func now() -> Int { Int(Date().timeIntervalSince1970) }
    static func newResponseID() -> String { "resp_" + UUID().uuidString.replacingOccurrences(of: "-", with: "") }
    static func newItemID(prefix: String) -> String {
        prefix + "_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
    }

    static func status(for error: SwamaError) -> HTTPResponseStatus {
        switch error.code {
        case .contextLimitExceeded,
             .invalidImage,
             .invalidRequest:
            .badRequest
        case .modelNotFound:
            .notFound
        default:
            .internalServerError
        }
    }

    static func encodedArguments(_ arguments: [String: SwamaCore.JSONValue]) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(SwamaCore.JSONValue.object(arguments)) else {
            return "{}"
        }

        return String(decoding: data, as: UTF8.self)
    }

    static func jsonValue(_ any: Any) -> JSONValue {
        switch any {
        case let value as [String: Any]:
            .object(value.mapValues(jsonValue))
        case let value as [Any]:
            .array(value.map(jsonValue))
        case let value as String:
            .string(value)
        case let value as Bool where CFGetTypeID(value as CFTypeRef) == CFBooleanGetTypeID():
            .bool(value)
        case let value as NSNumber:
            numberValue(value)
        case is NSNull:
            .null
        default:
            .null
        }
    }

    private static func numberValue(_ number: NSNumber) -> JSONValue {
        if CFGetTypeID(number as CFTypeRef) == CFBooleanGetTypeID() {
            return .bool(number.boolValue)
        }
        let double = number.doubleValue
        if double == double.rounded(), abs(double) < 9_007_199_254_740_992 {
            return .int(number.intValue)
        }
        return .double(double)
    }
}

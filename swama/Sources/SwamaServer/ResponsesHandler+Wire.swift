import Foundation
import NIOCore
import NIOHTTP1
import SwamaCore

// MARK: - Response object + item builders

extension ResponsesHandler {
    /// Shared scaffold of every Response object. The official schema requires
    /// `tool_choice` and `tools` on responses and models the rest as nullable;
    /// all echoed values come verbatim from the parsed request — never invented.
    static func baseResponse(
        id: String,
        createdAt: Int,
        model: String,
        parsed: ParsedRequest,
        status: String
    ) -> [String: Any] {
        [
            "id": id,
            "object": "response",
            "created_at": createdAt,
            "model": model,
            "status": status,
            "output": [[String: Any]](),
            "parallel_tool_calls": parsed.parallelToolCalls,
            "tool_choice": parsed.toolChoice,
            "tools": parsed.request.tools.map(toolPayload),
            "temperature": Double(parsed.temperature),
            "top_p": Double(parsed.topP),
            "max_output_tokens": parsed.maxOutputTokens as Any? ?? NSNull(),
            "instructions": parsed.instructions as Any? ?? NSNull(),
            "previous_response_id": NSNull(),
            "truncation": "disabled",
            "store": false,
            "background": false,
            "metadata": [String: Any](),
            "text": ["format": ["type": "text"]],
            "error": NSNull(),
            "incomplete_details": NSNull(),
            "usage": NSNull(),
            "user": NSNull(),
        ]
    }

    static func responseObject(
        id: String,
        createdAt: Int,
        model: String,
        parsed: ParsedRequest,
        result: GenerationResponse,
        status: String? = nil,
        streamedOutput: [[String: Any]]? = nil
    ) -> [String: Any] {
        // A streaming response must close with the exact item ids/indexes its
        // typed events announced; only the non-streaming path mints fresh ids.
        let resolvedStatus = status ?? ((result.finishReason == .length) ? "incomplete" : "completed")
        var output: [[String: Any]] = streamedOutput ?? []
        if streamedOutput == nil {
            if !result.output.isEmpty {
                // A truncated response marks its message item incomplete too.
                output.append(messageItem(
                    id: newItemID(prefix: "msg"),
                    text: result.output,
                    status: resolvedStatus == "incomplete" ? "incomplete" : "completed"
                ))
            }
            for toolCall in result.toolCalls {
                output.append(functionCallItem(
                    id: newItemID(prefix: "fc"),
                    toolCall: toolCall,
                    status: "completed"
                ))
            }
        }
        var object = baseResponse(
            id: id,
            createdAt: createdAt,
            model: model,
            parsed: parsed,
            status: resolvedStatus
        )
        object["output"] = output
        object["usage"] = usagePayload(result.usage)
        if resolvedStatus == "incomplete" {
            object["incomplete_details"] = ["reason": "max_output_tokens"]
        }
        return object
    }

    static func inProgressResponse(
        id: String,
        createdAt: Int,
        model: String,
        parsed: ParsedRequest
    ) -> [String: Any] {
        baseResponse(id: id, createdAt: createdAt, model: model, parsed: parsed, status: "in_progress")
    }

    static func failedResponse(
        id: String,
        createdAt: Int,
        model: String,
        parsed: ParsedRequest,
        message: String
    ) -> [String: Any] {
        var object = baseResponse(id: id, createdAt: createdAt, model: model, parsed: parsed, status: "failed")
        object["error"] = ["code": "server_error", "message": message]
        return object
    }

    /// Wire form of a request tool definition, echoed into Response objects.
    static func toolPayload(_ tool: ToolDefinition) -> [String: Any] {
        [
            "type": "function",
            "name": tool.name,
            "description": tool.description as Any? ?? NSNull(),
            "parameters": anyValue(tool.parameters),
            "strict": false,
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
            // Required by the official usage schema. The local runtime has no
            // prompt cache or separated reasoning accounting, so zero is the
            // honest value, not a placeholder.
            "input_tokens_details": ["cached_tokens": 0, "cache_write_tokens": 0],
            "output_tokens_details": ["reasoning_tokens": 0],
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
        // The Responses stream ends with its typed terminal event
        // (`response.completed`/`.incomplete`/`.failed`); the Chat-Completions
        // `data: [DONE]` sentinel is not part of the Responses event union.
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
        // Official error objects carry a string-or-null `code`, never the HTTP
        // status number, plus a nullable `param`.
        let payload: [String: Any] = [
            "error": [
                "message": reason.errorDescription ?? "invalid request",
                "type": status.code < 500 ? reason.wireType : "server_error",
                "code": reason.wireCode,
                "param": NSNull(),
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

    /// Owns the identity of every streamed output item: `output_index` values are
    /// allocated in emission order, and the terminal `response` reuses the exact
    /// item ids and indexes the typed events announced. Clients correlate events
    /// with the final output through these, so they must not be regenerated.
    /// One streamed output item's identity plus what is needed to rebuild its
    /// terminal payload outside the actor (dictionaries are not Sendable).
    enum OutputRecord: Sendable {
        case message(id: String)
        case functionCall(id: String, toolCall: ToolCall)
    }

    actor OutputAssembler {
        struct Slot: Sendable {
            var id: String
            var index: Int
        }

        private var nextIndex = 0
        private var textSlot: Slot?
        private var text = ""
        private(set) var records: [OutputRecord] = []

        var assembledText: String { text }
        var textSlotIfStarted: Slot? { textSlot }

        /// First text delta: allocate the message item's index exactly once.
        func startTextIfNeeded() -> (slot: Slot, isFirst: Bool) {
            if let textSlot {
                return (textSlot, false)
            }
            let slot = Slot(id: ResponsesHandler.newItemID(prefix: "msg"), index: allocate())
            textSlot = slot
            records.append(.message(id: slot.id))
            return (slot, true)
        }

        func appendText(_ chunk: String) {
            text += chunk
        }

        /// Core's final output is authoritative over accumulated deltas.
        func setFinalText(_ finalText: String) {
            text = finalText
        }

        func addFunctionCall(_ toolCall: ToolCall) -> Slot {
            let slot = Slot(id: ResponsesHandler.newItemID(prefix: "fc"), index: allocate())
            records.append(.functionCall(id: slot.id, toolCall: toolCall))
            return slot
        }

        private func allocate() -> Int {
            defer { nextIndex += 1 }
            return nextIndex
        }
    }

    /// Terminal `output` array in emission order, reusing every streamed item id.
    static func streamedOutput(
        records: [OutputRecord],
        text: String,
        textStatus: String
    ) -> [[String: Any]] {
        records.map { record in
            switch record {
            case let .message(id):
                messageItem(id: id, text: text, status: textStatus)
            case let .functionCall(id, toolCall):
                functionCallItem(id: id, toolCall: toolCall, status: "completed")
            }
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

    /// Inverse of `jsonValue`: JSONSerialization-compatible representation.
    static func anyValue(_ value: JSONValue) -> Any {
        switch value {
        case let .object(object):
            object.mapValues(anyValue)
        case let .array(array):
            array.map(anyValue)
        case let .string(string):
            string
        case let .bool(bool):
            bool
        case let .int(int):
            int
        case let .double(double):
            double
        case .null:
            NSNull()
        }
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

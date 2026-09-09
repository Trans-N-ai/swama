//
//  CompletionsHandler.swift
//  SwamaKit
//

import Foundation
import NIOCore
import NIOHTTP1
import SwamaCore

// MARK: - CompletionsHandler

public enum CompletionsHandler {
    // MARK: Public

    public struct CompletionRequest: Decodable, Sendable {
        let model: String
        let messages: [Message]
        let temperature: Float?
        let top_p: Float?
        let top_k: Int?
        let min_p: Float?
        let repetition_penalty: Float?
        let repetition_context_size: Int?
        let presence_penalty: Float?
        let frequency_penalty: Float?
        let context_limit: Int?
        let max_tokens: Int?
        let stream: Bool?
        let tools: [Tool]?
        let tool_choice: ToolChoice?
    }

    public struct Message: Decodable, Encodable, Sendable {
        let role: String
        let content: MessageContent
        let tool_calls: [ResponseToolCall]?
        let tool_call_id: String?

        private enum CodingKeys: String, CodingKey {
            case role
            case content
            case tool_calls
            case tool_call_id
        }

        public init(role: String, content: MessageContent, tool_calls: [ResponseToolCall]? = nil) {
            self.role = role
            self.content = content
            self.tool_calls = tool_calls
            tool_call_id = nil
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            role = try container.decode(String.self, forKey: .role)
            content = try container.decode(MessageContent.self, forKey: .content)
            tool_calls = try container.decodeIfPresent([ResponseToolCall].self, forKey: .tool_calls)
            tool_call_id = try container.decodeIfPresent(String.self, forKey: .tool_call_id)
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(role, forKey: .role)
            try container.encode(content, forKey: .content)
            if let tool_calls, !tool_calls.isEmpty {
                try container.encode(tool_calls, forKey: .tool_calls)
            }
            try container.encodeIfPresent(tool_call_id, forKey: .tool_call_id)
        }
    }

    public enum MessageContent: Decodable, Encodable, Sendable {
        case text(String)
        case multimodal([ContentPartValue])

        var textContent: String {
            switch self {
            case let .text(text):
                text
            case let .multimodal(parts):
                parts.compactMap { part in
                    if case let .text(text) = part {
                        return text
                    }
                    return nil
                }
                .joined(separator: " ")
            }
        }

        var imageURLs: [String] {
            switch self {
            case .text:
                []
            case let .multimodal(parts):
                parts.compactMap { part in
                    if case let .imageURL(imageURL) = part {
                        return imageURL.url
                    }
                    return nil
                }
            }
        }
    }

    public struct ContentPart: Decodable, Encodable, Sendable {
        let type: String
        let text: String?
        let image_url: ImageURL?

        enum CodingKeys: String, CodingKey {
            case type
            case text
            case image_url
        }
    }

    public enum ContentPartValue: Decodable, Encodable, Sendable {
        case text(String)
        case imageURL(ImageURL)

        enum CodingKeys: String, CodingKey {
            case type
            case text
            case image_url
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decode(String.self, forKey: .type)

            switch type {
            case "text":
                let text = try container.decode(String.self, forKey: .text)
                self = .text(text)

            case "image_url":
                let imageURL = try container.decode(ImageURL.self, forKey: .image_url)
                self = .imageURL(imageURL)

            default:
                throw DecodingError.dataCorruptedError(
                    forKey: .type,
                    in: container,
                    debugDescription: "Invalid content part type"
                )
            }
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            switch self {
            case let .text(text):
                try container.encode("text", forKey: .type)
                try container.encode(text, forKey: .text)

            case let .imageURL(imageURL):
                try container.encode("image_url", forKey: .type)
                try container.encode(imageURL, forKey: .image_url)
            }
        }
    }

    public struct ImageURL: Decodable, Encodable, Sendable {
        let url: String
    }

    public struct CompletionResponse: Encodable, Sendable {
        let id: String
        let object: String
        let created: Int
        let model: String
        let choices: [CompletionChoice]
        let usage: CompletionUsage
    }

    public struct CompletionChoice: Encodable, Sendable {
        let index: Int
        let message: Message
        let finish_reason: String

        private enum CodingKeys: String, CodingKey {
            case index
            case message
            case finish_reason
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(index, forKey: .index)
            try container.encode(message, forKey: .message)
            try container.encode(finish_reason, forKey: .finish_reason)
        }
    }

    public struct CompletionUsage: Encodable, Sendable {
        let prompt_tokens: Int
        let completion_tokens: Int
        let total_tokens: Int
        let response_token_s: Double?
        let total_duration: Double?

        private enum CodingKeys: String, CodingKey {
            case prompt_tokens
            case completion_tokens
            case total_tokens
            case response_token_s = "response_token/s"
            case total_duration
        }
    }

    // MARK: - Tool Calling Support

    /// OpenAI-compatible tool call structures for response
    public struct ResponseToolCall: Encodable, Decodable, Sendable {
        let index: Int?
        let id: String
        let type: String
        let function: ResponseFunction

        public init(index: Int = 0, id: String, type: String = "function", function: ResponseFunction) {
            self.index = index
            self.id = id
            self.type = type
            self.function = function
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            index = try container.decodeIfPresent(Int.self, forKey: .index)
            id = try container.decode(String.self, forKey: .id)
            type = try container.decode(String.self, forKey: .type)
            function = try container.decode(ResponseFunction.self, forKey: .function)
        }
    }

    public struct ResponseFunction: Encodable, Decodable, Sendable {
        let name: String
        let arguments: String // JSON string

        public init(name: String, arguments: String) {
            self.name = name
            self.arguments = arguments
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            name = try container.decode(String.self, forKey: .name)
            arguments = try container.decode(String.self, forKey: .arguments)
        }
    }

    /// Helper for JSON decoding
    private enum JSONValue: Decodable {
        case string(String)
        case number(Double)
        case bool(Bool)
        case null
        case array([JSONValue])
        case object([String: JSONValue])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if container.decodeNil() {
                self = .null
            }
            else if let bool = try? container.decode(Bool.self) {
                self = .bool(bool)
            }
            else if let number = try? container.decode(Double.self) {
                self = .number(number)
            }
            else if let string = try? container.decode(String.self) {
                self = .string(string)
            }
            else if let array = try? container.decode([JSONValue].self) {
                self = .array(array)
            }
            else if let object = try? container.decode([String: JSONValue].self) {
                self = .object(object)
            }
            else {
                throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid JSON value")
            }
        }

        var anyValue: Any {
            switch self {
            case let .string(s): s
            case let .number(n): n
            case let .bool(b): b
            case .null: NSNull()
            case let .array(a): a.map(\.anyValue)
            case let .object(o): o.mapValues { $0.anyValue }
            }
        }
    }

    public struct Tool: Decodable, Encodable, Sendable {
        let type: String
        let function: Function
    }

    public struct Function: Decodable, Encodable, Sendable {
        let name: String
        let description: String?
        let parameters: String? // JSON string

        private enum CodingKeys: String, CodingKey {
            case name
            case description
            case parameters
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            name = try container.decode(String.self, forKey: .name)
            description = try container.decodeIfPresent(String.self, forKey: .description)

            // Try to decode parameters as JSON and convert to string
            if let parametersValue = try? container.decodeIfPresent(JSONValue.self, forKey: .parameters) {
                let jsonData = try JSONSerialization.data(
                    withJSONObject: parametersValue.anyValue,
                    options: [.fragmentsAllowed]
                )
                parameters = String(data: jsonData, encoding: .utf8)
            }
            else {
                parameters = nil
            }
        }
    }

    public enum ToolChoice: Decodable, Encodable, Sendable {
        case none
        case auto
        case required
        case function(String)

        public func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .none:
                try container.encode("none")
            case .auto:
                try container.encode("auto")
            case .required:
                try container.encode("required")
            case let .function(name):
                let functionChoice: [String: Any] = ["type": "function", "function": ["name": name]]
                let jsonData = try JSONSerialization.data(withJSONObject: functionChoice)
                let jsonString = String(data: jsonData, encoding: .utf8) ?? ""
                try container.encode(jsonString)
            }
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let string = try? container.decode(String.self) {
                switch string {
                case "none":
                    self = .none
                case "auto":
                    self = .auto
                case "required":
                    self = .required
                default:
                    // Try to parse as JSON for function choice
                    if let data = string.data(using: .utf8),
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       json["type"] as? String == "function",
                       let functionDict = json["function"] as? [String: Any],
                       let name = functionDict["name"] as? String
                    {
                        self = .function(name)
                    }
                    else {
                        throw DecodingError.dataCorruptedError(
                            in: container,
                            debugDescription: "Invalid tool choice string"
                        )
                    }
                }
            }
            else if let jsonValue = try? container.decode(JSONValue.self) {
                if case let .object(dict) = jsonValue,
                   case let .string(type) = dict["type"], type == "function",
                   case let .object(functionDict) = dict["function"],
                   case let .string(name) = functionDict["name"]
                {
                    self = .function(name)
                }
                else {
                    throw DecodingError.dataCorruptedError(
                        in: container,
                        debugDescription: "Invalid tool choice format"
                    )
                }
            }
            else {
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "Invalid tool choice format"
                )
            }
        }
    }

    public static func handle(
        requestHead: HTTPRequestHead,
        body: ByteBuffer?,
        channel: Channel
    ) async {
        await handle(
            requestHead: requestHead,
            body: body,
            channel: channel,
            engine: ServerCoreEngine.shared
        )
    }

    static func handle(
        requestHead _: HTTPRequestHead,
        body: ByteBuffer?,
        channel: Channel,
        engine: SwamaEngine
    ) async {
        do {
            guard let payload = parsePayload(body),
                  !payload.messages.isEmpty
            else {
                try? await respondError(
                    channel: channel,
                    status: .badRequest,
                    message: "Invalid request payload or missing messages"
                )
                return
            }

            let request = try coreRequest(from: payload)

            if payload.stream == true {
                try await sendStreamResponse(
                    channel: channel,
                    request: request,
                    model: payload.model,
                    engine: engine
                )
            }
            else {
                try await sendNonStreamResponse(
                    channel: channel,
                    request: request,
                    model: payload.model,
                    engine: engine
                )
            }
        }
        catch let error as SwamaError {
            try? await respondError(
                channel: channel,
                status: status(for: error),
                message: error.message
            )
        }
        catch let error as CompletionsError {
            try? await respondError(
                channel: channel,
                status: .badRequest,
                message: error.localizedDescription
            )
        }
        catch {
            try? await respondError(
                channel: channel,
                status: .internalServerError,
                message: error.localizedDescription
            )
        }
    }

    // MARK: - Core conversion

    static func coreRequest(from payload: CompletionRequest) throws -> GenerationRequest {
        try .init(
            model: .init(payload.model),
            messages: payload.messages.map(coreMessage),
            options: .init(
                maxTokens: payload.max_tokens,
                temperature: payload.temperature ?? 0.6,
                topP: payload.top_p ?? 1,
                topK: payload.top_k ?? 0,
                minP: payload.min_p ?? 0,
                repetitionPenalty: payload.repetition_penalty,
                repetitionContextSize: payload.repetition_context_size ?? 20,
                presencePenalty: payload.presence_penalty,
                frequencyPenalty: payload.frequency_penalty,
                contextLimit: payload.context_limit
            ),
            tools: payload.tools?.map(coreTool) ?? []
        )
    }

    private static func coreMessage(_ message: Message) throws -> SwamaCore.Message {
        let role: SwamaCore.Message.Role =
            switch message.role {
            case "system":
                .system
            case "user":
                .user
            case "assistant":
                .assistant
            case "tool":
                .tool
            default:
                throw CompletionsError.invalidRole(message.role)
            }

        let content: [SwamaCore.ContentPart] =
            switch message.content {
            case let .text(text):
                [.text(text)]
            case let .multimodal(parts):
                try parts.map { part -> SwamaCore.ContentPart in
                    switch part {
                    case let .text(text):
                        return .text(text)
                    case let .imageURL(image):
                        guard let url = URL(string: image.url) else {
                            throw CompletionsError.invalidImageURL(image.url)
                        }

                        return SwamaCore.ContentPart.imageURL(url)
                    }
                }
            }

        return try .init(
            role: role,
            content: content,
            toolCalls: message.tool_calls?.map(coreToolCall) ?? [],
            toolCallID: message.tool_call_id
        )
    }

    private static func coreTool(_ tool: Tool) throws -> ToolDefinition {
        let parameters: SwamaCore.JSONValue
        if let raw = tool.function.parameters {
            let decoded = try JSONDecoder().decode(SwamaCore.JSONValue.self, from: Data(raw.utf8))
            guard case .object = decoded else {
                throw CompletionsError.invalidToolJSON(tool.function.name)
            }

            parameters = decoded
        }
        else {
            parameters = .object([:])
        }
        return .init(
            name: tool.function.name,
            description: tool.function.description,
            parameters: parameters
        )
    }

    private static func coreToolCall(_ toolCall: ResponseToolCall) throws -> SwamaCore.ToolCall {
        let decoded = try JSONDecoder().decode(
            SwamaCore.JSONValue.self,
            from: Data(toolCall.function.arguments.utf8)
        )
        guard case let .object(arguments) = decoded else {
            throw CompletionsError.invalidToolJSON(toolCall.function.name)
        }

        return .init(
            id: toolCall.id,
            name: toolCall.function.name,
            arguments: arguments
        )
    }

    // MARK: - Chat Response Methods

    static func sendNonStreamResponse(
        channel: Channel,
        request: GenerationRequest,
        model: String,
        engine: SwamaEngine
    ) async throws {
        let result = try await runCancellingOnClose(channel: channel) {
            try await engine.generate(request)
        }

        // The client disconnected mid-generation; the run above was cancelled promptly, and
        // there is nothing left to write to.
        guard channel.isActive else {
            return
        }

        let toolCalls: [ResponseToolCall]? = result.toolCalls.isEmpty ? nil : result.toolCalls
            .enumerated()
            .map { index, toolCall in
                ResponseToolCall(
                    index: index,
                    id: toolCall.id ?? "call_\(UUID().uuidString)",
                    type: "function",
                    function: ResponseFunction(
                        name: toolCall.name,
                        arguments: encodedArguments(toolCall.arguments)
                    )
                )
            }

        // Construct the message content for the response
        let responseMessageContent = MessageContent.text(result.output)
        let responseMessage = Message(
            role: "assistant",
            content: responseMessageContent,
            tool_calls: toolCalls
        )

        let choice = CompletionChoice(
            index: 0,
            message: responseMessage,
            finish_reason: wireFinishReason(result.finishReason)
        )

        let tokensPerSecond = result.metrics?.tokensPerSecond ?? 0
        let totalDuration = (result.metrics?.promptSeconds ?? 0) + (result.metrics?.generationSeconds ?? 0)

        let usage = CompletionUsage(
            prompt_tokens: result.usage.promptTokens,
            completion_tokens: result.usage.completionTokens,
            total_tokens: result.usage.totalTokens,
            response_token_s: tokensPerSecond > 0 ? tokensPerSecond : nil,
            total_duration: totalDuration > 0 ? totalDuration : nil
        )

        let response = CompletionResponse(
            id: "chatcmpl-" + UUID().uuidString,
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: model,
            choices: [choice],
            usage: usage
        )

        // Send JSON response using existing method
        let encoder = JSONEncoder()
        let jsonData = try encoder.encode(response)
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json")
        HTTPHandler.applyCORSHeaders(&headers)

        let responseHead = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        let responseBody = HTTPServerResponsePart.body(.byteBuffer(ByteBuffer(bytes: jsonData)))

        try await channel.writeAndFlush(HTTPServerResponsePart.head(responseHead))
        try await channel.writeAndFlush(responseBody)
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
    }

    static func sendStreamResponse(
        channel: Channel,
        request: GenerationRequest,
        model: String,
        engine: SwamaEngine
    ) async throws {
        actor ToolCallCounter {
            private var index = 0
            func next() -> Int {
                let current = index
                index += 1
                return current
            }
        }
        // Send SSE headers
        let headers = HTTPHeaders([
            ("Content-Type", "text/event-stream"),
            ("Cache-Control", "no-cache"),
            ("Connection", "keep-alive")
        ])
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)

        try await channel.writeAndFlush(HTTPServerResponsePart.head(head))

        let chunkId = "chatcmpl-" + UUID().uuidString
        let timestamp = Int(Date().timeIntervalSince1970)

        // Send initial chunk with role
        let initialJSON: [String: Any] = [
            "id": chunkId,
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": model,
            "choices": [["index": 0, "delta": ["role": "assistant"], "finish_reason": NSNull()]]
        ]

        try await writeSSEJSON(channel: channel, payload: initialJSON)

        let result: GenerationResponse

        do {
            result = try await runCancellingOnClose(channel: channel) {
                let toolCallCounter = ToolCallCounter()
                return try await engine.generate(request) { event in
                    switch event {
                    case let .textDelta(chunk):
                        let deltaJSON: [String: Any] = [
                            "id": chunkId,
                            "object": "chat.completion.chunk",
                            "created": timestamp,
                            "model": model,
                            "choices": [["index": 0, "delta": ["content": chunk], "finish_reason": NSNull()]]
                        ]
                        try await writeSSEJSON(channel: channel, payload: deltaJSON)

                    case let .toolCall(toolCall):
                        let index = await toolCallCounter.next()
                        let toolCallDict: [String: Any] = [
                            "index": index,
                            "id": toolCall.id ?? "call_\(UUID().uuidString)",
                            "type": "function",
                            "function": [
                                "name": toolCall.name,
                                "arguments": encodedArguments(toolCall.arguments)
                            ]
                        ]

                        let toolCallDelta: [String: Any] = [
                            "id": chunkId,
                            "object": "chat.completion.chunk",
                            "created": timestamp,
                            "model": model,
                            "choices": [["index": 0, "delta": ["tool_calls": [toolCallDict]],
                                         "finish_reason": NSNull()]]
                        ]

                        try await writeSSEJSON(channel: channel, payload: toolCallDelta)
                    }
                }
            }
        }
        catch {
            // The client disconnected mid-generation; there is nothing left to write to.
            guard channel.isActive else {
                return
            }

            // Send error through SSE instead of trying to change HTTP status
            let errorJSON: [String: Any] = [
                "id": chunkId,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model,
                "choices": [["index": 0, "delta": [:], "finish_reason": "error"]],
                "error": [
                    "message": error.localizedDescription,
                    "type": "request_error"
                ]
            ]
            try await writeSSEJSON(channel: channel, payload: errorJSON)
            try await writeSSELine(channel: channel, line: "data: [DONE]\n\n")
            try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
            return
        }

        // The client disconnected mid-generation; the run above was cancelled promptly, and
        // there is nothing left to write to.
        guard channel.isActive else {
            return
        }

        // Send final chunk with usage information
        let tokensPerSecond = result.metrics?.tokensPerSecond ?? 0
        let totalDuration = (result.metrics?.promptSeconds ?? 0) + (result.metrics?.generationSeconds ?? 0)

        let finishJSON: [String: Any] = [
            "id": chunkId,
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": model,
            "choices": [["index": 0, "delta": [:], "finish_reason": wireFinishReason(result.finishReason)]],
            "usage": [
                "prompt_tokens": result.usage.promptTokens,
                "completion_tokens": result.usage.completionTokens,
                "total_tokens": result.usage.totalTokens,
                "response_token/s": tokensPerSecond,
                "total_duration": totalDuration
            ]
        ]

        try await writeSSEJSON(channel: channel, payload: finishJSON)
        try await writeSSELine(channel: channel, line: "data: [DONE]\n\n")
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
    }

    // MARK: - Helper Methods

    /// Runs `operation` in a child task, cancelling that task as soon as `channel`'s connection
    /// closes (the client disconnecting mid-request). Core operations propagate
    /// `CancellationError`; already-written stream deltas are not retracted.
    ///
    /// Not marked `private` so tests can drive this channel-close -> cancellation path directly
    /// with a stub `operation` and an `EmbeddedChannel`, without needing a real model.
    static func runCancellingOnClose<T: Sendable>(
        channel: Channel,
        operation: @Sendable @escaping () async throws -> T
    ) async throws -> T {
        let task = Task { try await operation() }

        // `channel.closeFuture` only resolves when the *connection* closes, not when this one
        // request finishes -- so on a keep-alive connection, capturing `task` directly here
        // would keep the completed task (and everything it retains, including its returned
        // result) reachable for as long as the connection stays open, not just for the
        // lifetime of this request. Capture a clearable box instead, and clear it in the
        // `defer` below once this request's operation has completed, so the task and its
        // result can be released immediately rather than pinned until connection close.
        //
        // This does not fully eliminate per-request retention on a keep-alive connection: NIO
        // gives no way to deregister a `whenComplete` callback, so the (now-empty) closure
        // itself still accumulates on `closeFuture` for every request until the connection
        // finally closes. What this bounds is *what* each accumulated closure keeps alive --
        // an empty box, not the completed task and its captured state/result.
        let taskBox = CancellableTaskBox(task)
        channel.closeFuture.whenComplete { _ in taskBox.cancel() }
        defer { taskBox.clear() }

        return try await withTaskCancellationHandler {
            try await task.value
        } onCancel: {
            task.cancel()
        }
    }

    /// A clearable holder for a `Task`, used so `channel.closeFuture`'s callback can cancel an
    /// in-flight task without keeping a completed task (and its result) alive for the life of a
    /// keep-alive connection. See `runCancellingOnClose` for why this exists.
    private final class CancellableTaskBox<T: Sendable>: @unchecked Sendable {
        private let lock: NSLock = .init()
        private var task: Task<T, Error>?

        init(_ task: Task<T, Error>) {
            self.task = task
        }

        /// Cancels the held task, if it hasn't already been cleared.
        func cancel() {
            lock.lock()
            let current = task
            lock.unlock()
            current?.cancel()
        }

        /// Releases the held reference to the task (and, transitively, its result) without
        /// cancelling it.
        func clear() {
            lock.lock()
            task = nil
            lock.unlock()
        }
    }

    private static func sendFullResponse(
        channel: Channel,
        data: Data,
        status: HTTPResponseStatus,
        version: HTTPVersion
    ) async throws {
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json")
        headers.add(name: "Content-Length", value: String(data.count))
        HTTPHandler.applyCORSHeaders(&headers)

        let responseHead = HTTPResponseHead(version: version, status: status, headers: headers)
        let responseBody = HTTPServerResponsePart.body(.byteBuffer(ByteBuffer(bytes: data)))

        try await channel.writeAndFlush(HTTPServerResponsePart.head(responseHead))
        try await channel.writeAndFlush(responseBody)
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
    }

    private static func respondError(
        channel: Channel,
        status: HTTPResponseStatus,
        message: String
    ) async throws {
        let errorJSON: [String: Any] = [
            "error": [
                "message": message,
                "type": "invalid_request_error",
                "code": status.code
            ]
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: errorJSON)
        try await sendFullResponse(channel: channel, data: jsonData, status: status, version: .http1_1)
    }

    private static func encodedArguments(_ arguments: [String: SwamaCore.JSONValue]) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(SwamaCore.JSONValue.object(arguments)) else {
            return "{}"
        }

        return String(decoding: data, as: UTF8.self)
    }

    private static func wireFinishReason(_ reason: FinishReason) -> String {
        switch reason {
        case .completed,
             .unknown:
            "stop"
        case .length:
            "length"
        case .toolCall:
            "tool_calls"
        }
    }

    private static func status(for error: SwamaError) -> HTTPResponseStatus {
        switch error.code {
        case .contextLimitExceeded,
             .invalidImage,
             .invalidRequest:
            .badRequest
        case .modelNotFound:
            .notFound
        case .backendFailure,
             .downloadFailed,
             .embeddingFailed,
             .modelLoadFailed,
             .removalFailed:
            .internalServerError
        }
    }

    private static func parsePayload(_ buffer: ByteBuffer?) -> CompletionRequest? {
        guard let buffer,
              let data = buffer.getBytes(at: buffer.readerIndex, length: buffer.readableBytes)
        else {
            return nil
        }

        do {
            return try JSONDecoder().decode(CompletionRequest.self, from: Data(data))
        }
        catch {
            return nil
        }
    }

    private static func writeSSEJSON(channel: Channel, payload: [String: Any]) async throws {
        let jsonData = try JSONSerialization.data(withJSONObject: payload)
        guard let jsonString = String(data: jsonData, encoding: .utf8) else {
            throw NSError(
                domain: "EncodingError",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to encode JSON to string"]
            )
        }

        try await writeSSELine(channel: channel, line: "data: \(jsonString)\n\n")
    }

    private static func writeSSELine(channel: Channel, line: String) async throws {
        var buffer = channel.allocator.buffer(capacity: line.utf8.count)
        buffer.writeString(line)
        try await channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(buffer)))
    }
}

// MARK: - CompletionsError

enum CompletionsError: Error, LocalizedError {
    case invalidRole(String)
    case invalidImageURL(String)
    case invalidToolJSON(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidRole(role):
            "Invalid role: \(role). Must be 'system', 'user', 'assistant', or 'tool'"
        case let .invalidImageURL(value):
            "Invalid image URL: \(value)"
        case let .invalidToolJSON(name):
            "Tool '\(name)' requires JSON object arguments or parameters"
        }
    }
}

// MARK: - MessageContent Extensions

public extension CompletionsHandler.MessageContent {
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        // Handle null values
        if container.decodeNil() {
            self = .text("")
            return
        }

        // Try to decode as string first
        if let text = try? container.decode(String.self) {
            self = .text(text)
            return
        }

        // Try to decode as array of content parts
        if let parts = try? container.decode([CompletionsHandler.ContentPart].self) {
            let convertedParts = parts.map { part in
                if let text = part.text {
                    CompletionsHandler.ContentPartValue.text(text)
                }
                else if let imageURL = part.image_url {
                    CompletionsHandler.ContentPartValue.imageURL(imageURL)
                }
                else {
                    CompletionsHandler.ContentPartValue.text("")
                }
            }
            self = .multimodal(convertedParts)
            return
        }

        // Fallback to empty text if nothing else works
        self = .text("")
    }

    func encode(to encoder: Encoder) throws {
        switch self {
        case let .text(text):
            try text.encode(to: encoder)
        case let .multimodal(parts):
            try parts.encode(to: encoder)
        }
    }
}

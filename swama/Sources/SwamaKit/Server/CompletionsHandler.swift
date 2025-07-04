//
//  CompletionsHandler.swift
//  SwamaKit
//

import Foundation
@preconcurrency import MLXLMCommon
import NIOCore
import NIOHTTP1
import struct Tokenizers.ToolSpec

// MARK: - CompletionsHandler

public enum CompletionsHandler {
    // MARK: Public

    public struct CompletionRequest: Decodable, Sendable {
        let model: String
        let messages: [Message]
        let temperature: Float?
        let top_p: Float?
        let max_tokens: Int?
        let stream: Bool?
        let tools: [Tool]?
        let tool_choice: ToolChoice?
    }

    public struct Message: Decodable, Encodable, Sendable {
        let role: String
        let content: MessageContent
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
        let tool_calls: [ResponseToolCall]?

        private enum CodingKeys: String, CodingKey {
            case index
            case message
            case finish_reason
            case tool_calls
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(index, forKey: .index)
            try container.encode(message, forKey: .message)
            try container.encode(finish_reason, forKey: .finish_reason)
            if let tool_calls, !tool_calls.isEmpty {
                try container.encode(tool_calls, forKey: .tool_calls)
            }
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
    public struct ResponseToolCall: Encodable, Sendable {
        let id: String
        let type: String
        let function: ResponseFunction

        public init(id: String, type: String = "function", function: ResponseFunction) {
            self.id = id
            self.type = type
            self.function = function
        }
    }

    public struct ResponseFunction: Encodable, Sendable {
        let name: String
        let arguments: String // JSON string

        public init(name: String, arguments: String) {
            self.name = name
            self.arguments = arguments
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
                let jsonData = try JSONSerialization.data(withJSONObject: parametersValue.anyValue)
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
        requestHead _: HTTPRequestHead,
        body: ByteBuffer?,
        channel: Channel
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

            let resolvedModelName = ModelAliasResolver.resolve(name: payload.model)

            // Convert messages to MLX Chat.Message format
            let chatMessages = try convertToMLXChatMessages(payload.messages)

            let parameters = GenerateParameters(
                maxTokens: payload.max_tokens,
                temperature: payload.temperature ?? 0.6,
                topP: payload.top_p ?? 1.0
            )

            // Convert tools to MLX ToolSpec format once here
            let tools: [ToolSpec]? = convertToolsToMLX(payload.tools)

            if payload.stream == true {
                try await sendStreamResponse(
                    channel: channel,
                    modelName: resolvedModelName,
                    chatMessages: chatMessages,
                    model: payload.model,
                    parameters: parameters,
                    tools: tools
                )
            }
            else {
                try await sendNonStreamResponse(
                    channel: channel,
                    modelName: resolvedModelName,
                    chatMessages: chatMessages,
                    model: payload.model,
                    parameters: parameters,
                    mlxTools: tools
                )
            }
        }
        catch {
            // Use structured logging instead of NSLog for better production practices
            if ProcessInfo.processInfo.environment["DEBUG"] != nil {
                print("❌ SwamaKit.CompletionsHandler Error: \(error)")
            }
            try? await respondError(
                channel: channel,
                status: .internalServerError,
                message: error.localizedDescription
            )
        }
    }

    // MARK: - Tool Conversion Helper

    private static func convertToolsToMLX(_ tools: [Tool]?) -> [ToolSpec]? {
        tools?.map { tool in
            var toolSpec: ToolSpec = [
                "type": tool.type,
                "function": [
                    "name": tool.function.name
                ]
            ]

            if let description = tool.function.description {
                var functionDict = toolSpec["function"] as! [String: Any]
                functionDict["description"] = description
                toolSpec["function"] = functionDict
            }

            if let parameters = tool.function.parameters {
                var functionDict = toolSpec["function"] as! [String: Any]
                // Convert JSON string back to object
                if let jsonData = parameters.data(using: .utf8),
                   let jsonObject = try? JSONSerialization.jsonObject(with: jsonData)
                {
                    functionDict["parameters"] = jsonObject
                }
                toolSpec["function"] = functionDict
            }

            return toolSpec
        }
    }

    // MARK: - Chat Message Conversion

    private static func convertToMLXChatMessages(_ messages: [Message]) throws -> [MLXLMCommon.Chat.Message] {
        try messages.map { message in
            let role: MLXLMCommon.Chat.Message.Role
            switch message.role {
            case "system":
                role = .system
            case "user":
                role = .user
            case "assistant":
                role = .assistant
            case "tool":
                role = .tool
            default:
                throw CompletionsError.invalidRole(message.role)
            }

            let content = message.content.textContent
            let imageURLs = message.content.imageURLs
            let images = imageURLs.compactMap { urlString in
                URL(string: urlString).map { MLXLMCommon.UserInput.Image.url($0) }
            }

            return MLXLMCommon.Chat.Message(
                role: role,
                content: content,
                images: images
            )
        }
    }

    // MARK: - Chat Response Methods

    public static func sendNonStreamResponse(
        channel: Channel,
        modelName: String,
        chatMessages: [MLXLMCommon.Chat.Message],
        model: String,
        parameters: GenerateParameters,
        mlxTools: [ToolSpec]? = nil
    ) async throws {
        // Create UserInput with chat messages and tools
        let userInput = MLXLMCommon.UserInput(chat: chatMessages, tools: mlxTools)

        let result = try await modelPool.run(
            modelName: modelName
        ) { runner in
            try await runner.runChatNonStream(
                userInput: userInput,
                parameters: parameters
            )
        }

        // Calculate completion tokens from completion info
        let completionTokens = result.completionInfo?.generationTokenCount ?? 0

        // Convert MLX ToolCalls to OpenAI format
        let toolCalls: [ResponseToolCall]? = result.toolCalls.isEmpty ? nil : result.toolCalls.compactMap { toolCall in
            let argumentsDict = toolCall.function.arguments.mapValues { $0.anyValue }
            let argumentsJSON: String =
                if let jsonData = try? JSONSerialization.data(withJSONObject: argumentsDict),
                let jsonString = String(data: jsonData, encoding: .utf8) {
                    jsonString
                }
                else {
                    "{}"
                }

            return ResponseToolCall(
                id: "call_\(UUID().uuidString)",
                type: "function",
                function: ResponseFunction(
                    name: toolCall.function.name,
                    arguments: argumentsJSON
                )
            )
        }

        // Construct the message content for the response
        let responseMessageContent = MessageContent.text(result.output)
        let responseMessage = Message(role: "assistant", content: responseMessageContent)

        let choice = CompletionChoice(
            index: 0,
            message: responseMessage,
            finish_reason: toolCalls?.isEmpty == false ? "tool_calls" : "stop",
            tool_calls: toolCalls
        )

        // Calculate performance metrics
        let tokensPerSecond = result.completionInfo?.tokensPerSecond ?? 0.0
        let totalDuration = (result.completionInfo?.promptTime ?? 0.0) + (result.completionInfo?.generateTime ?? 0.0)

        let usage = CompletionUsage(
            prompt_tokens: result.promptTokens,
            completion_tokens: completionTokens,
            total_tokens: result.promptTokens + completionTokens,
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

        let responseHead = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        let responseBody = HTTPServerResponsePart.body(.byteBuffer(ByteBuffer(bytes: jsonData)))

        try await channel.writeAndFlush(HTTPServerResponsePart.head(responseHead))
        try await channel.writeAndFlush(responseBody)
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
    }

    public static func sendStreamResponse(
        channel: Channel,
        modelName: String,
        chatMessages: [MLXLMCommon.Chat.Message],
        model: String,
        parameters: GenerateParameters,
        tools: [ToolSpec]? = nil
    ) async throws {
        // Create UserInput with chat messages and tools
        let userInput = MLXLMCommon.UserInput(chat: chatMessages, tools: tools)

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

        // Execute model with error handling
        let result: (
            output: String,
            promptTokens: Int,
            completionInfo: GenerateCompletionInfo?,
            toolCalls: [MLXLMCommon.ToolCall]
        )

        do {
            result = try await modelPool.run(
                modelName: modelName
            ) { runner in
                try await runner.runChat(
                    userInput: userInput,
                    parameters: parameters,
                    onToken: { chunk in
                        let deltaJSON: [String: Any] = [
                            "id": chunkId,
                            "object": "chat.completion.chunk",
                            "created": timestamp,
                            "model": model,
                            "choices": [["index": 0, "delta": ["content": chunk], "finish_reason": NSNull()]]
                        ]
                        Task {
                            try? await writeSSEJSON(channel: channel, payload: deltaJSON)
                        }
                    },
                    onToolCall: { toolCall in
                        let argumentsDict = toolCall.function.arguments.mapValues { $0.anyValue }
                        let argumentsJSON: String =
                            if let jsonData = try? JSONSerialization
                                .data(withJSONObject: argumentsDict),
                                let jsonString = String(data: jsonData, encoding: .utf8)
                            {
                                jsonString
                            }
                            else {
                                "{}"
                            }

                        let toolCallDict: [String: Any] = [
                            "id": "call_\(UUID().uuidString)",
                            "type": "function",
                            "function": [
                                "name": toolCall.function.name,
                                "arguments": argumentsJSON
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
                        Task {
                            try? await writeSSEJSON(channel: channel, payload: toolCallDelta)
                        }
                    }
                )
            }
        }
        catch {
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

        // Send final chunk with usage information
        let finalCompletionTokens = result.completionInfo?.generationTokenCount ?? 0
        let tokensPerSecond = result.completionInfo?.tokensPerSecond ?? 0.0
        let totalDuration = (result.completionInfo?.promptTime ?? 0.0) + (result.completionInfo?.generateTime ?? 0.0)

        let finishJSON: [String: Any] = [
            "id": chunkId,
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": model,
            "choices": [["index": 0, "delta": [:], "finish_reason": "stop"]],
            "usage": [
                "prompt_tokens": result.promptTokens,
                "completion_tokens": finalCompletionTokens,
                "total_tokens": result.promptTokens + finalCompletionTokens,
                "response_token/s": tokensPerSecond,
                "total_duration": totalDuration
            ]
        ]

        try await writeSSEJSON(channel: channel, payload: finishJSON)
        try await writeSSELine(channel: channel, line: "data: [DONE]\n\n")
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
    }

    // MARK: - Helper Methods

    private static let modelPool: ModelPool = .shared

    private static func sendFullResponse(
        channel: Channel,
        data: Data,
        status: HTTPResponseStatus,
        version: HTTPVersion
    ) async throws {
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json")
        headers.add(name: "Content-Length", value: String(data.count))

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

    private static func parsePayload(_ buffer: ByteBuffer?) -> CompletionRequest? {
        guard let buffer,
              let data = buffer.getBytes(at: buffer.readerIndex, length: buffer.readableBytes)
        else {
            return nil
        }

        return try? JSONDecoder().decode(CompletionRequest.self, from: Data(data))
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

    public var errorDescription: String? {
        switch self {
        case let .invalidRole(role):
            "Invalid role: \(role). Must be 'system', 'user', or 'assistant'"
        }
    }
}

// MARK: - MessageContent Extensions

public extension CompletionsHandler.MessageContent {
    init(from decoder: Decoder) throws {
        if let text = try? String(from: decoder) {
            self = .text(text)
        }
        else {
            let parts = try [CompletionsHandler.ContentPart](from: decoder)
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
        }
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

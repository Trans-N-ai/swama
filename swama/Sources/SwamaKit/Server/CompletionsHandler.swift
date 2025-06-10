//
//  CompletionsHandler.swift
//  SwamaKit
//

import Foundation
@preconcurrency import MLXLMCommon
import NIOCore
import NIOHTTP1

public enum CompletionsHandler {
    // MARK: Public

    public struct CompletionRequest: Decodable, Sendable {
        let model: String
        let messages: [Message]
        let temperature: Float?
        let top_p: Float?
        let max_tokens: Int?
        let stream: Bool?
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
                return text
            case let .multimodal(parts):
                return parts.compactMap { part in
                    if case let .text(text) = part {
                        return text
                    }
                    return nil
                }.joined(separator: " ")
            }
        }
        
        var imageURLs: [String] {
            switch self {
            case .text:
                return []
            case let .multimodal(parts):
                return parts.compactMap { part in
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
            case type, text, image_url
        }
    }
    
    public enum ContentPartValue: Decodable, Encodable, Sendable {
        case text(String)
        case imageURL(ImageURL)
        
        enum CodingKeys: String, CodingKey {
            case type, text, image_url
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
                throw DecodingError.dataCorruptedError(forKey: .type, in: container, debugDescription: "Invalid content part type")
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
    }

    public struct CompletionUsage: Encodable, Sendable {
        let prompt_tokens: Int
        let completion_tokens: Int
        let total_tokens: Int
        let response_token_s: Double?
        let total_duration: Double?
        
        private enum CodingKeys: String, CodingKey {
            case prompt_tokens, completion_tokens, total_tokens
            case response_token_s = "response_token/s"
            case total_duration
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

            // Convert OpenAI messages to MLX Chat.Message format
            let chatMessages = try convertToMLXChatMessages(payload.messages)

            let parameters = GenerateParameters(
                maxTokens: payload.max_tokens,
                temperature: payload.temperature ?? 0.6,
                topP: payload.top_p ?? 1.0
            )

            if payload.stream == true {
                try await sendStreamResponse(
                    channel: channel,
                    modelName: resolvedModelName,
                    chatMessages: chatMessages,
                    model: payload.model,
                    parameters: parameters
                )
            }
            else {
                try await sendNonStreamResponse(
                    channel: channel,
                    modelName: resolvedModelName,
                    chatMessages: chatMessages,
                    model: payload.model,
                    parameters: parameters
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
                message: "Completions request failed"
            )
        }
    }

    // MARK: - Chat Message Conversion
    
    private static func convertToMLXChatMessages(_ messages: [Message]) throws -> [MLXLMCommon.Chat.Message] {
        return try messages.map { message in
            let role: MLXLMCommon.Chat.Message.Role
            switch message.role {
            case "system":
                role = .system
            case "user":
                role = .user
            case "assistant":
                role = .assistant
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
        parameters: GenerateParameters
    ) async throws {
        let (output, promptTokens, completionInfo) = try await modelPool.run(
            modelName: modelName
        ) { runner in
            try await runner.runWithChatUsageAndPerformance(
                chatMessages: chatMessages,
                parameters: parameters
            )
        }

        // Calculate completion tokens from completion info
        let completionTokens = completionInfo?.generationTokenCount ?? 0

        // Construct the message content for the response
        let responseMessageContent = MessageContent.text(output)
        let responseMessage = Message(role: "assistant", content: responseMessageContent)
        
        let choice = CompletionChoice(
            index: 0,
            message: responseMessage,
            finish_reason: "stop"
        )
        
        // Calculate performance metrics
        let tokensPerSecond = completionInfo?.tokensPerSecond ?? 0.0
        let totalDuration = (completionInfo?.promptTime ?? 0.0) + (completionInfo?.generateTime ?? 0.0)
        
        let usage = CompletionUsage(
            prompt_tokens: promptTokens,
            completion_tokens: completionTokens,
            total_tokens: promptTokens + completionTokens,
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
        parameters: GenerateParameters
    ) async throws {
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

        let (promptTokens, completionInfo) = try await modelPool.run(
            modelName: modelName
        ) { runner in
            try await runner.runWithChatStream(
                chatMessages: chatMessages,
                parameters: parameters
            ) { chunk in
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
            }
        }
        
        // Send final chunk with usage information
        let finalCompletionTokens = completionInfo?.generationTokenCount ?? 0
        let tokensPerSecond = completionInfo?.tokensPerSecond ?? 0.0
        let totalDuration = (completionInfo?.promptTime ?? 0.0) + (completionInfo?.generateTime ?? 0.0)
        
        let finishJSON: [String: Any] = [
            "id": chunkId,
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": model,
            "choices": [["index": 0, "delta": [:], "finish_reason": "stop"]],
            "usage": [
                "prompt_tokens": promptTokens,
                "completion_tokens": finalCompletionTokens,
                "total_tokens": promptTokens + finalCompletionTokens,
                "response_token/s": tokensPerSecond,
                "total_duration": totalDuration
            ]
        ]

        try await writeSSEJSON(channel: channel, payload: finishJSON)
        try await writeSSELine(channel: channel, line: "data: [DONE]\n\n")
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
    }

    // MARK: - Helper Methods

    private static let modelPool = ModelPool.shared

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
        guard let buffer = buffer,
              let data = buffer.getBytes(at: buffer.readerIndex, length: buffer.readableBytes)
        else {
            return nil
        }

        return try? JSONDecoder().decode(CompletionRequest.self, from: Data(data))
    }

    private static func writeSSEJSON(channel: Channel, payload: [String: Any]) async throws {
        let jsonData = try JSONSerialization.data(withJSONObject: payload)
        guard let jsonString = String(data: jsonData, encoding: .utf8) else {
            throw NSError(domain: "EncodingError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to encode JSON to string"])
        }
        
        try await writeSSELine(channel: channel, line: "data: \(jsonString)\n\n")
    }

    private static func writeSSELine(channel: Channel, line: String) async throws {
        var buffer = channel.allocator.buffer(capacity: line.utf8.count)
        buffer.writeString(line)
        try await channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(buffer)))
    }
}

enum CompletionsError: Error, LocalizedError {
    case invalidRole(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidRole(role):
            return "Invalid role: \(role). Must be 'system', 'user', or 'assistant'"
        }
    }
}

// MARK: - MessageContent Extensions

extension CompletionsHandler.MessageContent {
    public init(from decoder: Decoder) throws {
        if let text = try? String(from: decoder) {
            self = .text(text)
        } else {
            let parts = try [CompletionsHandler.ContentPart](from: decoder)
            let convertedParts = parts.map { part in
                if let text = part.text {
                    return CompletionsHandler.ContentPartValue.text(text)
                } else if let imageURL = part.image_url {
                    return CompletionsHandler.ContentPartValue.imageURL(imageURL)
                } else {
                    return CompletionsHandler.ContentPartValue.text("")
                }
            }
            self = .multimodal(convertedParts)
        }
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case let .text(text):
            try text.encode(to: encoder)
        case let .multimodal(parts):
            try parts.encode(to: encoder)
        }
    }
}

//
//  CompletionsHandler.swift
//  SwamaKit
//

import Foundation
import MLXLMCommon
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
        case multimodal([ContentPart])

        // MARK: Lifecycle

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()

            if let stringContent = try? container.decode(String.self) {
                self = .text(stringContent)
                return
            }

            if let arrayContent = try? container.decode([ContentPart].self) {
                self = .multimodal(arrayContent)
                return
            }

            throw DecodingError.typeMismatch(
                MessageContent.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Content must be a string or an array of content parts"
                )
            )
        }

        // MARK: Public

        /// Custom Encodable conformance if needed, or rely on synthesized
        public func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case let .text(string):
                try container.encode(string)
            case let .multimodal(parts):
                try container.encode(parts)
            }
        }

        // MARK: Internal

        var textContent: String {
            switch self {
            case let .text(string):
                string
            case let .multimodal(parts):
                parts.compactMap { part in
                    if case let .text(textContent) = part {
                        return textContent.text
                    }
                    return nil
                }
                .joined()
            }
        }

        var hasImages: Bool {
            switch self {
            case .text:
                false
            case let .multimodal(parts):
                parts.contains { part in
                    if case .imageURL = part {
                        return true
                    }
                    return false
                }
            }
        }

        var imageURLs: [String] {
            switch self {
            case .text:
                []
            case let .multimodal(parts):
                parts.compactMap { part in
                    if case let .imageURL(imageURLContent) = part {
                        return imageURLContent.url
                    }
                    return nil
                }
            }
        }
    }

    public enum ContentPart: Decodable, Encodable, Sendable {
        case text(TextContent)
        case imageURL(ImageURL)

        // MARK: Lifecycle

        public init(from decoder: Decoder) throws {
            let typeContainer = try decoder.container(keyedBy: CodingKeys.self)
            let type = try typeContainer.decode(String.self, forKey: .type)

            switch type {
            case "text":
                let textData = try TextContent(from: decoder)
                self = .text(textData)

            case "image_url":
                let imageURLData = try ImageURL(from: decoder)
                self = .imageURL(imageURLData)

            default:
                throw DecodingError.dataCorruptedError(
                    forKey: .type,
                    in: typeContainer,
                    debugDescription: "Invalid content part type:\(type)"
                )
            }
        }

        // MARK: Public

        /// Custom Encodable conformance for ContentPart
        public func encode(to encoder: Encoder) throws {
            var singleValueContainer = encoder.singleValueContainer()
            switch self {
            case let .text(textContent):
                // TextContent will encode itself as {"type": "text", "text": "..."}
                try singleValueContainer.encode(textContent)
            case let .imageURL(imageURL):
                // ImageURL will encode itself as {"type": "image_url", "image_url": {...}}
                try singleValueContainer.encode(imageURL)
            }
        }

        // MARK: Private

        private enum CodingKeys: String, CodingKey {
            case type
        }
    }

    public struct TextContent: Decodable, Encodable, Sendable {
        let type: String
        let text: String
    }

    public struct ImageURL: Decodable, Encodable, Sendable {
        // MARK: Internal

        let type: String
        let imageURL: ImageURLData

        var url: String {
            imageURL.url
        }

        // MARK: Private

        private enum CodingKeys: String, CodingKey {
            case type
            case imageURL = "image_url"
        }
    }

    public struct ImageURLData: Decodable, Encodable, Sendable {
        let url: String
        let detail: String?
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
    }

    public static func handle(
        requestHead _: HTTPRequestHead,
        body: ByteBuffer?,
        channel: Channel
    ) async {
        do {
            guard let payload = try? parsePayload(body),
                  let lastUserMessage = payload.messages.last(where: { $0.role == "user" })
            else {
                try? await respondError(
                    channel: channel,
                    status: .badRequest,
                    message: "Invalid request payload or missing user message"
                )
                return
            }

            let container = try await modelPool.get(modelName: payload.model)
            let runner = ModelRunner(container: container)

            let promptText = lastUserMessage.content.textContent
            let imageURLs = lastUserMessage.content.imageURLs

            var imagesData: [Data]?
            if !imageURLs.isEmpty {
                imagesData = try await processImageURLs(imageURLs)
            }

            let parameters = GenerateParameters(
                maxTokens: payload.max_tokens,
                temperature: payload.temperature ?? 0.6,
                topP: payload.top_p ?? 1.0
            )

            if payload.stream == true {
                try await sendStreamResponse(
                    channel: channel,
                    runner: runner,
                    prompt: promptText,
                    images: imagesData,
                    model: payload.model,
                    parameters: parameters
                )
            }
            else {
                try await sendNonStreamResponse(
                    channel: channel,
                    runner: runner,
                    prompt: promptText,
                    images: imagesData,
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

    public static func sendNonStreamResponse(
        channel: Channel,
        runner: ModelRunner,
        prompt: String,
        images: [Data]? = nil,
        model: String,
        parameters: GenerateParameters
    ) async throws {
        let (output, promptTokens, completionTokens) = try await runner.runWithUsage(
            prompt: prompt,
            images: images,
            parameters: parameters
        ) // Pass images

        // Construct the message content for the response
        let responseMessageContent = MessageContent.text(output)
        let responseMessage = Message(role: "assistant", content: responseMessageContent)

        let response = CompletionResponse(
            id: "chatcmpl-\\(UUID().uuidString)",
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: model,
            choices: [
                CompletionChoice(
                    index: 0,
                    message: responseMessage,
                    finish_reason: "stop"
                )
            ],
            usage: CompletionUsage(
                prompt_tokens: promptTokens,
                completion_tokens: completionTokens,
                total_tokens: promptTokens + completionTokens
            )
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(response)

        try await sendFullResponse(channel: channel, data: data, status: .ok, version: .http1_1)
    }

    public static func sendStreamResponse(
        channel: Channel,
        runner: ModelRunner,
        prompt: String,
        images: [Data]? = nil,
        model: String,
        parameters: GenerateParameters
    ) async throws {
        // Set up streaming headers
        let headers = HTTPHeaders([
            ("Content-Type", "text/event-stream"),
            ("Cache-Control", "no-cache"),
            ("Connection", "keep-alive")
        ])
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)

        try await channel.writeAndFlush(HTTPServerResponsePart.head(head))

        let (promptTokens, completionInfo) = try await runner.runStreamWithUsage(
            prompt: prompt,
            images: images,
            parameters: parameters
        ) { chunk in
            let deltaJSON: [String: Any] = [
                "id": "chatcmpl-\(UUID().uuidString)",
                "object": "chat.completion.chunk",
                "created": Int(Date().timeIntervalSince1970),
                "model": model,
                "choices": [["index": 0, "delta": ["content": chunk], "finish_reason": nil]]
            ]
            Task {
                try? await writeSSEJSON(channel: channel, payload: deltaJSON)
            }
        }

        // Send final usage information
        let finalCompletionTokens = completionInfo?.generationTokenCount ?? 0
        let totalDuration = completionInfo?.generateTime ?? 0.0
        var tokensPerSecond = completionInfo?.tokensPerSecond ?? 0.0

        if tokensPerSecond == 0.0, totalDuration > 0, finalCompletionTokens > 0 {
            tokensPerSecond = Double(finalCompletionTokens) / totalDuration
        }

        // Send final message with finish_reason
        let finishJSON: [String: Any] = [
            "id": "chatcmpl-\(UUID().uuidString)",
            "object": "chat.completion.chunk",
            "created": Int(Date().timeIntervalSince1970),
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

    // MARK: Private

    private static func parsePayload(_ body: ByteBuffer?) throws -> CompletionRequest {
        guard var buffer = body, let json = buffer.readString(length: buffer.readableBytes) else {
            throw NSError(domain: "InvalidBody", code: 400)
        }

        return try JSONDecoder().decode(CompletionRequest.self, from: Data(json.utf8))
    }

    /// Function to download image data from a URL
    private static func fetchImageData(from url: URL) async throws -> Data {
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }

        // Check MIME type for image validation
        if let mimeType = httpResponse.mimeType {
            let supportedImageTypes = [
                "image/jpeg",
                "image/jpg",
                "image/png",
                "image/gif",
                "image/webp",
                "image/bmp",
                "image/tiff"
            ]
            if !supportedImageTypes.contains(mimeType.lowercased()) {
                // Log warning only in debug mode
                if ProcessInfo.processInfo.environment["DEBUG"] != nil {
                    print("Warning: Unsupported MIME type '\(mimeType)' for image URL: \(url)")
                }
            }
        }

        return data
    }

    /// Extract image processing logic to reduce code duplication
    private static func processImageURLs(_ imageURLs: [String]) async throws -> [Data]? {
        var orderedImageData = [Data?](repeating: nil, count: imageURLs.count)

        try await withThrowingTaskGroup(of: (index: Int, data: Data?)?.self) { group in
            for (index, urlString) in imageURLs.enumerated() {
                if urlString.starts(with: "data:") {
                    group.addTask {
                        guard let commaIndex = urlString.firstIndex(of: ",") else {
                            return (index, nil)
                        }

                        let base64String = String(urlString.suffix(from: urlString.index(after: commaIndex)))

                        if let data = Data(base64Encoded: base64String, options: .ignoreUnknownCharacters) {
                            return (index, data)
                        }
                        else {
                            return (index, nil)
                        }
                    }
                }
                else if let url = URL(string: urlString) {
                    group.addTask {
                        do {
                            let data = try await fetchImageData(from: url)
                            return (index, data)
                        }
                        catch {
                            if ProcessInfo.processInfo.environment["DEBUG"] != nil {
                                print("Failed to download image from \(url): \(error.localizedDescription)")
                            }
                            return (index, nil)
                        }
                    }
                }
            }

            for try await result in group {
                if let res = result {
                    orderedImageData[res.index] = res.data
                }
            }
        }

        let processedImages = orderedImageData.compactMap(\.self)
        return processedImages.isEmpty ? nil : processedImages
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
        headers.add(name: "Connection", value: "close")

        let head = HTTPResponseHead(version: version, status: status, headers: headers)
        // channel.write is not async and does not throw by itself.
        // Use _ = to explicitly ignore the result if it's a non-void function.
        _ = channel.write(HTTPServerResponsePart.head(head))
        _ = channel.write(HTTPServerResponsePart.body(.byteBuffer(ByteBuffer(bytes: data))))
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
    }

    private static func respondError(channel: Channel, status: HTTPResponseStatus, message: String) async throws {
        let errorJSON = ["error": message]
        let data = try JSONSerialization.data(withJSONObject: errorJSON)
        // Assuming HTTP/1.1 for error responses
        try await sendFullResponse(channel: channel, data: data, status: status, version: .http1_1)
    }

    private static func writeSSEJSON(channel: Channel, payload: [String: Any]) async throws {
        let data = try JSONSerialization.data(withJSONObject: payload)
        let line = "data: \(String(data: data, encoding: .utf8)!)\n\n"
        try await writeSSELine(channel: channel, line: line)
    }

    private static func writeSSELine(channel: Channel, line: String) async throws {
        try await channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(ByteBuffer(string: line))))
    }
}

import Foundation
import NIOCore
import NIOHTTP1

// MARK: - EmbeddingsRequest

public struct EmbeddingsRequest: Codable {
    public let input: EmbeddingInput
    public let model: String
    public let encodingFormat: String?
    public let dimensions: Int?
    public let user: String?

    enum CodingKeys: String, CodingKey {
        case input
        case model
        case encodingFormat = "encoding_format"
        case dimensions, user
    }
}

// MARK: - EmbeddingInput

public enum EmbeddingInput: Codable {
    case string(String)
    case strings([String])
    case tokens([Int])
    case tokenArrays([[Int]])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()

        if let string = try? container.decode(String.self) {
            self = .string(string)
        }
        else if let strings = try? container.decode([String].self) {
            self = .strings(strings)
        }
        else if let tokens = try? container.decode([Int].self) {
            self = .tokens(tokens)
        }
        else if let tokenArrays = try? container.decode([[Int]].self) {
            self = .tokenArrays(tokenArrays)
        }
        else {
            throw DecodingError.typeMismatch(
                EmbeddingInput.self,
                DecodingError.Context(
                    codingPath: decoder.codingPath,
                    debugDescription: "Expected string, array of strings, array of integers, or array of integer arrays"
                )
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .string(string):
            try container.encode(string)
        case let .strings(strings):
            try container.encode(strings)
        case let .tokens(tokens):
            try container.encode(tokens)
        case let .tokenArrays(tokenArrays):
            try container.encode(tokenArrays)
        }
    }

    public var asStringArray: [String] {
        switch self {
        case let .string(string):
            [string]
        case let .strings(strings):
            strings
        case .tokenArrays,
             .tokens:
            // For token inputs, we'd need to detokenize, but for now return empty
            // This would require access to the tokenizer
            []
        }
    }
}

// MARK: - EmbeddingsResponse

public struct EmbeddingsResponse: Codable {
    public let object: String
    public let data: [EmbeddingData]
    public let model: String
    public let usage: Usage

    public struct EmbeddingData: Codable {
        public let object: String
        public let embedding: [Float]
        public let index: Int
    }

    public struct Usage: Codable {
        public let promptTokens: Int
        public let totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case totalTokens = "total_tokens"
        }
    }
}

// MARK: - EmbeddingsHandler

public enum EmbeddingsHandler {
    public static func handle(
        requestHead: HTTPRequestHead,
        body: ByteBuffer,
        channel: Channel
    ) async {
        do {
            // Parse JSON body
            var mutableBody = body
            guard let bodyBytes = mutableBody.readBytes(length: body.readableBytes) else {
                try await sendErrorResponse(
                    channel: channel,
                    status: .badRequest,
                    error: "Invalid request body",
                    requestVersion: requestHead.version
                )
                return
            }

            let bodyData = Data(bodyBytes)

            let request = try JSONDecoder().decode(EmbeddingsRequest.self, from: bodyData)

            // Get input strings
            let inputs = request.input.asStringArray
            guard !inputs.isEmpty else {
                try await sendErrorResponse(
                    channel: channel,
                    status: .badRequest,
                    error: "No valid input provided",
                    requestVersion: requestHead.version
                )
                return
            }

            // Generate embeddings using MLXEmbedders
            let (embeddings, usage): ([[Float]], EmbeddingUsage)
            do {
                // Use standard MLXEmbedders for all models
                let embeddingRunner: EmbeddingRunner
                // Check if model is already loaded in the pool
                if let existingRunner = await modelPool.getEmbeddingRunner(for: request.model) {
                    embeddingRunner = existingRunner
                }
                else {
                    // Load the embedding model
                    let container = try await loadEmbeddingModelContainer(modelName: request.model)
                    embeddingRunner = EmbeddingRunner(container: container)
                    await modelPool.setEmbeddingRunner(embeddingRunner, for: request.model)
                }

                (embeddings, usage) = try await embeddingRunner.generateEmbeddings(inputs: inputs)
            }
            catch {
                try await sendErrorResponse(
                    channel: channel,
                    status: .internalServerError,
                    error: "Failed to generate embeddings with model '\(request.model)': \(error.localizedDescription)",
                    requestVersion: requestHead.version
                )
                return
            }

            // Create response
            let embeddingData = embeddings.enumerated().map { index, embedding in
                EmbeddingsResponse.EmbeddingData(
                    object: "embedding",
                    embedding: embedding,
                    index: index
                )
            }

            let response = EmbeddingsResponse(
                object: "list",
                data: embeddingData,
                model: request.model,
                usage: EmbeddingsResponse.Usage(
                    promptTokens: usage.promptTokens,
                    totalTokens: usage.totalTokens
                )
            )

            // Send response
            try await sendJSONResponse(
                channel: channel,
                response: response,
                requestVersion: requestHead.version
            )
        }
        catch {
            NSLog("SwamaKit.EmbeddingsHandler Error: \(error)")
            try? await sendErrorResponse(
                channel: channel,
                status: .internalServerError,
                error: "Internal server error: \(error.localizedDescription)",
                requestVersion: requestHead.version
            )
        }
    }

    private static func sendJSONResponse(
        channel: Channel,
        response: some Codable,
        requestVersion: HTTPVersion
    ) async throws {
        let jsonData = try JSONEncoder().encode(response)

        var buffer = channel.allocator.buffer(capacity: jsonData.count)
        buffer.writeBytes(jsonData)

        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json")
        headers.add(name: "Content-Length", value: "\(buffer.readableBytes)")
        headers.add(name: "Connection", value: "close")

        try await channel.writeAndFlush(
            HTTPServerResponsePart.head(HTTPResponseHead(
                version: requestVersion,
                status: .ok,
                headers: headers
            ))
        )

        try await channel.writeAndFlush(
            HTTPServerResponsePart.body(.byteBuffer(buffer))
        )

        try await channel.writeAndFlush(
            HTTPServerResponsePart.end(nil)
        )
    }

    private static func sendErrorResponse(
        channel: Channel,
        status: HTTPResponseStatus,
        error: String,
        requestVersion: HTTPVersion
    ) async throws {
        let errorResponse = [
            "error": [
                "message": error,
                "type": "invalid_request_error"
            ]
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: errorResponse)

        var buffer = channel.allocator.buffer(capacity: jsonData.count)
        buffer.writeBytes(jsonData)

        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json")
        headers.add(name: "Content-Length", value: "\(buffer.readableBytes)")
        headers.add(name: "Connection", value: "close")

        try await channel.writeAndFlush(
            HTTPServerResponsePart.head(HTTPResponseHead(
                version: requestVersion,
                status: status,
                headers: headers
            ))
        )

        try await channel.writeAndFlush(
            HTTPServerResponsePart.body(.byteBuffer(buffer))
        )

        try await channel.writeAndFlush(
            HTTPServerResponsePart.end(nil)
        )
    }
}

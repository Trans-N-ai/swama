import Foundation

// MARK: - ModelID

public struct ModelID: RawRepresentable, Hashable, Codable, Sendable, CustomStringConvertible {
    public init(rawValue: String) {
        self.rawValue = rawValue
    }

    public init(_ rawValue: String) {
        self.rawValue = rawValue
    }

    public init(from decoder: Decoder) throws {
        rawValue = try decoder.singleValueContainer().decode(String.self)
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }

    public let rawValue: String
    public var description: String {
        rawValue
    }
}

// MARK: - SwamaConfiguration

public struct SwamaConfiguration: Hashable, Codable, Sendable {
    public init(defaultContextLimit: Int? = nil) {
        self.defaultContextLimit = defaultContextLimit
    }

    public var defaultContextLimit: Int?
}

// MARK: - JSONValue

public enum JSONValue: Hashable, Codable, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        }
        else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        }
        else if let value = try? container.decode(Int.self) {
            self = .int(value)
        }
        else if let value = try? container.decode(Double.self) {
            self = .double(value)
        }
        else if let value = try? container.decode(String.self) {
            self = .string(value)
        }
        else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        }
        else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        }
        else {
            throw DecodingError.dataCorruptedError(
                in: container,
                debugDescription: "unsupported JSON value"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case let .bool(value):
            try container.encode(value)
        case let .int(value):
            try container.encode(value)
        case let .double(value):
            try container.encode(value)
        case let .string(value):
            try container.encode(value)
        case let .array(value):
            try container.encode(value)
        case let .object(value):
            try container.encode(value)
        }
    }
}

// MARK: - ToolDefinition

public struct ToolDefinition: Hashable, Codable, Sendable {
    public init(name: String, description: String? = nil, parameters: JSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    public var name: String
    public var description: String?
    public var parameters: JSONValue
}

// MARK: - ToolCall

public struct ToolCall: Hashable, Codable, Sendable {
    public init(id: String? = nil, name: String, arguments: [String: JSONValue]) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    public var id: String?
    public var name: String
    public var arguments: [String: JSONValue]
}

// MARK: - ContentPart

public enum ContentPart: Hashable, Codable, Sendable {
    case text(String)
    case imageURL(URL)
    case imageData(Data, mediaType: String)

    private enum CodingKeys: String, CodingKey {
        case type
        case text
        case url
        case data
        case mediaType = "media_type"
    }

    private enum Kind: String, Codable {
        case text
        case imageURL = "image_url"
        case imageData = "image_data"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .type) {
        case .text:
            self = try .text(container.decode(String.self, forKey: .text))
        case .imageURL:
            self = try .imageURL(container.decode(URL.self, forKey: .url))
        case .imageData:
            self = try .imageData(
                container.decode(Data.self, forKey: .data),
                mediaType: container.decode(String.self, forKey: .mediaType)
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case let .text(value):
            try container.encode(Kind.text, forKey: .type)
            try container.encode(value, forKey: .text)

        case let .imageURL(value):
            try container.encode(Kind.imageURL, forKey: .type)
            try container.encode(value, forKey: .url)

        case let .imageData(data, mediaType):
            try container.encode(Kind.imageData, forKey: .type)
            try container.encode(data, forKey: .data)
            try container.encode(mediaType, forKey: .mediaType)
        }
    }
}

// MARK: - Message

public struct Message: Hashable, Codable, Sendable {
    public enum Role: String, Codable, Sendable {
        case system
        case user
        case assistant
        case tool
    }

    public init(
        role: Role,
        content: [ContentPart],
        toolCalls: [ToolCall] = [],
        toolCallID: String? = nil
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
    }

    public init(role: Role, text: String) {
        self.init(role: role, content: [.text(text)])
    }

    public var role: Role
    public var content: [ContentPart]
    public var toolCalls: [ToolCall]
    public var toolCallID: String?
}

// MARK: - GenerationOptions

public struct GenerationOptions: Hashable, Codable, Sendable {
    public init(
        maxTokens: Int? = nil,
        temperature: Float = 0.6,
        topP: Float = 1,
        topK: Int = 0,
        minP: Float = 0,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20,
        presencePenalty: Float? = nil,
        presenceContextSize: Int = 20,
        frequencyPenalty: Float? = nil,
        frequencyContextSize: Int = 20,
        seed: UInt64? = nil,
        contextLimit: Int? = nil
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.presenceContextSize = presenceContextSize
        self.frequencyPenalty = frequencyPenalty
        self.frequencyContextSize = frequencyContextSize
        self.seed = seed
        self.contextLimit = contextLimit
    }

    public var maxTokens: Int?
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var repetitionPenalty: Float?
    public var repetitionContextSize: Int
    public var presencePenalty: Float?
    public var presenceContextSize: Int
    public var frequencyPenalty: Float?
    public var frequencyContextSize: Int
    public var seed: UInt64?
    public var contextLimit: Int?
}

// MARK: - GenerationRequest

public struct GenerationRequest: Hashable, Codable, Sendable {
    public init(
        model: ModelID,
        messages: [Message],
        options: GenerationOptions = .init(),
        tools: [ToolDefinition] = []
    ) {
        self.model = model
        self.messages = messages
        self.options = options
        self.tools = tools
    }

    public var model: ModelID
    public var messages: [Message]
    public var options: GenerationOptions
    public var tools: [ToolDefinition]
}

// MARK: - GenerationEvent

public enum GenerationEvent: Hashable, Codable, Sendable {
    case textDelta(String)
    case toolCall(ToolCall)

    private enum CodingKeys: String, CodingKey {
        case type
        case text
        case toolCall = "tool_call"
    }

    private enum Kind: String, Codable {
        case textDelta = "text_delta"
        case toolCall = "tool_call"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .type) {
        case .textDelta:
            self = try .textDelta(container.decode(String.self, forKey: .text))
        case .toolCall:
            self = try .toolCall(container.decode(ToolCall.self, forKey: .toolCall))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case let .textDelta(value):
            try container.encode(Kind.textDelta, forKey: .type)
            try container.encode(value, forKey: .text)

        case let .toolCall(value):
            try container.encode(Kind.toolCall, forKey: .type)
            try container.encode(value, forKey: .toolCall)
        }
    }
}

// MARK: - FinishReason

public enum FinishReason: String, Hashable, Codable, Sendable {
    case completed
    case length
    case toolCall = "tool_call"
    case unknown

    public init(from decoder: Decoder) throws {
        let rawValue = try decoder.singleValueContainer().decode(String.self)
        self = Self(rawValue: rawValue) ?? .unknown
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

// MARK: - Usage

public struct Usage: Hashable, Codable, Sendable {
    public init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }

    public var promptTokens: Int
    public var completionTokens: Int
    public var totalTokens: Int {
        promptTokens + completionTokens
    }
}

// MARK: - GenerationMetrics

public struct GenerationMetrics: Hashable, Codable, Sendable {
    public init(promptSeconds: Double, generationSeconds: Double, tokensPerSecond: Double) {
        self.promptSeconds = promptSeconds
        self.generationSeconds = generationSeconds
        self.tokensPerSecond = tokensPerSecond
    }

    public var promptSeconds: Double
    public var generationSeconds: Double
    public var tokensPerSecond: Double
}

// MARK: - GenerationResponse

public struct GenerationResponse: Hashable, Codable, Sendable {
    public init(
        output: String,
        toolCalls: [ToolCall],
        usage: Usage,
        finishReason: FinishReason,
        metrics: GenerationMetrics? = nil
    ) {
        self.output = output
        self.toolCalls = toolCalls
        self.usage = usage
        self.finishReason = finishReason
        self.metrics = metrics
    }

    public var output: String
    public var toolCalls: [ToolCall]
    public var usage: Usage
    public var finishReason: FinishReason
    public var metrics: GenerationMetrics?
}

// MARK: - EmbeddingRequest

public struct EmbeddingRequest: Hashable, Codable, Sendable {
    public init(model: ModelID, inputs: [String]) {
        self.model = model
        self.inputs = inputs
    }

    public var model: ModelID
    public var inputs: [String]
}

// MARK: - EmbeddingResponse

public struct EmbeddingResponse: Hashable, Codable, Sendable {
    public init(embeddings: [[Float]], usage: Usage) {
        self.embeddings = embeddings
        self.usage = usage
    }

    public var embeddings: [[Float]]
    public var usage: Usage
}

// MARK: - ModelCapabilities

public struct ModelCapabilities: Hashable, Codable, Sendable {
    public init(
        textGeneration: Bool = false,
        vision: Bool = false,
        tools: Bool = false,
        embeddings: Bool = false
    ) {
        self.textGeneration = textGeneration
        self.vision = vision
        self.tools = tools
        self.embeddings = embeddings
    }

    public var textGeneration: Bool
    public var vision: Bool
    public var tools: Bool
    public var embeddings: Bool
}

// MARK: - ModelInfo

public struct ModelInfo: Identifiable, Hashable, Codable, Sendable {
    public init(
        id: ModelID,
        created: Date,
        sizeInBytes: Int64,
        capabilities: ModelCapabilities
    ) {
        self.id = id
        self.created = created
        self.sizeInBytes = sizeInBytes
        self.capabilities = capabilities
    }

    public var id: ModelID
    public var created: Date
    public var sizeInBytes: Int64
    public var capabilities: ModelCapabilities
}

// MARK: - SwamaError

public struct SwamaError: Error, Hashable, Codable, Sendable, LocalizedError {
    public enum Code: String, Hashable, Codable, Sendable {
        case invalidRequest = "invalid_request"
        case invalidImage = "invalid_image"
        case modelNotFound = "model_not_found"
        case modelLoadFailed = "model_load_failed"
        case contextLimitExceeded = "context_limit_exceeded"
        case embeddingFailed = "embedding_failed"
        case downloadFailed = "download_failed"
        case removalFailed = "removal_failed"
        case backendFailure = "backend_failure"
    }

    public init(code: Code, message: String, model: ModelID? = nil) {
        self.code = code
        self.message = message
        self.model = model
    }

    public var code: Code
    public var message: String
    public var model: ModelID?
    public var errorDescription: String? {
        message
    }
}

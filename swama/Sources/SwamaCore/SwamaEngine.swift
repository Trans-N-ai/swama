import Foundation
import SwamaRuntime

// MARK: - SwamaEngine

public actor SwamaEngine {
    public init(configuration: SwamaConfiguration = .init()) {
        _ = Self.resourceBundle
        backend = RuntimeEngineBackend(configuration: configuration)
    }

    package init(backend: any SwamaEngineBackend) {
        _ = Self.resourceBundle
        self.backend = backend
    }

    public func generate(
        _ request: GenerationRequest,
        onEvent: (@Sendable (GenerationEvent) async throws -> Void)? = nil
    ) async throws -> GenerationResponse {
        try validate(request)
        do {
            let response = try await backend.generate(request, onEvent: onEvent)
            try Task.checkCancellation()
            return response
        }
        catch is CancellationError {
            throw CancellationError()
        }
        catch let error as SwamaError {
            throw error
        }
        catch {
            throw SwamaError(
                code: .backendFailure,
                message: "The local model backend failed.",
                model: request.model
            )
        }
    }

    public func embed(_ request: EmbeddingRequest) async throws -> EmbeddingResponse {
        try validate(request)
        do {
            let response = try await backend.embed(request)
            try Task.checkCancellation()
            return response
        }
        catch is CancellationError {
            throw CancellationError()
        }
        catch let error as SwamaError {
            throw error
        }
        catch {
            throw SwamaError(
                code: .backendFailure,
                message: "The local embedding backend failed.",
                model: request.model
            )
        }
    }

    public func models() async throws -> [ModelInfo] {
        try await backend.models()
    }

    public func fetch(_ model: ModelID) async throws {
        _ = try await fetchResolved(model)
    }

    package func fetchResolved(_ model: ModelID) async throws -> ModelID {
        try validate(model)
        return try await backend.fetch(model)
    }

    public func remove(_ model: ModelID) async throws {
        try validate(model)
        try await backend.remove(model)
    }

    public func clearCache(for model: ModelID) async {
        guard RuntimeCoreEngine.isValidModelID(model.rawValue) else {
            return
        }

        await backend.clearCache(for: model)
    }

    public func clearCache() async {
        await backend.clearCache()
    }

    package static func withCLIDiagnostics<T>(
        operation: () async throws -> T
    ) async throws -> T {
        try await SwamaDiagnostics.withSession(mode: .cli, operation: operation)
    }

    private let backend: any SwamaEngineBackend
    private static let resourceBundle: Bundle = .module

    private func validate(_ request: GenerationRequest) throws {
        try validate(request.model)
        guard request.messages.isEmpty == false else {
            throw invalidRequest("Generation requires at least one message.", model: request.model)
        }

        for message in request.messages {
            try validate(message, model: request.model)
            for part in message.content {
                if case let .imageData(data, mediaType) = part,
                   data.isEmpty || mediaType.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                {
                    throw invalidRequest("Image data and media type must be non-empty.", model: request.model)
                }
            }
        }
        for tool in request.tools {
            guard tool.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false,
                  case .object = tool.parameters
            else {
                throw invalidRequest("Tools require a name and an object parameter schema.", model: request.model)
            }
        }
        let options = request.options
        guard options.maxTokens.map({ $0 > 0 }) ?? true,
              options.temperature.isFinite,
              options.temperature >= 0,
              options.topP.isFinite,
              (0 ... 1).contains(options.topP),
              options.topK >= 0,
              options.minP.isFinite,
              (0 ... 1).contains(options.minP),
              options.repetitionPenalty.map({ $0.isFinite && $0 >= 0 }) ?? true,
              options.repetitionContextSize > 0,
              options.presencePenalty.map({ $0.isFinite && (-2 ... 2).contains($0) }) ?? true,
              options.presenceContextSize > 0,
              options.frequencyPenalty.map({ $0.isFinite && (-2 ... 2).contains($0) }) ?? true,
              options.frequencyContextSize > 0,
              options.contextLimit.map({ $0 > 0 }) ?? true
        else {
            throw invalidRequest("Generation options are outside their supported ranges.", model: request.model)
        }
    }

    private func validate(_ request: EmbeddingRequest) throws {
        try validate(request.model)
        guard request.inputs.isEmpty == false,
              request.inputs.allSatisfy({ $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false })
        else {
            throw invalidRequest("Embedding inputs must be non-empty.", model: request.model)
        }
    }

    private func validate(_ model: ModelID) throws {
        guard RuntimeCoreEngine.isValidModelID(model.rawValue) else {
            throw invalidRequest("Model identifier is invalid.", model: nil)
        }
    }

    private func validate(_ message: Message, model: ModelID) throws {
        let toolCallID = message.toolCallID?.trimmingCharacters(in: .whitespacesAndNewlines)
        let isValid =
            switch message.role {
            case .system,
                 .user:
                message.content.isEmpty == false && message.toolCalls.isEmpty && message.toolCallID == nil
            case .assistant:
                (message.content.isEmpty == false || message.toolCalls.isEmpty == false) && message.toolCallID == nil
            case .tool:
                message.content.isEmpty == false &&
                    message.content.allSatisfy(\.isText) &&
                    message.toolCalls.isEmpty &&
                    toolCallID?.isEmpty == false
            }
        guard isValid else {
            throw invalidRequest("Message fields are invalid for its role.", model: model)
        }
    }

    private func invalidRequest(_ message: String, model: ModelID?) -> SwamaError {
        .init(code: .invalidRequest, message: message, model: model)
    }
}

private extension ContentPart {
    var isText: Bool {
        if case .text = self {
            return true
        }
        return false
    }
}

// MARK: - SwamaEngineBackend

package protocol SwamaEngineBackend: Sendable {
    func generate(
        _ request: GenerationRequest,
        onEvent: (@Sendable (GenerationEvent) async throws -> Void)?
    ) async throws -> GenerationResponse

    func embed(_ request: EmbeddingRequest) async throws -> EmbeddingResponse
    func models() async throws -> [ModelInfo]
    func fetch(_ model: ModelID) async throws -> ModelID
    func remove(_ model: ModelID) async throws
    func clearCache(for model: ModelID) async
    func clearCache() async
}

// MARK: - RuntimeEngineBackend

private struct RuntimeEngineBackend: SwamaEngineBackend {
    init(configuration: SwamaConfiguration) {
        runtime = RuntimeCoreEngine(defaultContextLimit: configuration.defaultContextLimit)
    }

    func generate(
        _ request: GenerationRequest,
        onEvent: (@Sendable (GenerationEvent) async throws -> Void)?
    ) async throws -> GenerationResponse {
        do {
            let result = try await runtime.generate(request.runtimeValue) { event in
                try await onEvent?(event.coreValue)
            }
            return result.coreValue
        }
        catch is CancellationError {
            throw CancellationError()
        }
        catch let error as RuntimeCoreError {
            throw error.coreValue
        }
    }

    func embed(_ request: EmbeddingRequest) async throws -> EmbeddingResponse {
        do {
            return try await runtime.embed(.init(
                model: request.model.rawValue,
                inputs: request.inputs
            ))
            .coreValue
        }
        catch is CancellationError {
            throw CancellationError()
        }
        catch let error as RuntimeCoreError {
            throw error.coreValue
        }
    }

    func models() async throws -> [ModelInfo] {
        await runtime.models().map(\.coreValue)
    }

    func fetch(_ model: ModelID) async throws -> ModelID {
        do {
            let resolved = try await runtime.fetch(model.rawValue)
            return .init(resolved)
        }
        catch let error as RuntimeCoreError {
            throw error.coreValue
        }
    }

    func remove(_ model: ModelID) async throws {
        do {
            try await runtime.remove(model.rawValue)
        }
        catch let error as RuntimeCoreError {
            throw error.coreValue
        }
    }

    func clearCache(for model: ModelID) async {
        await runtime.clearCache(for: model.rawValue)
    }

    func clearCache() async {
        await runtime.clearCache()
    }

    private let runtime: RuntimeCoreEngine
}

// MARK: - Runtime mapping

private extension GenerationRequest {
    var runtimeValue: RuntimeGenerationRequest {
        .init(
            model: model.rawValue,
            messages: messages.map(\.runtimeValue),
            options: options.runtimeValue,
            tools: tools.map(\.runtimeValue)
        )
    }
}

private extension Message {
    var runtimeValue: RuntimeMessage {
        .init(
            role: role.runtimeValue,
            content: content.map(\.runtimeValue),
            toolCalls: toolCalls.map(\.runtimeValue),
            toolCallID: toolCallID
        )
    }
}

private extension Message.Role {
    var runtimeValue: RuntimeMessageRole {
        switch self {
        case .system:
            .system
        case .user:
            .user
        case .assistant:
            .assistant
        case .tool:
            .tool
        }
    }
}

private extension ContentPart {
    var runtimeValue: RuntimeContentPart {
        switch self {
        case let .text(value):
            .text(value)
        case let .imageURL(value):
            .imageURL(value)
        case let .imageData(data, mediaType):
            .imageData(data, mediaType: mediaType)
        }
    }
}

private extension JSONValue {
    var runtimeValue: RuntimeJSONValue {
        switch self {
        case .null:
            .null
        case let .bool(value):
            .bool(value)
        case let .int(value):
            .int(value)
        case let .double(value):
            .double(value)
        case let .string(value):
            .string(value)
        case let .array(value):
            .array(value.map(\.runtimeValue))
        case let .object(value):
            .object(value.mapValues(\.runtimeValue))
        }
    }
}

private extension RuntimeJSONValue {
    var coreValue: JSONValue {
        switch self {
        case .null:
            .null
        case let .bool(value):
            .bool(value)
        case let .int(value):
            .int(value)
        case let .double(value):
            .double(value)
        case let .string(value):
            .string(value)
        case let .array(value):
            .array(value.map(\.coreValue))
        case let .object(value):
            .object(value.mapValues(\.coreValue))
        }
    }
}

private extension ToolDefinition {
    var runtimeValue: RuntimeToolDefinition {
        .init(name: name, description: description, parameters: parameters.runtimeValue)
    }
}

private extension ToolCall {
    var runtimeValue: RuntimeToolCall {
        .init(id: id, name: name, arguments: arguments.mapValues(\.runtimeValue))
    }
}

private extension RuntimeToolCall {
    var coreValue: ToolCall {
        .init(id: id, name: name, arguments: arguments.mapValues(\.coreValue))
    }
}

private extension GenerationOptions {
    var runtimeValue: RuntimeGenerationOptions {
        .init(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP,
            repetitionPenalty: repetitionPenalty,
            repetitionContextSize: repetitionContextSize,
            presencePenalty: presencePenalty,
            presenceContextSize: presenceContextSize,
            frequencyPenalty: frequencyPenalty,
            frequencyContextSize: frequencyContextSize,
            seed: seed,
            contextLimit: contextLimit
        )
    }
}

private extension RuntimeGenerationEvent {
    var coreValue: GenerationEvent {
        switch self {
        case let .textDelta(value):
            .textDelta(value)
        case let .toolCall(value):
            .toolCall(value.coreValue)
        }
    }
}

private extension RuntimeGenerationResult {
    var coreValue: GenerationResponse {
        .init(
            output: output,
            toolCalls: toolCalls.map(\.coreValue),
            usage: usage.coreValue,
            finishReason: finishReason.coreValue,
            metrics: metrics?.coreValue
        )
    }
}

private extension RuntimeUsage {
    var coreValue: Usage {
        .init(promptTokens: promptTokens, completionTokens: completionTokens)
    }
}

private extension RuntimeFinishReason {
    var coreValue: FinishReason {
        switch self {
        case .completed:
            .completed
        case .length:
            .length
        case .toolCall:
            .toolCall
        }
    }
}

private extension RuntimeGenerationMetrics {
    var coreValue: GenerationMetrics {
        .init(
            promptSeconds: promptSeconds,
            generationSeconds: generationSeconds,
            tokensPerSecond: tokensPerSecond
        )
    }
}

private extension RuntimeEmbeddingResult {
    var coreValue: EmbeddingResponse {
        .init(embeddings: embeddings, usage: usage.coreValue)
    }
}

private extension RuntimeModelInfo {
    var coreValue: ModelInfo {
        .init(
            id: .init(id),
            created: Date(timeIntervalSince1970: TimeInterval(created)),
            sizeInBytes: sizeInBytes,
            capabilities: capabilities.coreValue
        )
    }
}

private extension RuntimeModelCapabilities {
    var coreValue: ModelCapabilities {
        .init(
            textGeneration: textGeneration,
            vision: vision,
            tools: tools,
            embeddings: embeddings
        )
    }
}

private extension RuntimeCoreError {
    var coreValue: SwamaError {
        let coreCode: SwamaError.Code =
            switch code {
            case .invalidRequest:
                .invalidRequest
            case .invalidImage:
                .invalidImage
            case .modelNotFound:
                .modelNotFound
            case .modelLoadFailed:
                .modelLoadFailed
            case .contextLimitExceeded:
                .contextLimitExceeded
            case .embeddingFailed:
                .embeddingFailed
            case .downloadFailed:
                .downloadFailed
            case .removalFailed:
                .removalFailed
            case .backendFailure:
                .backendFailure
            }
        return .init(
            code: coreCode,
            message: coreCode.safeMessage,
            model: model.map { ModelID($0) }
        )
    }
}

private extension SwamaError.Code {
    var safeMessage: String {
        switch self {
        case .invalidRequest:
            "The request is invalid."
        case .invalidImage:
            "An image could not be decoded."
        case .modelNotFound:
            "The requested model is not available locally."
        case .modelLoadFailed:
            "The requested model could not be loaded."
        case .contextLimitExceeded:
            "The request exceeds the configured context limit."
        case .embeddingFailed:
            "The embedding request failed."
        case .downloadFailed:
            "The model could not be downloaded."
        case .removalFailed:
            "The model could not be removed."
        case .backendFailure:
            "The local model backend failed."
        }
    }
}

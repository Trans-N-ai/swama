import CoreImage
import Foundation
import MLXLMCommon

// MARK: - RuntimeMessageRole

package enum RuntimeMessageRole: String, Sendable {
    case system
    case user
    case assistant
    case tool
}

// MARK: - RuntimeContentPart

package enum RuntimeContentPart: Sendable {
    case text(String)
    case imageURL(URL)
    case imageData(Data, mediaType: String)
}

// MARK: - RuntimeJSONValue

package enum RuntimeJSONValue: Hashable, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([RuntimeJSONValue])
    case object([String: RuntimeJSONValue])

    var sendableValue: any Sendable {
        switch self {
        case .null:
            NSNull()
        case let .bool(value):
            value
        case let .int(value):
            value
        case let .double(value):
            value
        case let .string(value):
            value
        case let .array(value):
            value.map(\.sendableValue)
        case let .object(value):
            value.mapValues(\.sendableValue)
        }
    }

    init(_ value: MLXLMCommon.JSONValue) {
        self =
            switch value {
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
                .array(value.map(Self.init))
            case let .object(value):
                .object(value.mapValues(Self.init))
            }
    }
}

// MARK: - RuntimeToolCall

package struct RuntimeToolCall: Hashable, Sendable {
    package init(id: String?, name: String, arguments: [String: RuntimeJSONValue]) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    package let id: String?
    package let name: String
    package let arguments: [String: RuntimeJSONValue]

    init(_ value: MLXLMCommon.ToolCall) {
        id = value.id
        name = value.function.name
        arguments = value.function.arguments.mapValues(RuntimeJSONValue.init)
    }

    var mlxValue: MLXLMCommon.ToolCall {
        .init(
            function: .init(
                name: name,
                arguments: arguments.mapValues { value in
                    MLXLMCommon.JSONValue.from(value.sendableValue)
                }
            ),
            id: id
        )
    }
}

// MARK: - RuntimeToolDefinition

package struct RuntimeToolDefinition: Hashable, Sendable {
    package init(name: String, description: String?, parameters: RuntimeJSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    package let name: String
    package let description: String?
    package let parameters: RuntimeJSONValue

    var mlxValue: [String: any Sendable] {
        var function: [String: any Sendable] = [
            "name": name,
            "parameters": parameters.sendableValue
        ]
        if let description {
            function["description"] = description
        }
        return [
            "type": "function",
            "function": function
        ]
    }
}

// MARK: - RuntimeMessage

package struct RuntimeMessage: Sendable {
    package init(
        role: RuntimeMessageRole,
        content: [RuntimeContentPart],
        toolCalls: [RuntimeToolCall],
        toolCallID: String?
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
    }

    package let role: RuntimeMessageRole
    package let content: [RuntimeContentPart]
    package let toolCalls: [RuntimeToolCall]
    package let toolCallID: String?
}

// MARK: - RuntimeGenerationOptions

package struct RuntimeGenerationOptions: Sendable {
    package init(
        maxTokens: Int?,
        temperature: Float,
        topP: Float,
        topK: Int,
        minP: Float,
        repetitionPenalty: Float?,
        repetitionContextSize: Int,
        presencePenalty: Float?,
        presenceContextSize: Int,
        frequencyPenalty: Float?,
        frequencyContextSize: Int,
        seed: UInt64?,
        contextLimit: Int?
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

    package let maxTokens: Int?
    package let temperature: Float
    package let topP: Float
    package let topK: Int
    package let minP: Float
    package let repetitionPenalty: Float?
    package let repetitionContextSize: Int
    package let presencePenalty: Float?
    package let presenceContextSize: Int
    package let frequencyPenalty: Float?
    package let frequencyContextSize: Int
    package let seed: UInt64?
    package let contextLimit: Int?
}

// MARK: - RuntimeGenerationRequest

package struct RuntimeGenerationRequest: Sendable {
    package init(
        model: String,
        messages: [RuntimeMessage],
        options: RuntimeGenerationOptions,
        tools: [RuntimeToolDefinition]
    ) {
        self.model = model
        self.messages = messages
        self.options = options
        self.tools = tools
    }

    package let model: String
    package let messages: [RuntimeMessage]
    package let options: RuntimeGenerationOptions
    package let tools: [RuntimeToolDefinition]
}

// MARK: - RuntimeGenerationEvent

package enum RuntimeGenerationEvent: Sendable {
    case textDelta(String)
    case toolCall(RuntimeToolCall)
}

// MARK: - RuntimeFinishReason

package enum RuntimeFinishReason: String, Sendable {
    case completed
    case length
    case toolCall
}

// MARK: - RuntimeUsage

package struct RuntimeUsage: Hashable, Sendable {
    package init(promptTokens: Int, completionTokens: Int) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
    }

    package let promptTokens: Int
    package let completionTokens: Int

    package var totalTokens: Int { promptTokens + completionTokens }
}

// MARK: - RuntimeGenerationMetrics

package struct RuntimeGenerationMetrics: Hashable, Sendable {
    package init(promptSeconds: Double, generationSeconds: Double, tokensPerSecond: Double) {
        self.promptSeconds = promptSeconds
        self.generationSeconds = generationSeconds
        self.tokensPerSecond = tokensPerSecond
    }

    package let promptSeconds: Double
    package let generationSeconds: Double
    package let tokensPerSecond: Double
}

// MARK: - RuntimeGenerationResult

package struct RuntimeGenerationResult: Sendable {
    package let output: String
    package let toolCalls: [RuntimeToolCall]
    package let usage: RuntimeUsage
    package let finishReason: RuntimeFinishReason
    package let metrics: RuntimeGenerationMetrics?
}

// MARK: - RuntimeEmbeddingRequest

package struct RuntimeEmbeddingRequest: Sendable {
    package init(model: String, inputs: [String]) {
        self.model = model
        self.inputs = inputs
    }

    package let model: String
    package let inputs: [String]
}

// MARK: - RuntimeEmbeddingResult

package struct RuntimeEmbeddingResult: Sendable {
    package let embeddings: [[Float]]
    package let usage: RuntimeUsage
}

// MARK: - RuntimeModelCapabilities

package struct RuntimeModelCapabilities: Hashable, Sendable {
    package init(textGeneration: Bool, vision: Bool, tools: Bool, embeddings: Bool) {
        self.textGeneration = textGeneration
        self.vision = vision
        self.tools = tools
        self.embeddings = embeddings
    }

    package let textGeneration: Bool
    package let vision: Bool
    package let tools: Bool
    package let embeddings: Bool
}

// MARK: - RuntimeModelInfo

package struct RuntimeModelInfo: Sendable {
    package let id: String
    package let created: Int
    package let sizeInBytes: Int64
    package let capabilities: RuntimeModelCapabilities
}

// MARK: - RuntimeCoreErrorCode

package enum RuntimeCoreErrorCode: String, Sendable {
    case invalidRequest
    case invalidImage
    case modelNotFound
    case modelLoadFailed
    case contextLimitExceeded
    case embeddingFailed
    case downloadFailed
    case removalFailed
    case backendFailure
}

// MARK: - RuntimeCoreError

package struct RuntimeCoreError: Error, Sendable {
    package let code: RuntimeCoreErrorCode
    package let model: String?
}

// MARK: - RuntimeCoreEngine

package actor RuntimeCoreEngine {
    package init(defaultContextLimit: Int? = nil) {
        pool = ModelPool()
        self.defaultContextLimit = defaultContextLimit
    }

    init(pool: ModelPool, defaultContextLimit: Int? = nil) {
        self.pool = pool
        self.defaultContextLimit = defaultContextLimit
    }

    package func generate(
        _ request: RuntimeGenerationRequest,
        onEvent: (@Sendable (RuntimeGenerationEvent) async throws -> Void)? = nil
    ) async throws -> RuntimeGenerationResult {
        try rejectUnsupportedAudioModel(request.model)
        do {
            let parameters = makeParameters(request.options)
            let contextLimit = request.options.contextLimit ?? defaultContextLimit
            let messages = request.messages
            let tools = request.tools
            let result = try await pool.run(modelName: request.model) { runner in
                let input = try self.makeUserInput(messages, tools: tools)
                return try await runner.runChat(
                    userInput: input,
                    parameters: parameters,
                    contextLimit: contextLimit,
                    onToken: { token in
                        try await onEvent?(.textDelta(token))
                    },
                    onToolCall: { toolCall in
                        try await onEvent?(.toolCall(.init(toolCall)))
                    }
                )
            }

            try Task.checkCancellation()
            if result.completionInfo?.stopReason == .cancelled {
                throw CancellationError()
            }

            let toolCalls = result.toolCalls.map(RuntimeToolCall.init)
            let completionTokens = result.completionInfo?.generationTokenCount ?? 0
            return RuntimeGenerationResult(
                output: result.output,
                toolCalls: toolCalls,
                usage: .init(
                    promptTokens: result.promptTokens,
                    completionTokens: completionTokens
                ),
                finishReason: finishReason(
                    result.completionInfo?.stopReason,
                    hasToolCalls: !toolCalls.isEmpty
                ),
                metrics: result.completionInfo.map {
                    .init(
                        promptSeconds: $0.promptTime,
                        generationSeconds: $0.generateTime,
                        tokensPerSecond: $0.tokensPerSecond
                    )
                }
            )
        }
        catch is CancellationError {
            throw CancellationError()
        }
        catch {
            throw mapError(error, model: request.model, fallback: .backendFailure)
        }
    }

    package func embed(_ request: RuntimeEmbeddingRequest) async throws -> RuntimeEmbeddingResult {
        try rejectUnsupportedAudioModel(request.model)
        do {
            let result = try await pool.runEmbeddingWithConcurrencyControl(modelName: request.model) { runner in
                try await runner.generateEmbeddings(inputs: request.inputs)
            }
            try Task.checkCancellation()
            return .init(
                embeddings: result.embeddings,
                usage: .init(promptTokens: result.usage.promptTokens, completionTokens: 0)
            )
        }
        catch is CancellationError {
            throw CancellationError()
        }
        catch {
            throw mapError(error, model: request.model, fallback: .embeddingFailed)
        }
    }

    package func models() -> [RuntimeModelInfo] {
        ModelManager.models()
            .filter { !Self.isUnsupportedAudioModelID($0.id) }
            .map { model in
                RuntimeModelInfo(
                    id: model.id,
                    created: model.created,
                    sizeInBytes: model.sizeInBytes,
                    capabilities: capabilities(for: model.id)
                )
            }
            .sorted { $0.id < $1.id }
    }

    package func fetch(_ model: String) async throws {
        try rejectUnsupportedAudioModel(model)
        do {
            _ = try await ModelDownloader.fetchModel(modelName: model)
        }
        catch {
            throw mapError(error, model: model, fallback: .downloadFailed)
        }
    }

    package func remove(_ model: String) async throws {
        try rejectUnsupportedAudioModel(model)
        await pool.remove(modelName: model)
        do {
            guard try ModelPaths.removeModel(model) else {
                throw RuntimeCoreError(code: .modelNotFound, model: model)
            }
        }
        catch let error as RuntimeCoreError {
            throw error
        }
        catch {
            throw RuntimeCoreError(code: .removalFailed, model: model)
        }
    }

    package func clearCache(for model: String) async {
        guard !Self.isUnsupportedAudioModelID(model) else {
            return
        }

        await pool.remove(modelName: model)
    }

    package func clearCache() async {
        await pool.clearCache()
    }

    private let pool: ModelPool
    private let defaultContextLimit: Int?

    static func isUnsupportedAudioModelID(_ model: String) -> Bool {
        let value = model.lowercased()
        let markers = [
            "whisper", "qwen3-asr", "glm-asr", "glmasr", "sensevoice", "voxtral",
            "parakeet", "moss-transcribe", "nemotron", "canary", "moonshine", "wav2vec",
            "mms-", "granite-speech", "tts", "orpheus", "marvis", "chatterbox", "vyvo",
            "fish-speech", "soprano", "kokoro", "cosyvoice", "omnivoice"
        ]
        return markers.contains(where: value.contains)
    }

    private func rejectUnsupportedAudioModel(_ model: String) throws {
        guard !Self.isUnsupportedAudioModelID(model) else {
            throw RuntimeCoreError(code: .invalidRequest, model: model)
        }
    }

    private nonisolated func makeUserInput(
        _ messages: [RuntimeMessage],
        tools: [RuntimeToolDefinition]
    ) throws -> MLXLMCommon.UserInput {
        let chat = try messages.map(makeChatMessage)
        return .init(
            chat: chat,
            tools: tools.isEmpty ? nil : tools.map(\.mlxValue)
        )
    }

    private nonisolated func makeChatMessage(_ message: RuntimeMessage) throws -> MLXLMCommon.Chat.Message {
        var textParts: [String] = []
        var images: [MLXLMCommon.UserInput.Image] = []
        for part in message.content {
            switch part {
            case let .text(text):
                textParts.append(text)
            case let .imageURL(url):
                images.append(.url(url))
            case let .imageData(data, _):
                guard let image = CIImage(data: data) else {
                    throw RuntimeCoreError(code: .invalidImage, model: nil)
                }

                images.append(.ciImage(image))
            }
        }
        let text = textParts.joined(separator: "\n")

        switch message.role {
        case .system:
            return .system(text, images: images)

        case .user:
            return .user(text, images: images)

        case .assistant:
            return .assistant(
                text,
                images: images,
                toolCalls: message.toolCalls.isEmpty ? nil : message.toolCalls.map(\.mlxValue)
            )

        case .tool:
            return .tool(text, id: message.toolCallID)
        }
    }

    private func makeParameters(_ options: RuntimeGenerationOptions) -> GenerateParameters {
        .init(
            maxTokens: options.maxTokens,
            temperature: options.temperature,
            topP: options.topP,
            topK: options.topK,
            minP: options.minP,
            repetitionPenalty: options.repetitionPenalty,
            repetitionContextSize: options.repetitionContextSize,
            presencePenalty: options.presencePenalty,
            presenceContextSize: options.presenceContextSize,
            frequencyPenalty: options.frequencyPenalty,
            frequencyContextSize: options.frequencyContextSize,
            seed: options.seed
        )
    }

    private func finishReason(
        _ stopReason: GenerateStopReason?,
        hasToolCalls: Bool
    ) -> RuntimeFinishReason {
        if hasToolCalls {
            return .toolCall
        }
        return stopReason == .length ? .length : .completed
    }

    private func capabilities(for model: String) -> RuntimeModelCapabilities {
        Self.capabilities(for: model, modelType: modelType(for: model))
    }

    static func capabilities(for model: String, modelType: String?) -> RuntimeModelCapabilities {
        let normalized = model.lowercased()
        let embeddingHint = ["embed", "bge", "e5-", "gte-"].contains(where: normalized.contains)
        let supportedEmbeddingTypes: Set<String> = [
            "bert", "roberta", "xlm-roberta", "distilbert", "nomic_bert", "qwen3",
            "lfm2", "gemma3", "gemma3_text", "gemma3n"
        ]
        let embedding = embeddingHint && modelType.map(supportedEmbeddingTypes.contains) == true
        let textGeneration = !embeddingHint
        let vision = textGeneration && ModelTypeDetector.isVLMModelName(model)
        return .init(
            textGeneration: textGeneration,
            vision: vision,
            tools: textGeneration,
            embeddings: embedding
        )
    }

    private func modelType(for model: String) -> String? {
        let url = ModelPaths.getModelDirectory(for: model).appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        return (object["model_type"] as? String)?.lowercased()
    }

    private func mapError(
        _ error: Error,
        model: String?,
        fallback: RuntimeCoreErrorCode
    ) -> RuntimeCoreError {
        if let error = error as? RuntimeCoreError {
            return error
        }
        if let error = error as? ModelPoolError {
            return switch error {
            case .modelNotFoundLocally:
                .init(code: .modelNotFound, model: model)
            case .failedToLoadModel:
                .init(code: .modelLoadFailed, model: model)
            }
        }
        if error is ContextLimitError {
            return .init(code: .contextLimitExceeded, model: model)
        }
        if error is EmbeddingError {
            return .init(code: .embeddingFailed, model: model)
        }
        return .init(code: fallback, model: model)
    }
}

import CoreImage
import Foundation
import MLXLMCommon
import SwamaCore
import SwamaKit

// MARK: - ServerModelPool

enum ServerModelPool {
    static let shared: ModelPool = .shared
}

// MARK: - ServerCoreEngine

enum ServerCoreEngine {
    static let backend: LegacyServerCoreBackend = .init(modelPool: ServerModelPool.shared)
    static let shared: SwamaEngine = .init(backend: backend)
    static var modelPoolIdentity: ObjectIdentifier { backend.modelPoolIdentity }
}

// MARK: - LegacyServerCoreBackend

/// Transitional package-only Core backend for the HTTP server.
///
/// The server still exposes legacy audio routes in v2, so non-audio Core requests must use the
/// exact same `ModelPool.shared` actor as STT/TTS. Constructing the default Runtime-backed engine
/// here would split the accepted global concurrency, live-operation, eviction, and cleanup state.
/// This adapter is removed with the v3 Runtime migration tracked by task #30.
struct LegacyServerCoreBackend: SwamaEngineBackend {
    init(modelPool: ModelPool) {
        self.modelPool = modelPool
    }

    var modelPoolIdentity: ObjectIdentifier {
        ObjectIdentifier(modelPool)
    }

    func generate(
        _ request: GenerationRequest,
        onEvent: (@Sendable (GenerationEvent) async throws -> Void)?
    ) async throws -> GenerationResponse {
        let modelName = ModelAliasResolver.resolve(name: request.model.rawValue)
        do {
            let result = try await modelPool.run(modelName: modelName) { runner in
                let input = try makeUserInput(
                    request.messages,
                    tools: request.tools,
                    modelName: modelName
                )
                return try await runner.runChatForCore(
                    userInput: input,
                    parameters: makeParameters(request.options),
                    contextLimit: request.options.contextLimit,
                    onToken: { try await onEvent?(.textDelta($0)) },
                    onToolCall: { try await onEvent?(.toolCall(.init($0))) }
                )
            }
            try Task.checkCancellation()
            if result.completionInfo?.stopReason == .cancelled {
                throw CancellationError()
            }

            let toolCalls = result.toolCalls.map(ToolCall.init)
            let completionTokens = result.completionInfo?.generationTokenCount ?? 0
            return GenerationResponse(
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

    func embed(_ request: EmbeddingRequest) async throws -> EmbeddingResponse {
        let modelName = ModelAliasResolver.resolve(name: request.model.rawValue)
        do {
            let result = try await modelPool.runEmbeddingWithConcurrencyControl(modelName: modelName) { runner in
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

    func models() async throws -> [SwamaCore.ModelInfo] {
        ModelManager.models()
            .filter { model in
                !ModelAliasResolver.isAudioModel(model.id) && !ModelAliasResolver.isTTSModel(model.id)
            }
            .map { model in
                SwamaCore.ModelInfo(
                    id: .init(model.id),
                    created: Date(timeIntervalSince1970: TimeInterval(model.created)),
                    sizeInBytes: model.sizeInBytes,
                    capabilities: capabilities(for: model.id)
                )
            }
            .sorted { $0.id.rawValue < $1.id.rawValue }
    }

    func fetch(_ model: ModelID) async throws -> ModelID {
        do {
            let resolved = try await ModelDownloader.fetchModel(modelName: model.rawValue)
            return .init(resolved)
        }
        catch {
            throw mapError(error, model: model, fallback: .downloadFailed)
        }
    }

    func remove(_ model: ModelID) async throws {
        let modelName = ModelAliasResolver.resolve(name: model.rawValue)
        await modelPool.remove(modelName: modelName)
        do {
            guard try ModelPaths.removeModel(modelName) else {
                throw SwamaError(code: .modelNotFound, message: "The model was not found.", model: model)
            }
        }
        catch let error as SwamaError {
            throw error
        }
        catch {
            throw SwamaError(code: .removalFailed, message: "The model could not be removed.", model: model)
        }
    }

    func clearCache(for model: ModelID) async {
        await modelPool.remove(modelName: ModelAliasResolver.resolve(name: model.rawValue))
    }

    func clearCache() async {
        await modelPool.clearCache()
    }

    private let modelPool: ModelPool

    private func makeUserInput(
        _ messages: [SwamaCore.Message],
        tools: [ToolDefinition],
        modelName: String
    ) throws -> MLXLMCommon.UserInput {
        let chat = try messages.map(makeChatMessage)
        let mlxTools = tools.isEmpty ? nil : tools.map(\.mlxValue)
        if messages.contains(where: \.hasMedia), shouldApplyQwen35MultimodalSafety(modelName: modelName) {
            return .init(
                chat: chat,
                processing: .init(resize: .init(width: 1344, height: 1344)),
                tools: mlxTools
            )
        }
        return .init(chat: chat, tools: mlxTools)
    }

    private func makeChatMessage(_ message: SwamaCore.Message) throws -> MLXLMCommon.Chat.Message {
        var textParts = [String]()
        var images = [MLXLMCommon.UserInput.Image]()
        for part in message.content {
            switch part {
            case let .text(text):
                textParts.append(text)
            case let .imageURL(url):
                images.append(.url(url))
            case let .imageData(data, _):
                guard let image = CIImage(data: data) else {
                    throw SwamaError(code: .invalidImage, message: "The image data is invalid.")
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

    func makeParameters(_ options: GenerationOptions) -> GenerateParameters {
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

    private func finishReason(_ reason: GenerateStopReason?, hasToolCalls: Bool) -> FinishReason {
        if hasToolCalls {
            return .toolCall
        }
        return reason == .length ? .length : .completed
    }

    private func mapError(
        _ error: Error,
        model: ModelID?,
        fallback: SwamaError.Code
    ) -> SwamaError {
        if let error = error as? SwamaError {
            return error
        }
        if let error = error as? ModelPoolError {
            return switch error {
            case .modelNotFoundLocally:
                .init(code: .modelNotFound, message: "The model is not available locally.", model: model)
            case .failedToLoadModel:
                .init(code: .modelLoadFailed, message: "The model could not be loaded.", model: model)
            }
        }
        if error is ContextLimitError {
            return .init(code: .contextLimitExceeded, message: error.localizedDescription, model: model)
        }
        if error is EmbeddingError {
            return .init(code: .embeddingFailed, message: "Embedding generation failed.", model: model)
        }
        return .init(code: fallback, message: "The local model backend failed.", model: model)
    }

    private func capabilities(for model: String) -> ModelCapabilities {
        let normalized = model.lowercased()
        let embedding = ["embed", "bge", "e5-", "gte-"].contains(where: normalized.contains)
        return .init(
            textGeneration: !embedding,
            vision: !embedding && ["vl", "vision", "gemma-3", "gemma3"].contains(where: normalized.contains),
            tools: !embedding,
            embeddings: embedding
        )
    }

    private func shouldApplyQwen35MultimodalSafety(modelName: String) -> Bool {
        let lowered = modelName.lowercased()
        return lowered.contains("qwen3.5") || lowered.contains("qwen3_5")
    }
}

private extension SwamaCore.Message {
    var hasMedia: Bool {
        content.contains { part in
            switch part {
            case .imageData,
                 .imageURL:
                true
            case .text:
                false
            }
        }
    }
}

private extension ToolDefinition {
    var mlxValue: [String: any Sendable] {
        var function: [String: any Sendable] = [
            "name": name,
            "parameters": parameters.sendableValue
        ]
        if let description {
            function["description"] = description
        }
        return ["type": "function", "function": function]
    }
}

private extension SwamaCore.ToolCall {
    init(_ value: MLXLMCommon.ToolCall) {
        self.init(
            id: value.id,
            name: value.function.name,
            arguments: value.function.arguments.mapValues { SwamaCore.JSONValue($0) }
        )
    }

    var mlxValue: MLXLMCommon.ToolCall {
        .init(
            function: .init(
                name: name,
                arguments: arguments.mapValues { MLXLMCommon.JSONValue.from($0.sendableValue) }
            ),
            id: id
        )
    }
}

private extension SwamaCore.JSONValue {
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
}

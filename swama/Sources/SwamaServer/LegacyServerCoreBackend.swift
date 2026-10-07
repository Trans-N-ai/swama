import CoreImage
import Foundation
import ImageIO
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

    func decide(_ request: DecisionRequest) async throws -> DecisionResponse {
        let modelName = ModelAliasResolver.resolve(name: request.model.rawValue)
        let contextLimit = await ContextLimitConfig.shared.currentLimit()
        var answers = [String: DecisionAnswer]()
        var promptTokens = 0
        do {
            for question in request.questions {
                try Task.checkCancellation()
                let labels: [String] =
                    if case let .choice(_, _, options) = question, options.count > 26 {
                        try await decisionPairLabelPrefix(
                            count: options.count, usable: pairLabels(modelName: modelName)
                        )
                    }
                    else {
                        question.decisionLabels
                    }
                let prompt = question.decisionPrompt(input: request.input, labels: labels)
                // Tokenize before taking the exclusive model slot when the model is already loaded,
                // so this CPU work overlaps other requests' model work. Cold loads keep the old path.
                // Prepared prompts are text-only; image decisions are prepared inside the slot.
                var prepared: PreparedDecisionPrompt?
                if request.images.isEmpty, let container = await modelPool.loadedContainer(modelName: modelName) {
                    prepared = try await prepareDecisionPrompt(
                        container: container, content: prompt, contextLimit: contextLimit
                    )
                }
                let images = request.images.map(\.data)
                let processing = decisionImageProcessing(request, modelName: modelName)
                let scored = try await modelPool.run(modelName: modelName) { [prepared] runner in
                    try await runner.scoreDecision(
                        content: prompt, labels: labels, contextLimit: contextLimit, prepared: prepared,
                        images: images, imageProcessing: processing
                    )
                }
                answers[question.id] = try question.decisionAnswer(
                    logProbs: scored.labelLogProbs,
                    temperature: request.temperature,
                    promptTokenIDs: request.returnPromptTokenIDs ? scored.promptTokenIDs : nil,
                    labelTokenIDs: request.returnPromptTokenIDs ? scored.labelTokenIDs : nil
                )
                promptTokens += scored.promptTokenIDs.count
            }
            try Task.checkCancellation()
            return .init(
                model: request.model,
                answers: answers,
                usage: .init(promptTokens: promptTokens, completionTokens: 0)
            )
        }
        catch is CancellationError {
            throw CancellationError()
        }
        catch let error as DecisionScoringError {
            throw SwamaError(
                code: Self.errorCode(for: error), message: error.localizedDescription, model: request.model
            )
        }
        catch {
            throw mapError(error, model: request.model, fallback: .backendFailure)
        }
    }

    /// The core error code a decision scoring failure is reported as.
    static func errorCode(for error: DecisionScoringError) -> SwamaError.Code {
        switch error {
        case .contextLimitExceeded: .contextLimitExceeded
        case .invalidLogits: .backendFailure
        case .invalidImage,
             .unprocessableImage: .invalidImage
        default: .invalidRequest
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

    /// The model's two-letter labels, from the loaded container when there is one. A cold model is
    /// loaded in the slot, as the scoring call that follows would do.
    private func pairLabels(modelName: String) async throws -> [String]? {
        if let container = await modelPool.loadedContainer(modelName: modelName) {
            return try await cachedDecisionPairLabels(container: container)
        }
        return try await modelPool.run(modelName: modelName) { runner in
            try await runner.decisionPairLabels()
        }
    }

    /// Internal for tests: task #81 compares decision image prompts with the real chat path.
    func makeUserInput(
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

    /// The image resize policy for a decision: the request's own maximum when given, otherwise the
    /// chat path's policy for the same model as the starting bound (the cap below then lowers it for
    /// images smaller than that bound, which chat does not do).
    /// The bound is capped at the longest side of the largest image, so images are not enlarged beyond
    /// what the model needs: enlarging adds tokens and latency but no information. Two exceptions: a
    /// tiny image is still enlarged until its short side reaches `minimumShortSide`, and since resize is
    /// per request, with several images of different sizes the smaller ones may be enlarged up to the
    /// largest one's size.
    private func decisionImageProcessing(_ request: DecisionRequest, modelName: String) -> MLXLMCommon.UserInput
        .Processing
    {
        let bound = Self.decisionResizeBound(
            requested: request.imageMaxDimension,
            appliesQwenDefault: !request.images.isEmpty && shouldApplyQwen35MultimodalSafety(modelName: modelName),
            imageSizes: request.images.map { Self.pixelSize(of: $0.data) }
        )
        guard let bound else {
            return .init()
        }

        return .init(resize: .init(width: bound, height: bound))
    }

    /// The square resize bound for a decision, or `nil` for no resize. Never exceeds the longest side
    /// of the largest image, except that every image's short side must still reach
    /// `minimumShortSide` after fitting (vision processors refuse sides below their patch factor, 32
    /// for Qwen3.5). An image whose size is unknown (`nil`) leaves the bound uncapped.
    static func decisionResizeBound(requested: Int?, appliesQwenDefault: Bool, imageSizes: [CGSize?]) -> Int? {
        let bound: Int
        if let requested {
            bound = requested
        }
        else if appliesQwenDefault {
            bound = 1344
        }
        else {
            return nil
        }

        var needed = 0
        for size in imageSizes {
            guard let size, size.width > 0, size.height > 0 else {
                return bound
            }

            let longest = Double(max(size.width, size.height))
            let shortest = Double(min(size.width, size.height))
            let keepsShortSide = Int((Double(minimumShortSide) * longest / shortest).rounded(.up))
            needed = max(needed, Int(longest), keepsShortSide)
        }
        guard needed > 0 else {
            return bound
        }

        return min(bound, needed)
    }

    /// Smallest short side an image may have after resizing, with margin over the Qwen3.5 patch factor.
    static let minimumShortSide = 64

    /// Pixel size from the image header, without decoding. Orientation does not matter because callers
    /// only use the longer side.
    private static func pixelSize(of data: Data) -> CGSize? {
        guard
            let source = CGImageSourceCreateWithData(data as CFData, nil),
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? Int,
            let height = properties[kCGImagePropertyPixelHeight] as? Int
        else {
            return nil
        }

        return CGSize(width: width, height: height)
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

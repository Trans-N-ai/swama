import Foundation
import MLX
@testable import MLXEmbedders
import MLXLLM
@preconcurrency import MLXLMCommon
import MLXNN
@testable import SwamaRuntime
import Testing

// MARK: - RuntimeCoreEngineTests

@Suite("Runtime Core bridge", .serialized)
struct RuntimeCoreEngineTests {
    @Test func generationMapsTextUsageFinishAndAwaitedEvents() async throws {
        let engine = RuntimeCoreEngine(pool: makePool())
        let events = RuntimeEventCollector()
        let result = try await engine.generate(request(maxTokens: 4)) { event in
            await events.append(event)
        }

        #expect(!result.output.isEmpty)
        #expect(result.usage.promptTokens > 0)
        #expect(result.usage.completionTokens == 4)
        #expect(result.finishReason == .length)
        #expect(await events.text == result.output)
    }

    @Test func callbackCancellationThrowsInsteadOfReturningPartialResponse() async throws {
        let engine = RuntimeCoreEngine(pool: makePool())
        do {
            _ = try await engine.generate(request(maxTokens: 20)) { event in
                if case .textDelta = event {
                    throw CancellationError()
                }
            }
            Issue.record("cancelled runtime generation returned a response")
        }
        catch {
            #expect(error is CancellationError)
        }
    }

    @Test func invalidImageDataUsesABoundedRuntimeError() async throws {
        let engine = RuntimeCoreEngine(pool: makePool())
        let invalid = RuntimeGenerationRequest(
            model: "test/model",
            messages: [.init(
                role: .user,
                content: [.imageData(Data([0x00]), mediaType: "image/png")],
                toolCalls: [],
                toolCallID: nil
            )],
            options: options(maxTokens: 1),
            tools: []
        )

        do {
            _ = try await engine.generate(invalid)
            Issue.record("invalid image unexpectedly reached generation")
        }
        catch let error as RuntimeCoreError {
            #expect(error.code == .invalidImage)
            #expect(error.model == nil)
        }
    }

    @Test func requestScopedContextLimitIsEnforced() async throws {
        let engine = RuntimeCoreEngine(pool: makePool())
        var limited = request(maxTokens: 1)
        limited = .init(
            model: limited.model,
            messages: limited.messages,
            options: options(maxTokens: 1, contextLimit: 1),
            tools: limited.tools
        )

        do {
            _ = try await engine.generate(limited)
            Issue.record("request-scoped context limit was ignored")
        }
        catch let error as RuntimeCoreError {
            #expect(error.code == .contextLimitExceeded)
        }
    }

    @Test func embeddingMapsVectorsAndUsage() async throws {
        let tokenizer = WordVocabTokenizer(contentVocabSize: 16)
        let container = EmbedderModelContainer(context: .init(
            configuration: .init(id: "runtime-core-embedding"),
            model: FixedEmbeddingModel(),
            tokenizer: tokenizer,
            pooling: Pooling(strategy: .mean)
        ))
        let runner = EmbeddingRunner(container: container)
        let pool = ModelPool(
            memoryHooks: .init(activeMemory: { 0 }, clearCache: {}),
            loadOverrides: .init(
                modelExistsLocally: { _ in true },
                determineIsVLM: { _ in false },
                loadLanguage: { _, _ in fatalError("unexpected language load") },
                loadEmbedding: { _ in runner }
            )
        )
        let result = try await RuntimeCoreEngine(pool: pool).embed(.init(
            model: "test/embedding",
            inputs: ["w1 w2", "w3"]
        ))

        #expect(result.embeddings.count == 2)
        #expect(result.embeddings.allSatisfy { $0.count == 2 })
        #expect(result.usage.promptTokens == 3)
        #expect(result.usage.completionTokens == 0)
    }

    @Test func publicModelCatalogRejectsLegacyAudioIdentifiers() {
        for model in [
            "whisper-tiny",
            "funasr",
            "beshkenadze/cohere-transcribe-03-2026-mlx-fp16",
            "FireRedTeam/FireRedASR-AED-L-MLX",
            "mlx-community/LASR-CTC-Large",
            "mlx-community/Qwen3-ASR-0.6B-4bit",
            "mlx-community/Qwen3-TTS-0.6B",
            "mlx-community/Kokoro-82M"
        ] {
            #expect(RuntimeCoreEngine.isUnsupportedAudioModelID(model))
        }
        for model in [
            "mlx-community/Qwen3-1.7B-4bit",
            "mlx-community/embeddinggemma-300m-4bit",
            "mlx-community/gemma-3-4b-it-4bit",
            "acme/watts-language-model"
        ] {
            #expect(!RuntimeCoreEngine.isUnsupportedAudioModelID(model))
        }
    }

    @Test func modelIdentifiersAndFilesystemOperationsRejectTraversal() async throws {
        let fileManager = FileManager.default
        let temporaryRoot = fileManager.temporaryDirectory
            .appendingPathComponent("swama-core-path-test-\(UUID().uuidString)")
        let modelsRoot = temporaryRoot.appendingPathComponent("models")
        let outsideRoot = temporaryRoot.appendingPathComponent("outside-model-root")
        try fileManager.createDirectory(at: modelsRoot, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: outsideRoot, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: outsideRoot.appendingPathComponent(".swama-meta.json"))

        let priorModelsRoot = ProcessInfo.processInfo.environment["SWAMA_MODELS"]
        setenv("SWAMA_MODELS", modelsRoot.path, 1)
        defer {
            if let priorModelsRoot {
                setenv("SWAMA_MODELS", priorModelsRoot, 1)
            }
            else {
                unsetenv("SWAMA_MODELS")
            }
            try? fileManager.removeItem(at: temporaryRoot)
        }

        let invalid = "../outside-model-root"
        #expect(!RuntimeCoreEngine.isValidModelID(invalid))
        do {
            try await RuntimeCoreEngine(pool: makePool()).remove(invalid)
            Issue.record("runtime accepted a traversing model ID")
        }
        catch let error as RuntimeCoreError {
            #expect(error.code == .invalidRequest)
        }
        #expect(fileManager.fileExists(atPath: outsideRoot.path))

        let containedModel = modelsRoot.appendingPathComponent("org/model")
        try fileManager.createDirectory(at: containedModel, withIntermediateDirectories: true)
        let forgedMetadata = try JSONSerialization.data(withJSONObject: ["path": outsideRoot.path])
        try forgedMetadata.write(to: containedModel.appendingPathComponent(".swama-meta.json"))
        #expect(ModelPaths.getModelDirectory(for: "org/model").standardizedFileURL == containedModel
            .standardizedFileURL
        )

        do {
            _ = try ModelPaths.removeModel(invalid)
            Issue.record("filesystem boundary accepted a traversing model ID")
        }
        catch {
            #expect(error is ModelPathError)
        }
        #expect(fileManager.fileExists(atPath: outsideRoot.path))

        do {
            _ = try ModelPaths.containedURL(in: modelsRoot, relativePath: "../../escaped.json")
            Issue.record("download destination escaped its model directory")
        }
        catch {
            #expect(error is ModelPathError)
        }

        let symlink = modelsRoot.appendingPathComponent("linked-outside")
        try fileManager.createSymbolicLink(at: symlink, withDestinationURL: outsideRoot)
        do {
            _ = try ModelPaths.containedURL(in: modelsRoot, relativePath: "linked-outside/file.json")
            Issue.record("symlinked download destination escaped its model directory")
        }
        catch {
            #expect(error is ModelPathError)
        }
    }

    @Test func embeddingCapabilityRequiresASupportedLocalModelType() {
        let supported = RuntimeCoreEngine.capabilities(
            for: "mlx-community/embeddinggemma-300m-4bit",
            modelType: "gemma3"
        )
        #expect(supported.embeddings)
        #expect(!supported.textGeneration)
        #expect(!supported.vision)

        let unsupported = RuntimeCoreEngine.capabilities(
            for: "mlx-community/nomicai-modernbert-embed-base-4bit",
            modelType: "modernbert"
        )
        #expect(!unsupported.embeddings)
        #expect(!unsupported.textGeneration)
        #expect(!unsupported.vision)
        #expect(!unsupported.tools)
    }

    @Test func audioIdentifiersCannotEnterTheCoreRuntime() async throws {
        let engine = RuntimeCoreEngine(pool: makePool())
        do {
            _ = try await engine.generate(.init(
                model: "whisper-tiny",
                messages: [.init(
                    role: .user,
                    content: [.text("transcribe")],
                    toolCalls: [],
                    toolCallID: nil
                )],
                options: options(maxTokens: 1),
                tools: []
            ))
            Issue.record("audio identifier entered generation")
        }
        catch let error as RuntimeCoreError {
            #expect(error.code == .invalidRequest)
            #expect(error.model == "whisper-tiny")
        }
    }

    @Test func toolCallJSONRoundTripsAcrossTheRuntimeBoundary() {
        let call = RuntimeToolCall(
            id: "call-1",
            name: "lookup",
            arguments: [
                "enabled": .bool(true),
                "count": .int(2),
                "nested": .object(["value": .string("x")])
            ]
        )
        #expect(RuntimeToolCall(call.mlxValue) == call)
    }

    @Test func toolSchemasReachBypassCacheMissAndCacheHitGeneration() async throws {
        let tool = RuntimeToolDefinition(
            name: "lookup",
            description: nil,
            parameters: .object([
                "type": .string("object"),
                "properties": .object([
                    "enabled": .object(["type": .string("boolean")]),
                    "count": .object(["type": .string("integer")])
                ])
            ])
        )
        let tools = [tool.mlxValue]
        let parameters = GenerateParameters(maxTokens: 1, temperature: 0)

        let bypassRunner = ModelRunner(
            container: makeToolSchemaContainer(),
            promptCacheStore: PromptCacheStore()
        )
        let bypass = try await bypassRunner.runChat(
            userInput: .init(
                chat: [.user("first", images: [.url(#require(URL(string: "https://example.invalid/image.png")))])],
                tools: tools
            ),
            parameters: parameters
        )
        assertTypedToolCall(bypass.toolCalls)

        let cacheRunner = ModelRunner(
            container: makeToolSchemaContainer(),
            promptCacheStore: PromptCacheStore()
        )
        let miss = try await cacheRunner.runChat(
            userInput: .init(chat: [.user("first")], tools: tools),
            parameters: parameters
        )
        assertTypedToolCall(miss.toolCalls)

        let hit = try await cacheRunner.runChat(
            userInput: .init(chat: [.user("first"), .user("second")], tools: tools),
            parameters: parameters
        )
        assertTypedToolCall(hit.toolCalls)
    }

    private func request(maxTokens: Int) -> RuntimeGenerationRequest {
        .init(
            model: "test/model",
            messages: [.init(
                role: .user,
                content: [.text("w1 w2")],
                toolCalls: [],
                toolCallID: nil
            )],
            options: options(maxTokens: maxTokens),
            tools: []
        )
    }

    private func options(maxTokens: Int, contextLimit: Int = 4096) -> RuntimeGenerationOptions {
        .init(
            maxTokens: maxTokens,
            temperature: 0,
            topP: 1,
            topK: 0,
            minP: 0,
            repetitionPenalty: nil,
            repetitionContextSize: 20,
            presencePenalty: nil,
            presenceContextSize: 20,
            frequencyPenalty: nil,
            frequencyContextSize: 20,
            seed: 1,
            contextLimit: contextLimit
        )
    }

    private func makePool() -> ModelPool {
        let container = makeTestContainer().container
        return ModelPool(
            memoryHooks: .init(activeMemory: { 0 }, clearCache: {}),
            loadOverrides: .init(
                modelExistsLocally: { _ in true },
                determineIsVLM: { _ in false },
                loadLanguage: { _, _ in container },
                loadEmbedding: { _ in fatalError("unexpected embedding load") }
            )
        )
    }
}

private func assertTypedToolCall(_ calls: [MLXLMCommon.ToolCall]) {
    #expect(calls.count == 1)
    guard let call = calls.first else {
        return
    }

    let runtimeCall = RuntimeToolCall(call)
    #expect(runtimeCall.name == "lookup")
    #expect(runtimeCall.arguments["enabled"] == .bool(true))
    #expect(runtimeCall.arguments["count"] == .int(2))
}

private func makeToolSchemaContainer() -> MLXLMCommon.ModelContainer {
    let tokenizer = ToolScriptTokenizer()
    let configuration = MLXLMCommon.ModelConfiguration(
        id: "tool-schema-test",
        toolCallFormat: .lfm2
    )
    let context = MLXLMCommon.ModelContext(
        configuration: configuration,
        model: ToolScriptModel(),
        processor: ToolScriptProcessor(),
        tokenizer: tokenizer
    )
    return MLXLMCommon.ModelContainer(context: context)
}

// MARK: - ToolScriptTokenizer

private struct ToolScriptTokenizer: MLXLMCommon.Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text _: String, addSpecialTokens _: Bool) -> [Int] { [0] }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIds.map { token in
            token == 1
                ? "<|tool_call_start|>[lookup(enabled='true', count='2')]<|tool_call_end|>"
                : ""
        }
        .joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        token.isEmpty ? 0 : 1
    }

    func convertIdToToken(_ id: Int) -> String? {
        decode(tokenIds: [id], skipSpecialTokens: false)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        Array(repeating: 0, count: max(messages.count, 1))
    }
}

// MARK: - ToolScriptProcessor

private struct ToolScriptProcessor: UserInputProcessor {
    private let generator = MLXLMCommon.DefaultMessageGenerator()

    func prepare(input: MLXLMCommon.UserInput) throws -> LMInput {
        let messages = generator.generate(from: input)
        return LMInput(tokens: MLXArray(Array(repeating: Int32(0), count: max(messages.count, 1))))
    }
}

// MARK: - ToolScriptModel

private final class ToolScriptModel: MLXNN.Module, LLMModel, KVCacheDimensionProvider {
    let kvHeads = [1]
    var loraLayers: [MLXNN.Module] { [] }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let sequenceLength = inputs.dim(1)
        let values = inputs.asType(.float32).reshaped([1, 1, sequenceLength, 1])
        if let cache {
            for layer in cache {
                _ = layer.update(keys: values, values: values)
            }
        }

        let logits: [Float] = [0, 10, 0]
        return MLXArray(Array(repeating: logits, count: sequenceLength).flatMap(\.self), [1, sequenceLength, 3])
    }
}

// MARK: - RuntimeEventCollector

private actor RuntimeEventCollector {
    func append(_ event: RuntimeGenerationEvent) {
        events.append(event)
    }

    var text: String {
        events.compactMap { event in
            if case let .textDelta(value) = event {
                value
            }
            else {
                nil
            }
        }
        .joined()
    }

    private var events: [RuntimeGenerationEvent] = []
}

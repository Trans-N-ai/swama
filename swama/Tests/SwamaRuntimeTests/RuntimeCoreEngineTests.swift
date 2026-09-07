import Foundation
@testable import MLXEmbedders
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
            "mlx-community/Qwen3-ASR-0.6B-4bit",
            "mlx-community/Qwen3-TTS-0.6B",
            "mlx-community/Kokoro-82M"
        ] {
            #expect(RuntimeCoreEngine.isUnsupportedAudioModelID(model))
        }
        for model in [
            "mlx-community/Qwen3-1.7B-4bit",
            "mlx-community/embeddinggemma-300m-4bit",
            "mlx-community/gemma-3-4b-it-4bit"
        ] {
            #expect(!RuntimeCoreEngine.isUnsupportedAudioModelID(model))
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

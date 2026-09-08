import Foundation
@testable import SwamaCore
import Testing

// MARK: - SwamaCoreTests

@Suite("SwamaCore public contract")
struct SwamaCoreTests {
    @Test func publicValuesHaveStableCodableRoundTrips() throws {
        let modelID = ModelID("org/model")
        let encodedModelID = try JSONEncoder().encode(modelID)
        #expect(try JSONDecoder().decode(String.self, from: encodedModelID) == "org/model")
        #expect(try JSONDecoder().decode(ModelID.self, from: Data("\"org/model\"".utf8)) == modelID)

        let request = try GenerationRequest(
            model: .init("org/model"),
            messages: [
                .init(role: .system, text: "Be concise."),
                .init(role: .user, content: [
                    .text("Describe this."),
                    .imageURL(#require(URL(string: "https://example.invalid/image.png"))),
                    .imageData(Data([0x01, 0x02]), mediaType: "image/png")
                ])
            ],
            options: .init(maxTokens: 32, temperature: 0, seed: 7, contextLimit: 4096),
            tools: [.init(
                name: "lookup",
                description: "Look up a value",
                parameters: .object(["key": .object(["type": .string("string")])])
            )]
        )

        let data = try JSONEncoder().encode(request)
        #expect(try JSONDecoder().decode(GenerationRequest.self, from: data) == request)

        let response = GenerationResponse(
            output: "done",
            toolCalls: [.init(id: "call-1", name: "lookup", arguments: ["key": .string("x")])],
            usage: .init(promptTokens: 4, completionTokens: 2),
            finishReason: .toolCall,
            metrics: .init(promptSeconds: 0.1, generationSeconds: 0.2, tokensPerSecond: 10)
        )
        #expect(try JSONDecoder().decode(
            GenerationResponse.self,
            from: JSONEncoder().encode(response)
        ) == response)
        #expect(try JSONDecoder().decode(
            FinishReason.self,
            from: Data("\"future_backend_reason\"".utf8)
        ) == .unknown)
        #expect(try String(decoding: JSONEncoder().encode(FinishReason.unknown), as: UTF8.self) == "\"unknown\"")
    }

    @Test func engineForwardsAwaitedEventsAndResult() async throws {
        let response = GenerationResponse(
            output: "hello",
            toolCalls: [],
            usage: .init(promptTokens: 2, completionTokens: 1),
            finishReason: .completed
        )
        let backend = StubBackend(generationResponse: response)
        let engine = SwamaEngine(backend: backend)
        let events = EventCollector()

        let result = try await engine.generate(.init(
            model: .init("org/model"),
            messages: [.init(role: .user, text: "hi")]
        )) { event in
            await events.append(event)
        }

        #expect(result == response)
        #expect(await events.values == [.textDelta("hello")])
    }

    @Test func cancellationNeverReturnsAPartialResponse() async throws {
        let backend = StubBackend(
            generationResponse: .init(
                output: "partial",
                toolCalls: [],
                usage: .init(promptTokens: 1, completionTokens: 1),
                finishReason: .completed
            ),
            cancelGeneration: true
        )
        let engine = SwamaEngine(backend: backend)

        do {
            _ = try await engine.generate(.init(
                model: .init("org/model"),
                messages: [.init(role: .user, text: "hi")]
            ))
            Issue.record("cancelled generation returned a response")
        }
        catch {
            #expect(error is CancellationError)
        }
    }

    @Test func invalidRequestsFailBeforeBackendExecution() async throws {
        let backend = StubBackend(generationResponse: .init(
            output: "unexpected",
            toolCalls: [],
            usage: .init(promptTokens: 0, completionTokens: 0),
            finishReason: .completed
        ))
        let engine = SwamaEngine(backend: backend)

        do {
            _ = try await engine.generate(.init(model: .init(""), messages: []))
            Issue.record("invalid request unexpectedly succeeded")
        }
        catch let error as SwamaError {
            #expect(error.code == .invalidRequest)
        }
        #expect(await backend.generationCalls == 0)
    }

    @Test func invalidModelIdentifiersNeverReachLifecycleBackends() async throws {
        let backend = StubBackend(generationResponse: emptyResponse)
        let engine = SwamaEngine(backend: backend)
        let invalidIDs = [
            "../outside", "/absolute", "org//model", "org/./model", "org/../model",
            "org\\model", "org/model/extra", " org/model", "org/model\n"
        ]

        for rawValue in invalidIDs {
            let model = ModelID(rawValue)
            do {
                _ = try await engine.generate(.init(
                    model: model,
                    messages: [.init(role: .user, text: "hi")]
                ))
                Issue.record("invalid model ID reached generation: \(rawValue)")
            }
            catch let error as SwamaError {
                #expect(error.code == .invalidRequest)
            }

            do {
                _ = try await engine.embed(.init(model: model, inputs: ["hi"]))
                Issue.record("invalid model ID reached embedding: \(rawValue)")
            }
            catch let error as SwamaError {
                #expect(error.code == .invalidRequest)
            }

            do {
                try await engine.fetch(model)
                Issue.record("invalid model ID reached fetch: \(rawValue)")
            }
            catch let error as SwamaError {
                #expect(error.code == .invalidRequest)
            }

            do {
                try await engine.remove(model)
                Issue.record("invalid model ID reached removal: \(rawValue)")
            }
            catch let error as SwamaError {
                #expect(error.code == .invalidRequest)
            }

            await engine.clearCache(for: model)
        }

        #expect(await backend.generationCalls == 0)
        #expect(await backend.embeddingCalls == 0)
        #expect(await backend.fetchCalls == 0)
        #expect(await backend.removeCalls == 0)
        #expect(await backend.modelCacheClearCalls == 0)
    }

    @Test func invalidRoleFieldsAndSamplingValuesFailBeforeBackendExecution() async throws {
        let backend = StubBackend(generationResponse: emptyResponse)
        let engine = SwamaEngine(backend: backend)
        let model = ModelID("org/model")
        let toolCall = ToolCall(name: "lookup", arguments: [:])
        let invalidMessages: [Message] = [
            .init(role: .user, content: [.text("hi")], toolCalls: [toolCall]),
            .init(role: .system, content: [.text("hi")], toolCallID: "call-1"),
            .init(role: .assistant, content: [.text("hi")], toolCallID: "call-1"),
            .init(role: .tool, content: [.text("result")]),
            .init(role: .tool, content: [.text("result")], toolCalls: [toolCall], toolCallID: "call-1")
        ]
        let invalidOptions: [GenerationOptions] = [
            .init(temperature: .infinity),
            .init(temperature: .nan),
            .init(topP: .infinity),
            .init(minP: .nan),
            .init(repetitionPenalty: -0.1),
            .init(repetitionPenalty: .infinity),
            .init(presencePenalty: 2.1),
            .init(presencePenalty: .nan),
            .init(frequencyPenalty: -2.1),
            .init(frequencyPenalty: .infinity)
        ]

        for message in invalidMessages {
            do {
                _ = try await engine.generate(.init(model: model, messages: [message]))
                Issue.record("role-inapplicable message fields reached the backend")
            }
            catch let error as SwamaError {
                #expect(error.code == .invalidRequest)
            }
        }
        for options in invalidOptions {
            do {
                _ = try await engine.generate(.init(
                    model: model,
                    messages: [.init(role: .user, text: "hi")],
                    options: options
                ))
                Issue.record("invalid sampling value reached the backend")
            }
            catch let error as SwamaError {
                #expect(error.code == .invalidRequest)
            }
        }

        #expect(await backend.generationCalls == 0)
    }

    private var emptyResponse: GenerationResponse {
        .init(
            output: "",
            toolCalls: [],
            usage: .init(promptTokens: 0, completionTokens: 0),
            finishReason: .completed
        )
    }
}

// MARK: - EventCollector

private actor EventCollector {
    func append(_ event: GenerationEvent) {
        values.append(event)
    }

    private(set) var values: [GenerationEvent] = []
}

// MARK: - StubBackend

private actor StubBackend: SwamaEngineBackend {
    init(generationResponse: GenerationResponse, cancelGeneration: Bool = false) {
        self.generationResponse = generationResponse
        self.cancelGeneration = cancelGeneration
    }

    func generate(
        _: GenerationRequest,
        onEvent: (@Sendable (GenerationEvent) async throws -> Void)?
    ) async throws -> GenerationResponse {
        generationCalls += 1
        try await onEvent?(.textDelta(generationResponse.output))
        if cancelGeneration {
            throw CancellationError()
        }
        return generationResponse
    }

    func embed(_: EmbeddingRequest) async throws -> EmbeddingResponse {
        embeddingCalls += 1
        return .init(embeddings: [[1]], usage: .init(promptTokens: 1, completionTokens: 0))
    }

    func models() async throws -> [ModelInfo] { [] }
    func fetch(_: ModelID) async throws { fetchCalls += 1 }
    func remove(_: ModelID) async throws { removeCalls += 1 }
    func clearCache(for _: ModelID) async { modelCacheClearCalls += 1 }
    func clearCache() async {}

    private let generationResponse: GenerationResponse
    private let cancelGeneration: Bool
    private(set) var generationCalls = 0
    private(set) var embeddingCalls = 0
    private(set) var fetchCalls = 0
    private(set) var removeCalls = 0
    private(set) var modelCacheClearCalls = 0
}

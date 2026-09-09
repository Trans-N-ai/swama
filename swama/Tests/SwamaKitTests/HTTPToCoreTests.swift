import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
import SwamaCore
@testable import SwamaKit
@testable import SwamaServer
import Testing

// MARK: - HTTPToCoreTests

@MainActor @Suite("HTTP to Core adapters", .serialized)
struct HTTPToCoreTests {
    @Test func serverCoreAndAudioRoutesShareTheLegacyPool() {
        let legacyIdentity = ObjectIdentifier(ModelPool.shared)
        #expect(ObjectIdentifier(ServerModelPool.shared) == legacyIdentity)
        #expect(ServerCoreEngine.modelPoolIdentity == legacyIdentity)
    }

    @Test func legacyBackendMapsEveryCoreSamplingOption() {
        let backend = LegacyServerCoreBackend(modelPool: ServerModelPool.shared)
        let parameters = backend.makeParameters(.init(
            maxTokens: 17,
            temperature: 0.2,
            topP: 0.8,
            topK: 7,
            minP: 0.1,
            repetitionPenalty: 1.05,
            repetitionContextSize: 64,
            presencePenalty: 0.4,
            presenceContextSize: 30,
            frequencyPenalty: 0.3,
            frequencyContextSize: 40,
            seed: 42,
            contextLimit: 4096
        ))

        #expect(parameters.maxTokens == 17)
        #expect(parameters.temperature == 0.2)
        #expect(parameters.topP == 0.8)
        #expect(parameters.topK == 7)
        #expect(parameters.minP == 0.1)
        #expect(parameters.repetitionPenalty == 1.05)
        #expect(parameters.repetitionContextSize == 64)
        #expect(parameters.presencePenalty == 0.4)
        #expect(parameters.presenceContextSize == 30)
        #expect(parameters.frequencyPenalty == 0.3)
        #expect(parameters.frequencyContextSize == 40)
        #expect(parameters.seed == 42)
    }

    @Test func completionWireFieldsMapToCoreValues() throws {
        let data = Data(#"""
        {
          "model":"alias",
          "messages":[
            {"role":"system","content":"system"},
            {"role":"user","content":[
              {"type":"text","text":"look"},
              {"type":"image_url","image_url":{"url":"https://example.invalid/image.png"}}
            ]},
            {"role":"assistant","content":"calling","tool_calls":[{
              "id":"call-1","type":"function","function":{"name":"lookup","arguments":"{\"key\":\"x\"}"}
            }]},
            {"role":"tool","content":"result","tool_call_id":"call-1"}
          ],
          "temperature":0.2,
          "top_p":0.8,
          "top_k":7,
          "min_p":0.1,
          "repetition_penalty":1.05,
          "repetition_context_size":64,
          "presence_penalty":0.4,
          "frequency_penalty":0.3,
          "context_limit":4096,
          "max_tokens":17,
          "tools":[{"type":"function","function":{
            "name":"lookup","description":"Lookup","parameters":{"type":"object","properties":{}}
          }}]
        }
        """#.utf8)
        let payload = try JSONDecoder().decode(CompletionsHandler.CompletionRequest.self, from: data)
        let request = try CompletionsHandler.coreRequest(from: payload)

        #expect(request.model == ModelID("alias"))
        #expect(request.options == .init(
            maxTokens: 17,
            temperature: 0.2,
            topP: 0.8,
            topK: 7,
            minP: 0.1,
            repetitionPenalty: 1.05,
            repetitionContextSize: 64,
            presencePenalty: 0.4,
            frequencyPenalty: 0.3,
            contextLimit: 4096
        ))
        #expect(request.messages.count == 4)
        #expect(try request.messages[1].content == [
            .text("look"),
            .imageURL(#require(URL(string: "https://example.invalid/image.png")))
        ])
        #expect(request.messages[2].toolCalls == [
            .init(id: "call-1", name: "lookup", arguments: ["key": .string("x")])
        ])
        #expect(request.messages[3].toolCallID == "call-1")
        #expect(request.tools == [.init(
            name: "lookup",
            description: "Lookup",
            parameters: .object([
                "type": .string("object"),
                "properties": .object([:])
            ])
        )])
    }

    @Test func nonStreamingHandlerUsesInjectedCoreAndPreservesWireResponse() async throws {
        let backend = HTTPStubBackend(
            generationResponse: .init(
                output: "answer",
                toolCalls: [.init(id: "call-1", name: "lookup", arguments: ["key": .string("x")])],
                usage: .init(promptTokens: 3, completionTokens: 2),
                finishReason: .toolCall,
                metrics: .init(promptSeconds: 0.1, generationSeconds: 0.2, tokensPerSecond: 10)
            )
        )
        let engine = SwamaEngine(backend: backend)
        let channel = NIOAsyncTestingChannel()
        try await channel.connect(to: .init(ipAddress: "127.0.0.1", port: 28100))
        await channel.testingEventLoop.run()
        let completed = AsyncFlag()
        let body = ByteBuffer(bytes: Data(#"""
        {"model":"org/model","messages":[{"role":"user","content":"hello"}],"stream":false}
        """#.utf8))

        let task = Task {
            await CompletionsHandler.handle(
                requestHead: .init(version: .http1_1, method: .POST, uri: "/v1/chat/completions"),
                body: body,
                channel: channel,
                engine: engine
            )
            await completed.set()
        }
        try await pump(channel, until: completed)
        await task.value
        await channel.testingEventLoop.run()

        let response = try await responseJSON(from: channel)
        #expect(response["model"] as? String == "org/model")
        let choices = try #require(response["choices"] as? [[String: Any]])
        #expect(choices.first?["finish_reason"] as? String == "tool_calls")
        let usage = try #require(response["usage"] as? [String: Any])
        #expect(usage["prompt_tokens"] as? Int == 3)
        #expect(usage["completion_tokens"] as? Int == 2)
        #expect(await backend.generationRequests.map(\.model) == [ModelID("org/model")])
    }

    @Test func embeddingHandlerUsesInjectedCore() async throws {
        let backend = HTTPStubBackend(
            embeddingResponse: .init(
                embeddings: [[1, 2], [3, 4]],
                usage: .init(promptTokens: 5, completionTokens: 0)
            )
        )
        let engine = SwamaEngine(backend: backend)
        let channel = NIOAsyncTestingChannel()
        let completed = AsyncFlag()
        let body = ByteBuffer(bytes: Data(#"""
        {"model":"org/embed","input":["one","two"]}
        """#.utf8))

        let task = Task {
            await EmbeddingsHandler.handle(
                requestHead: .init(version: .http1_1, method: .POST, uri: "/v1/embeddings"),
                body: body,
                channel: channel,
                engine: engine
            )
            await completed.set()
        }
        try await pump(channel, until: completed)
        await task.value
        await channel.testingEventLoop.run()

        let response = try await responseJSON(from: channel)
        let data = try #require(response["data"] as? [[String: Any]])
        #expect(data.count == 2)
        let usage = try #require(response["usage"] as? [String: Any])
        #expect(usage["prompt_tokens"] as? Int == 5)
        #expect(await backend.embeddingRequests == [
            .init(model: .init("org/embed"), inputs: ["one", "two"])
        ])
    }

    @Test func invalidEmbeddingModelReturnsBadRequestBeforeBackendExecution() async throws {
        let backend = HTTPStubBackend()
        let engine = SwamaEngine(backend: backend)
        let channel = NIOAsyncTestingChannel()
        let completed = AsyncFlag()
        let body = ByteBuffer(bytes: Data(#"""
        {"model":"../private/model","input":"hello"}
        """#.utf8))

        let task = Task {
            await EmbeddingsHandler.handle(
                requestHead: .init(version: .http1_1, method: .POST, uri: "/v1/embeddings"),
                body: body,
                channel: channel,
                engine: engine
            )
            await completed.set()
        }
        try await pump(channel, until: completed)
        await task.value
        await channel.testingEventLoop.run()

        let response = try await response(from: channel)
        #expect(response.status == .badRequest)
        #expect(await backend.embeddingRequests.isEmpty)
    }

    @Test func invalidCoreMessageFieldsReturnBadRequestBeforeBackendExecution() async throws {
        let backend = HTTPStubBackend()
        let engine = SwamaEngine(backend: backend)
        let channel = NIOAsyncTestingChannel()
        let completed = AsyncFlag()
        let body = ByteBuffer(bytes: Data(#"""
        {"model":"org/model","messages":[{"role":"tool","content":"result"}]}
        """#.utf8))

        let task = Task {
            await CompletionsHandler.handle(
                requestHead: .init(version: .http1_1, method: .POST, uri: "/v1/chat/completions"),
                body: body,
                channel: channel,
                engine: engine
            )
            await completed.set()
        }
        try await pump(channel, until: completed)
        await task.value
        await channel.testingEventLoop.run()

        let response = try await response(from: channel)
        #expect(response.status == .badRequest)
        let error = try #require(response.json["error"] as? [String: Any])
        #expect(error["type"] as? String == "invalid_request_error")
        #expect(await backend.generationRequests.isEmpty)
    }

    @Test func streamingHandlerPreservesCoreEventOrderAndTerminalUsage() async throws {
        let backend = HTTPStubBackend(
            generationResponse: .init(
                output: "hello",
                toolCalls: [.init(id: "call-1", name: "lookup", arguments: ["key": .string("x")])],
                usage: .init(promptTokens: 3, completionTokens: 2),
                finishReason: .toolCall
            ),
            generationEvents: [
                .textDelta("hel"),
                .textDelta("lo"),
                .toolCall(.init(id: "call-1", name: "lookup", arguments: ["key": .string("x")]))
            ]
        )
        let engine = SwamaEngine(backend: backend)
        let channel = NIOAsyncTestingChannel()
        try await channel.connect(to: .init(ipAddress: "127.0.0.1", port: 28100))
        await channel.testingEventLoop.run()
        let completed = AsyncFlag()
        let body = ByteBuffer(bytes: Data(#"""
        {"model":"org/model","messages":[{"role":"user","content":"hello"}],"stream":true}
        """#.utf8))

        let task = Task {
            await CompletionsHandler.handle(
                requestHead: .init(version: .http1_1, method: .POST, uri: "/v1/chat/completions"),
                body: body,
                channel: channel,
                engine: engine
            )
            await completed.set()
        }
        try await pump(channel, until: completed)
        await task.value
        await channel.testingEventLoop.run()

        let output = try await responseBodyString(from: channel)
        let role = try #require(output.range(of: #""role":"assistant""#))
        let firstText = try #require(output.range(of: #""content":"hel""#))
        let secondText = try #require(output.range(of: #""content":"lo""#))
        let tool = try #require(output.range(of: #""tool_calls""#))
        let terminal = try #require(output.range(of: #""finish_reason":"tool_calls""#))
        let done = try #require(output.range(of: "data: [DONE]"))
        #expect(role.lowerBound < firstText.lowerBound)
        #expect(firstText.lowerBound < secondText.lowerBound)
        #expect(secondText.lowerBound < tool.lowerBound)
        #expect(tool.lowerBound < terminal.lowerBound)
        #expect(terminal.lowerBound < done.lowerBound)
        #expect(output.contains(#""prompt_tokens":3"#))
        #expect(output.contains(#""completion_tokens":2"#))
    }

    private func pump(_ channel: NIOAsyncTestingChannel, until flag: AsyncFlag) async throws {
        for _ in 0 ..< 1000 {
            await channel.testingEventLoop.run()
            if await flag.value {
                return
            }
            await Task.yield()
        }
        throw HTTPToCoreTestError.timeout
    }

    private func responseJSON(from channel: NIOAsyncTestingChannel) async throws -> [String: Any] {
        let result = try await response(from: channel)
        return result.json
    }

    private func response(
        from channel: NIOAsyncTestingChannel
    ) async throws -> (status: HTTPResponseStatus, json: [String: Any]) {
        var status: HTTPResponseStatus?
        var bytes = ByteBuffer()
        while let part = try await channel.readOutbound(as: HTTPServerResponsePart.self) {
            switch part {
            case let .head(head):
                status = head.status
            case var .body(.byteBuffer(buffer)):
                bytes.writeBuffer(&buffer)
            case .body(.fileRegion),
                 .end:
                break
            }
        }
        let body = String(decoding: bytes.readableBytesView, as: UTF8.self)
        let data = Data(body.utf8)
        return try (
            #require(status),
            #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        )
    }

    private func responseBodyString(from channel: NIOAsyncTestingChannel) async throws -> String {
        var bytes = ByteBuffer()
        while let part = try await channel.readOutbound(as: HTTPServerResponsePart.self) {
            if case var .body(.byteBuffer(buffer)) = part {
                bytes.writeBuffer(&buffer)
            }
        }
        return String(decoding: bytes.readableBytesView, as: UTF8.self)
    }
}

// MARK: - HTTPStubBackend

private actor HTTPStubBackend: SwamaEngineBackend {
    init(
        generationResponse: GenerationResponse = .init(
            output: "",
            toolCalls: [],
            usage: .init(promptTokens: 0, completionTokens: 0),
            finishReason: .completed
        ),
        embeddingResponse: SwamaCore.EmbeddingResponse = .init(
            embeddings: [],
            usage: .init(promptTokens: 0, completionTokens: 0)
        ),
        generationEvents: [GenerationEvent] = []
    ) {
        self.generationResponse = generationResponse
        self.embeddingResponse = embeddingResponse
        self.generationEvents = generationEvents
    }

    func generate(
        _ request: GenerationRequest,
        onEvent: (@Sendable (GenerationEvent) async throws -> Void)?
    ) async throws -> GenerationResponse {
        generationRequests.append(request)
        for event in generationEvents {
            try await onEvent?(event)
        }
        return generationResponse
    }

    func embed(_ request: EmbeddingRequest) async throws -> SwamaCore.EmbeddingResponse {
        embeddingRequests.append(request)
        return embeddingResponse
    }

    func models() async throws -> [SwamaCore.ModelInfo] { [] }
    func fetch(_ model: ModelID) async throws -> ModelID { model }
    func remove(_: ModelID) async throws {}
    func clearCache(for _: ModelID) async {}
    func clearCache() async {}

    private let generationResponse: GenerationResponse
    private let embeddingResponse: SwamaCore.EmbeddingResponse
    private(set) var generationRequests: [GenerationRequest] = []
    private(set) var embeddingRequests: [EmbeddingRequest] = []
    private let generationEvents: [GenerationEvent]
}

// MARK: - AsyncFlag

private actor AsyncFlag {
    func set() { value = true }
    private(set) var value = false
}

// MARK: - HTTPToCoreTestError

private enum HTTPToCoreTestError: Error {
    case timeout
}

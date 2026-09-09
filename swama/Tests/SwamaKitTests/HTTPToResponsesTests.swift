import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
import SwamaCore
@testable import SwamaServer
import Testing

// MARK: - HTTPToResponsesTests

@MainActor @Suite("OpenAI Responses adapter", .serialized)
struct HTTPToResponsesTests {
    // MARK: Parsing / mapping

    @Test func stringInputBecomesUserMessageAndInstructionsBecomeSystem() throws {
        let parsed = try ResponsesHandler.parse(bytes(#"""
        {"model":"org/model","instructions":"be terse","input":"hello",
         "temperature":0.25,"top_p":0.7,"max_output_tokens":42}
        """#))
        #expect(parsed.stream == false)
        #expect(parsed.request.model == ModelID("org/model"))
        #expect(parsed.request.messages.count == 2)
        #expect(parsed.request.messages[0].role == .system)
        #expect(parsed.request.messages[0].content == [.text("be terse")])
        #expect(parsed.request.messages[1].role == .user)
        #expect(parsed.request.messages[1].content == [.text("hello")])
        #expect(parsed.request.options.maxTokens == 42)
        #expect(parsed.request.options.temperature == 0.25)
        #expect(parsed.request.options.topP == 0.7)
    }

    @Test func messageArrayInputWithImageAndFunctionToolMap() throws {
        let parsed = try ResponsesHandler.parse(bytes(#"""
        {"model":"m","stream":true,
         "input":[{"type":"message","role":"user","content":[
            {"type":"input_text","text":"describe"},
            {"type":"input_image","image_url":"https://example.com/a.png"}]}],
         "tools":[{"type":"function","name":"lookup","description":"d",
                   "parameters":{"type":"object","properties":{"k":{"type":"string"}}}}]}
        """#))
        #expect(parsed.stream == true)
        #expect(parsed.request.messages.count == 1)
        #expect(parsed.request.messages[0].content == [
            .text("describe"),
            .imageURL(URL(string: "https://example.com/a.png")!),
        ])
        #expect(parsed.request.tools.count == 1)
        #expect(parsed.request.tools[0].name == "lookup")
    }

    @Test func everyUnsupportedFeatureIsRejectedWith400NotSilentlyIgnored() {
        let rejected: [String] = [
            #"{"model":"m","input":"x","store":true}"#,
            #"{"model":"m","input":"x","previous_response_id":"resp_1"}"#,
            #"{"model":"m","input":"x","conversation":"conv_1"}"#,
            #"{"model":"m","input":"x","background":true}"#,
            #"{"model":"m","input":"x","truncation":"auto"}"#,
            #"{"model":"m","input":"x","include":["message.output_text.logprobs"]}"#,
            #"{"model":"m","input":"x","response_format":{"type":"json_schema"}}"#,
            #"{"model":"m","input":"x","text":{"format":{"type":"json_schema"}}}"#,
            #"{"model":"m","input":"x","tool_choice":"required"}"#,
            #"{"model":"m","input":"x","tool_choice":{"type":"function","name":"f"}}"#,
            #"{"model":"m","input":"x","tools":[{"type":"web_search_preview"}]}"#,
            #"{"model":"m","input":"x","tools":[{"type":"mcp","server_label":"s"}]}"#,
        ]
        for payload in rejected {
            #expect(throws: RejectionReason.self, "must reject: \(payload)") {
                _ = try ResponsesHandler.parse(bytes(payload))
            }
        }
    }

    @Test func supportedToolChoiceAndMissingModelBehaveCorrectly() throws {
        // auto / none are honourable and must NOT throw.
        _ = try ResponsesHandler.parse(bytes(#"{"model":"m","input":"x","tool_choice":"auto"}"#))
        _ = try ResponsesHandler.parse(bytes(#"{"model":"m","input":"x","tool_choice":"none"}"#))
        // missing model / empty input are 400s.
        #expect(throws: RejectionReason.self) {
            _ = try ResponsesHandler.parse(bytes(#"{"input":"x"}"#))
        }
        #expect(throws: RejectionReason.self) {
            _ = try ResponsesHandler.parse(bytes(#"{"model":"m","input":""}"#))
        }
        #expect(throws: RejectionReason.self) {
            _ = try ResponsesHandler.parse(bytes(#"{"model":"m","input":[]}"#))
        }
    }

    // MARK: Non-streaming response object

    @Test func nonStreamingBuildsResponseObjectFromInjectedCore() async throws {
        let backend = ResponsesStubBackend(response: .init(
            output: "the answer",
            toolCalls: [.init(id: "call-1", name: "lookup", arguments: ["k": .string("v")])],
            usage: .init(promptTokens: 5, completionTokens: 3),
            finishReason: .toolCall
        ))
        let json = try await run(backend: backend, body: #"""
        {"model":"org/m","input":"hi","stream":false}
        """#)
        #expect(json.status == .ok)
        #expect(json.body["object"] as? String == "response")
        #expect(json.body["status"] as? String == "completed")
        let output = try #require(json.body["output"] as? [[String: Any]])
        #expect(output.contains { $0["type"] as? String == "message" })
        #expect(output.contains { $0["type"] as? String == "function_call" })
        let usage = try #require(json.body["usage"] as? [String: Any])
        #expect(usage["input_tokens"] as? Int == 5)
        #expect(usage["output_tokens"] as? Int == 3)
        #expect(usage["total_tokens"] as? Int == 8)
        #expect(await backend.requests.map(\.model) == [ModelID("org/m")])
    }

    @Test func lengthFinishReasonYieldsIncompleteStatus() async throws {
        let backend = ResponsesStubBackend(response: .init(
            output: "partial", toolCalls: [],
            usage: .init(promptTokens: 1, completionTokens: 1), finishReason: .length
        ))
        let json = try await run(backend: backend, body: #"{"model":"m","input":"hi"}"#)
        #expect(json.body["status"] as? String == "incomplete")
    }

    // MARK: Streaming typed SSE

    @Test func streamingEmitsMonotonicSequenceAndTypedEvents() async throws {
        let backend = ResponsesStubBackend(
            response: .init(
                output: "hello world", toolCalls: [],
                usage: .init(promptTokens: 2, completionTokens: 2), finishReason: .completed
            ),
            events: [.textDelta("hello "), .textDelta("world")]
        )
        let events = try await runStream(backend: backend, body: #"""
        {"model":"m","input":"hi","stream":true}
        """#)
        // sequence_number strictly increasing from 0.
        let sequences = events.compactMap { $0["sequence_number"] as? Int }
        #expect(sequences == Array(0 ..< sequences.count))
        let types = events.compactMap { $0["type"] as? String }
        #expect(types.first == "response.created")
        #expect(types.contains("response.in_progress"))
        #expect(types.contains("response.output_item.added"))
        #expect(types.contains("response.output_text.delta"))
        #expect(types.contains("response.output_text.done"))
        #expect(types.last == "response.completed")
    }

    // MARK: - Harness

    private func bytes(_ json: String) -> ByteBuffer { ByteBuffer(bytes: Data(json.utf8)) }

    private func run(
        backend: ResponsesStubBackend,
        body: String
    ) async throws -> (status: HTTPResponseStatus, body: [String: Any]) {
        let (status, raw) = try await drive(backend: backend, body: body)
        let object = try #require(JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any])
        return (status, object)
    }

    private func runStream(backend: ResponsesStubBackend, body: String) async throws -> [[String: Any]] {
        let (_, raw) = try await drive(backend: backend, body: body)
        var events: [[String: Any]] = []
        for line in raw.split(separator: "\n") {
            guard line.hasPrefix("data: ") else { continue }

            let payload = line.dropFirst("data: ".count)
            if payload == "[DONE]" { continue }
            if let object = try? JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any] {
                events.append(object)
            }
        }
        return events
    }

    private func drive(
        backend: ResponsesStubBackend,
        body: String
    ) async throws -> (status: HTTPResponseStatus, body: String) {
        let engine = SwamaEngine(backend: backend)
        let channel = NIOAsyncTestingChannel()
        try await channel.connect(to: .init(ipAddress: "127.0.0.1", port: 28100))
        await channel.testingEventLoop.run()
        let done = ResponsesAsyncFlag()
        let buffer = ByteBuffer(bytes: Data(body.utf8))
        let task = Task {
            await ResponsesHandler.handle(
                requestHead: .init(version: .http1_1, method: .POST, uri: "/v1/responses"),
                body: buffer,
                channel: channel,
                engine: engine
            )
            await done.set()
        }
        for _ in 0 ..< 2000 {
            await channel.testingEventLoop.run()
            if await done.value {
                break
            }
            await Task.yield()
        }
        await task.value
        await channel.testingEventLoop.run()

        var status: HTTPResponseStatus?
        var out = ByteBuffer()
        while let part = try await channel.readOutbound(as: HTTPServerResponsePart.self) {
            switch part {
            case let .head(head): status = head.status
            case var .body(.byteBuffer(chunk)): out.writeBuffer(&chunk)
            default: break
            }
        }
        return (status ?? .internalServerError, String(decoding: out.readableBytesView, as: UTF8.self))
    }
}

// MARK: - ResponsesStubBackend

private actor ResponsesStubBackend: SwamaEngineBackend {
    init(response: GenerationResponse, events: [GenerationEvent] = []) {
        self.response = response
        self.events = events
    }

    func generate(
        _ request: GenerationRequest,
        onEvent: (@Sendable (GenerationEvent) async throws -> Void)?
    ) async throws -> GenerationResponse {
        requests.append(request)
        for event in events {
            try await onEvent?(event)
        }
        return response
    }

    func embed(_: EmbeddingRequest) async throws -> SwamaCore.EmbeddingResponse {
        .init(embeddings: [], usage: .init(promptTokens: 0, completionTokens: 0))
    }

    func models() async throws -> [SwamaCore.ModelInfo] { [] }
    func fetch(_ model: ModelID) async throws -> ModelID { model }
    func remove(_: ModelID) async throws {}
    func clearCache(for _: ModelID) async {}
    func clearCache() async {}

    private let response: GenerationResponse
    private let events: [GenerationEvent]
    private(set) var requests: [GenerationRequest] = []
}

// MARK: - ResponsesAsyncFlag

private actor ResponsesAsyncFlag {
    func set() { value = true }
    private(set) var value = false
}

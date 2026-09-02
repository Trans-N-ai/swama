import Foundation
@preconcurrency import MLXLMCommon
import NIOCore
import NIOEmbedded
import NIOHTTP1
@testable import SwamaKit
@testable import SwamaServer
import Testing

// MARK: - CompletionsHandlerTests

@MainActor @Suite(.serialized)
final class CompletionsHandlerTests {
    // MARK: - Test Data

    private func createValidCompletionRequest(stream: Bool = false) -> Data {
        let request: [String: Any] = [
            "model": "test-model",
            "messages": [
                [
                    "role": "user",
                    "content": "Hello, world!"
                ]
            ],
            "stream": stream,
            "temperature": 0.7,
            "max_tokens": 100
        ]
        return try! JSONSerialization.data(withJSONObject: request)
    }

    private func createInvalidCompletionRequest() -> Data {
        let request: [String: Any] = [
            "model": "test-model",
            "messages": [] // Empty messages should be invalid
        ]
        return try! JSONSerialization.data(withJSONObject: request)
    }

    // MARK: - Payload Parsing Tests

    @Test func parseValidPayload() {
        let requestData = createValidCompletionRequest()
        let buffer = ByteBuffer(bytes: requestData)

        let payload = CompletionsHandler.parsePayload(buffer)

        #expect(payload != nil)
        #expect(payload?.model == "test-model")
        #expect(payload?.messages.count == 1)
        #expect(payload?.messages.first?.role == "user")
        #expect(payload?.stream == false)
        #expect(payload?.temperature == 0.7)
        #expect(payload?.max_tokens == 100)
    }

    @Test func parseStreamingPayload() {
        let requestData = createValidCompletionRequest(stream: true)
        let buffer = ByteBuffer(bytes: requestData)

        let payload = CompletionsHandler.parsePayload(buffer)

        #expect(payload != nil)
        #expect(payload?.stream == true)
    }

    @Test func parseInvalidPayload() {
        let buffer = ByteBuffer(string: "invalid json")

        let payload = CompletionsHandler.parsePayload(buffer)

        #expect(payload == nil)
    }

    @Test func parseEmptyPayload() {
        let payload = CompletionsHandler.parsePayload(nil)

        #expect(payload == nil)
    }

    @Test func parsePayloadWithEmptyMessages() {
        let requestData = createInvalidCompletionRequest()
        let buffer = ByteBuffer(bytes: requestData)

        let payload = CompletionsHandler.parsePayload(buffer)

        #expect(payload != nil)
        #expect(payload?.messages.isEmpty == true)
    }

    // MARK: - Sampling Parameter Tests

    private func createSamplingCompletionRequest() -> Data {
        let request: [String: Any] = [
            "model": "test-model",
            "messages": [
                [
                    "role": "user",
                    "content": "Hello, world!"
                ]
            ],
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1.15,
            "repetition_context_size": 64,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3
        ]
        return try! JSONSerialization.data(withJSONObject: request)
    }

    @Test func parseSamplingParameters() {
        let requestData = createSamplingCompletionRequest()
        let buffer = ByteBuffer(bytes: requestData)

        let payload = CompletionsHandler.parsePayload(buffer)

        #expect(payload != nil)
        #expect(payload?.top_k == 40)
        #expect(payload?.min_p == 0.05)
        #expect(payload?.repetition_penalty == 1.15)
        #expect(payload?.repetition_context_size == 64)
        #expect(payload?.presence_penalty == 0.5)
        #expect(payload?.frequency_penalty == 0.3)
    }

    @Test func samplingParametersMapToGenerateParameters() throws {
        let requestData = createSamplingCompletionRequest()
        let buffer = ByteBuffer(bytes: requestData)

        let payload = try #require(CompletionsHandler.parsePayload(buffer))
        let parameters = CompletionsHandler.generateParameters(from: payload)

        #expect(parameters.topK == 40)
        #expect(parameters.minP == 0.05)
        #expect(parameters.repetitionPenalty == 1.15)
        #expect(parameters.repetitionContextSize == 64)
        #expect(parameters.presencePenalty == 0.5)
        #expect(parameters.frequencyPenalty == 0.3)
    }

    @Test func absentSamplingParametersKeepEngineDefaults() throws {
        let requestData = createValidCompletionRequest()
        let buffer = ByteBuffer(bytes: requestData)

        let payload = try #require(CompletionsHandler.parsePayload(buffer))
        #expect(payload.top_k == nil)
        #expect(payload.min_p == nil)
        #expect(payload.repetition_penalty == nil)
        #expect(payload.repetition_context_size == nil)
        #expect(payload.presence_penalty == nil)
        #expect(payload.frequency_penalty == nil)

        let parameters = CompletionsHandler.generateParameters(from: payload)
        let defaults = GenerateParameters()

        #expect(parameters.topK == defaults.topK)
        #expect(parameters.minP == defaults.minP)
        #expect(parameters.repetitionPenalty == defaults.repetitionPenalty)
        #expect(parameters.repetitionContextSize == defaults.repetitionContextSize)
        #expect(parameters.presencePenalty == defaults.presencePenalty)
        #expect(parameters.frequencyPenalty == defaults.frequencyPenalty)
    }

    // MARK: - Message Content Tests

    @Test func textMessageContent() {
        let content = CompletionsHandler.MessageContent.text("Hello, world!")

        #expect(content.textContent == "Hello, world!")
        #expect(content.imageURLs.isEmpty == true)
    }

    @Test func multimodalMessageContent() {
        let parts = [
            CompletionsHandler.ContentPartValue.text("What's in this image?"),
            CompletionsHandler.ContentPartValue.imageURL(
                CompletionsHandler.ImageURL(url: "data:image/png;base64,...")
            )
        ]
        let content = CompletionsHandler.MessageContent.multimodal(parts)

        #expect(content.textContent == "What's in this image?")
        #expect(content.imageURLs.count == 1)
        #expect(content.imageURLs.first == "data:image/png;base64,...")
    }

    // MARK: - Tool Choice Tests

    @Test func toolChoiceNone() throws {
        let jsonData = "\"none\"".data(using: .utf8)!
        let toolChoice = try JSONDecoder().decode(CompletionsHandler.ToolChoice.self, from: jsonData)

        if case .none = toolChoice {
            // Expected
        }
        else {
            Issue.record("Expected .none tool choice")
        }
    }

    @Test func toolChoiceAuto() throws {
        let jsonData = "\"auto\"".data(using: .utf8)!
        let toolChoice = try JSONDecoder().decode(CompletionsHandler.ToolChoice.self, from: jsonData)

        if case .auto = toolChoice {
            // Expected
        }
        else {
            Issue.record("Expected .auto tool choice")
        }
    }

    @Test func toolChoiceFunction() throws {
        let functionChoice: [String: Any] = [
            "type": "function",
            "function": ["name": "get_weather"]
        ]
        let jsonData = try JSONSerialization.data(withJSONObject: functionChoice)
        let jsonString = String(data: jsonData, encoding: .utf8)!
        let wrappedData = "\"\(jsonString.replacingOccurrences(of: "\"", with: "\\\""))\"".data(using: .utf8)!

        let toolChoice = try JSONDecoder().decode(CompletionsHandler.ToolChoice.self, from: wrappedData)

        if case let .function(name) = toolChoice {
            #expect(name == "get_weather")
        }
        else {
            Issue.record("Expected .function tool choice")
        }
    }

    // MARK: - Error Response Format Tests

    @Test func errorResponseFormat() throws {
        let errorJSON: [String: Any] = [
            "error": [
                "message": "Test error message",
                "type": "invalid_request_error",
                "code": 400
            ]
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: errorJSON)
        let parsedJSON = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]

        #expect(parsedJSON != nil)
        let error = parsedJSON?["error"] as? [String: Any]
        #expect(error?["message"] as? String == "Test error message")
        #expect(error?["type"] as? String == "invalid_request_error")
        #expect(error?["code"] as? Int == 400)
    }

    // MARK: - SSE Format Tests

    @Test func sSEErrorFormat() throws {
        let chunkId = "chatcmpl-test"
        let timestamp = Int(Date().timeIntervalSince1970)
        let model = "test-model"
        let errorMessage = "Failed to process the image: Height: 16 must be larger than factor: 28"

        let errorJSON: [String: Any] = [
            "id": chunkId,
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": model,
            "choices": [["index": 0, "delta": [:], "finish_reason": "error"]],
            "error": [
                "message": errorMessage,
                "type": "request_error"
            ]
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: errorJSON)
        let parsedJSON = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]

        #expect(parsedJSON != nil)
        #expect(parsedJSON?["id"] as? String == chunkId)
        #expect(parsedJSON?["object"] as? String == "chat.completion.chunk")
        #expect(parsedJSON?["model"] as? String == model)

        let choices = parsedJSON?["choices"] as? [[String: Any]]
        #expect(choices?.count == 1)
        #expect(choices?.first?["finish_reason"] as? String == "error")

        let error = parsedJSON?["error"] as? [String: Any]
        #expect(error?["message"] as? String == errorMessage)
        #expect(error?["type"] as? String == "request_error")
    }

    // MARK: - Completion Response Format Tests

    @Test func completionResponseFormat() throws {
        let response = CompletionsHandler.CompletionResponse(
            id: "chatcmpl-test",
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: "test-model",
            choices: [
                CompletionsHandler.CompletionChoice(
                    index: 0,
                    message: CompletionsHandler.Message(
                        role: "assistant",
                        content: .text("Hello! How can I help you today?")
                    ),
                    finish_reason: "stop"
                )
            ],
            usage: CompletionsHandler.CompletionUsage(
                prompt_tokens: 10,
                completion_tokens: 15,
                total_tokens: 25,
                response_token_s: 12.5,
                total_duration: 1.2
            )
        )

        let jsonData = try JSONEncoder().encode(response)
        let parsedJSON = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]

        #expect(parsedJSON != nil)
        #expect(parsedJSON?["object"] as? String == "chat.completion")
        #expect(parsedJSON?["model"] as? String == "test-model")

        let choices = parsedJSON?["choices"] as? [[String: Any]]
        #expect(choices?.count == 1)
        #expect(choices?.first?["finish_reason"] as? String == "stop")

        let usage = parsedJSON?["usage"] as? [String: Any]
        #expect(usage?["prompt_tokens"] as? Int == 10)
        #expect(usage?["completion_tokens"] as? Int == 15)
        #expect(usage?["total_tokens"] as? Int == 25)
    }

    // MARK: - Tool Calls Response Format Tests

    @Test func toolCallsResponseFormat() throws {
        let toolCall = CompletionsHandler.ResponseToolCall(
            id: "call_123",
            type: "function",
            function: CompletionsHandler.ResponseFunction(
                name: "get_weather",
                arguments: "{\"location\": \"San Francisco\"}"
            )
        )

        let jsonData = try JSONEncoder().encode(toolCall)
        let parsedJSON = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]

        #expect(parsedJSON != nil)
        #expect(parsedJSON?["id"] as? String == "call_123")
        #expect(parsedJSON?["type"] as? String == "function")

        let function = parsedJSON?["function"] as? [String: Any]
        #expect(function?["name"] as? String == "get_weather")
        #expect(function?["arguments"] as? String == "{\"location\": \"San Francisco\"}")
    }

    // MARK: - Mid-Stream Disconnect Cancellation Tests

    /// Closing the channel mid-operation must cancel the wrapped operation and let its
    /// (possibly partial) result flow back through normally, mirroring how
    /// `ModelRunner.runChat` breaks out of its generation loop on cancellation instead of
    /// throwing. Uses `NIOAsyncTestingChannel` (not `EmbeddedChannel`) because this test
    /// necessarily crosses suspension points, and the embedded event loop is
    /// single-thread-only.
    ///
    /// The inner loop is bounded (not `while !Task.isCancelled { ... }`) and the assertion
    /// is made on the *actual* value of `Task.isCancelled` observed once the loop exits, not
    /// set unconditionally after it -- see `runCancellingOnCloseCancelsOperationWhenChannelAlreadyClosed`'s
    /// doc comment for why: an unconditional `observedCancellation.set()` here would make this
    /// test pass even if cancellation silently broke, which is the false-pass hole this
    /// rewrite closes. (If cancellation genuinely regresses, this now fails cleanly instead of
    /// hanging on an unbounded loop or passing vacuously.)
    @Test func runCancellingOnCloseCancelsOperationWhenChannelCloses() async throws {
        let channel = NIOAsyncTestingChannel()
        let started = CancellationFlag()
        let observedCancellation = CancellationFlag()

        let task = Task<String, Error> {
            try await CompletionsHandler.runCancellingOnClose(channel: channel) {
                await started.set()
                var iterations = 0
                while !Task.isCancelled, iterations < 10000 {
                    await Task.yield()
                    iterations += 1
                }
                // Record whatever `Task.isCancelled` actually is once the loop exits --
                // `true` because cancellation fired, or `false` because the iteration budget
                // ran out first. Either way this reflects reality instead of asserting a fact
                // that was never checked.
                await observedCancellation.set(Task.isCancelled)
                return "partial"
            }
        }

        // Bounded wait for the operation to actually start (and register isCancelled polling)
        // before we close the channel, so the close is guaranteed to race an in-flight
        // operation rather than a not-yet-started one.
        var waited = 0
        while await !started.value, waited < 10000 {
            await Task.yield()
            waited += 1
        }
        #expect(await started.value == true)

        try await channel.close()
        await channel.testingEventLoop.run()

        let result = try await task.value

        #expect(result == "partial")
        #expect(await observedCancellation.value == true)
        #expect(channel.isActive == false)
    }

    /// If the channel is already closed before the operation is even started,
    /// `runCancellingOnClose` must not hang or attempt to run the operation to completion --
    /// it should observe cancellation promptly.
    ///
    /// Previously this test's inner loop exited on cancellation *or* after 10000 iterations,
    /// then set `observedCancellation` unconditionally afterward -- so the assertion below
    /// passed even on a build where cancellation never fired at all (the loop would just run
    /// out its budget and the flag still got set to `true` regardless). That made this a
    /// vacuous assertion: it could not fail no matter what the code under test did. This
    /// rewrite captures the *actual* `Task.isCancelled` observed when the bounded loop exits
    /// and asserts on that value, so a broken cancellation path now produces a genuine,
    /// non-hanging test failure instead of a silent false pass.
    ///
    /// This rewrite is expected to still PASS on current code -- `runCancellingOnClose` itself
    /// is not the bug here (see the ModelPool and keep-alive tests elsewhere for those); this
    /// closes a hole in how the test verifies it, not a live product defect.
    @Test func runCancellingOnCloseCancelsOperationWhenChannelAlreadyClosed() async throws {
        let channel = NIOAsyncTestingChannel()
        try await channel.close()
        await channel.testingEventLoop.run()

        let observedCancellation = CancellationFlag()

        let result = try await CompletionsHandler.runCancellingOnClose(channel: channel) {
            var iterations = 0
            while !Task.isCancelled, iterations < 10000 {
                await Task.yield()
                iterations += 1
            }
            await observedCancellation.set(Task.isCancelled)
            return "partial"
        }

        #expect(result == "partial")
        #expect(await observedCancellation.value == true)
    }

    // MARK: - Keep-Alive Task Retention Tests

    /// TASK RETENTION ON KEEP-ALIVE: `runCancellingOnClose` registers
    /// `channel.closeFuture.whenComplete { _ in task.cancel() }`, which captures `task`
    /// strongly. `closeFuture` only resolves when the *connection* closes, so on a keep-alive
    /// connection (where the channel stays open across many requests) that callback -- and
    /// the `Task` it captures -- is never released after any single request completes; it
    /// only goes away when the whole connection eventually closes. Each request on a
    /// keep-alive connection therefore adds one more permanently-retained completed `Task` to
    /// the channel's close future for the lifetime of the connection.
    ///
    /// Demonstrated with a weak reference to a sentinel object returned as `operation`'s
    /// *result* (`T`), not merely captured by it: a `Task`'s completion storage must keep its
    /// result alive for as long as the `Task` handle itself is reachable (so that `.value`
    /// stays valid for any observer), which is exactly the piece of state this defect leaks.
    /// (A sentinel merely *captured inside* the closure, rather than returned as its result,
    /// was tried first and found to be released as soon as the operation returns regardless
    /// of this bug -- a completed async closure's own captures are freed promptly once it
    /// stops running -- so that shape does not distinguish buggy from fixed behavior here.)
    /// The retained *result*, however, does: once the operation completes and every other
    /// strong reference to it is dropped, the sentinel should deallocate even though the
    /// channel remains open -- but it does not today.
    ///
    /// See `runCancellingOnCloseReleasesOperationResultAfterChannelCloses` below for the
    /// control: the same setup, but with the channel actually closed, where the sentinel
    /// *does* deallocate -- confirming the retention observed here is specifically tied to
    /// the channel staying open, not some other artifact.
    @Test func runCancellingOnCloseRetainsOperationResultWhileChannelStaysOpenOnKeepAlive() async throws {
        let channel = NIOAsyncTestingChannel()

        final class Sentinel: @unchecked Sendable {}
        weak var weakSentinel: Sentinel?

        do {
            let result = try await CompletionsHandler.runCancellingOnClose(channel: channel) {
                Sentinel()
            }
            weakSentinel = result
            // `result` (our only other strong reference to the sentinel) goes out of scope
            // at the end of this `do` block.
        }

        // Give ARC every opportunity to actually run before we check.
        for _ in 0 ..< 5 {
            await Task.yield()
        }

        #expect(
            weakSentinel == nil,
            Comment(rawValue: "operation result should be released once the request completes, even though " +
                "the keep-alive channel stays open -- it is retained today via " +
                "channel.closeFuture.whenComplete's strong capture of the completed Task")
        )

        // Clean up the testing channel so it doesn't leak past the test. This close happens
        // only *after* the assertion above, so it cannot be responsible for the (expected)
        // failure of that assertion.
        _ = try? await channel.finish()
    }

    /// Control for `runCancellingOnCloseRetainsOperationResultWhileChannelStaysOpenOnKeepAlive`:
    /// identical setup, but the channel is actually closed before we check. This should PASS,
    /// confirming that once `closeFuture` fires, the retaining callback runs/releases and the
    /// operation's result is free to deallocate -- i.e. the retention above is specifically a
    /// function of the channel staying open (the keep-alive scenario), not some unrelated
    /// reason a completed `Task`'s result might outlive the caller.
    @Test func runCancellingOnCloseReleasesOperationResultAfterChannelCloses() async throws {
        let channel = NIOAsyncTestingChannel()

        final class Sentinel: @unchecked Sendable {}
        weak var weakSentinel: Sentinel?

        do {
            let result = try await CompletionsHandler.runCancellingOnClose(channel: channel) {
                Sentinel()
            }
            weakSentinel = result
        }

        try await channel.close()
        await channel.testingEventLoop.run()

        for _ in 0 ..< 5 {
            await Task.yield()
        }

        #expect(weakSentinel == nil)
    }
}

/// Thread-safe boolean box used to coordinate assertions with work happening inside a
/// separately-scheduled `Task` in the tests above.
private actor CancellationFlag {
    private(set) var value = false

    /// Records the given value (defaulting to `true` for simple started/reached-here signals).
    /// Tests that need to assert on an *actual observed* condition -- rather than merely
    /// signalling that some line of code ran -- pass the condition explicitly, e.g.
    /// `set(Task.isCancelled)`.
    func set(_ newValue: Bool = true) {
        value = newValue
    }
}

// MARK: - Extension for Private Method Testing

extension CompletionsHandler {
    static func parsePayload(_ buffer: ByteBuffer?) -> CompletionRequest? {
        guard let buffer,
              let data = buffer.getBytes(at: buffer.readerIndex, length: buffer.readableBytes)
        else {
            return nil
        }

        return try? JSONDecoder().decode(CompletionRequest.self, from: Data(data))
    }
}

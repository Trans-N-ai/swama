import Foundation
import NIOCore
import NIOHTTP1
@testable import SwamaKit
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
                    finish_reason: "stop",
                    tool_calls: nil
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

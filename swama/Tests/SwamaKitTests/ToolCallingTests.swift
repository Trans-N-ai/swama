import Foundation
@testable import SwamaKit
import Testing

@MainActor @Suite(.serialized)
final class ToolCallingTests {
    // Note: Tool conversion tests are moved to integration tests since the method is private

    // MARK: - Function JSON Decoding Tests

    @Test func functionDecoding_WithObjectParameters() throws {
        let json = """
        {
            "name": "calculate",
            "description": "Perform calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string"
                    }
                }
            }
        }
        """

        let data = json.data(using: .utf8)!
        let function = try JSONDecoder().decode(CompletionsHandler.Function.self, from: data)

        #expect(function.name == "calculate")
        #expect(function.description == "Perform calculation")
        #expect(function.parameters != nil)

        // Verify parameters can be parsed back to JSON
        if let parameters = function.parameters {
            let parametersData = parameters.data(using: .utf8)!
            let parametersObj = try JSONSerialization.jsonObject(with: parametersData) as? [String: Any]
            #expect(parametersObj?["type"] as? String == "object")
        }
    }

    @Test func functionDecoding_WithoutParameters() throws {
        let json = """
        {
            "name": "simple_function",
            "description": "A simple function"
        }
        """

        let data = json.data(using: .utf8)!
        let function = try JSONDecoder().decode(CompletionsHandler.Function.self, from: data)

        #expect(function.name == "simple_function")
        #expect(function.description == "A simple function")
        #expect(function.parameters == nil)
    }

    // MARK: - ToolChoice Tests

    @Test func toolChoice_StringValues() throws {
        // Test string values
        let noneData = "\"none\"".data(using: .utf8)!
        let noneChoice = try JSONDecoder().decode(CompletionsHandler.ToolChoice.self, from: noneData)

        let autoData = "\"auto\"".data(using: .utf8)!
        let autoChoice = try JSONDecoder().decode(CompletionsHandler.ToolChoice.self, from: autoData)

        let requiredData = "\"required\"".data(using: .utf8)!
        let requiredChoice = try JSONDecoder().decode(CompletionsHandler.ToolChoice.self, from: requiredData)

        switch noneChoice {
        case .none: break
        default: #expect(Bool(false), "Expected .none")
        }

        switch autoChoice {
        case .auto: break
        default: #expect(Bool(false), "Expected .auto")
        }

        switch requiredChoice {
        case .required: break
        default: #expect(Bool(false), "Expected .required")
        }
    }

    @Test func toolChoice_FunctionObject() throws {
        let json = """
        {
            "type": "function",
            "function": {
                "name": "get_weather"
            }
        }
        """

        let data = json.data(using: .utf8)!
        let choice = try JSONDecoder().decode(CompletionsHandler.ToolChoice.self, from: data)

        switch choice {
        case let .function(name):
            #expect(name == "get_weather")
        default:
            #expect(Bool(false), "Expected .function")
        }
    }

    // MARK: - ResponseToolCall Tests

    @Test func responseToolCall_Encoding() throws {
        let toolCall = CompletionsHandler.ResponseToolCall(
            id: "call_123",
            type: "function",
            function: CompletionsHandler.ResponseFunction(
                name: "get_weather",
                arguments: "{\"location\": \"San Francisco\"}"
            )
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(toolCall)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        #expect(json?["id"] as? String == "call_123")
        #expect(json?["type"] as? String == "function")

        if let function = json?["function"] as? [String: Any] {
            #expect(function["name"] as? String == "get_weather")
            #expect(function["arguments"] as? String == "{\"location\": \"San Francisco\"}")
        }
    }

    // MARK: - CompletionRequest Tests

    @Test func completionRequest_WithTools() throws {
        let json = """
        {
            "model": "llama-3.1-8b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather like?"
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string"
                                }
                            }
                        }
                    }
                }
            ],
            "tool_choice": "auto"
        }
        """

        let data = json.data(using: .utf8)!
        let request = try JSONDecoder().decode(CompletionsHandler.CompletionRequest.self, from: data)

        #expect(request.model == "llama-3.1-8b-instruct")
        #expect(request.messages.count == 1)
        #expect(request.tools?.count == 1)

        if case .auto = request.tool_choice {
            // Expected
        }
        else {
            #expect(Bool(false), "Expected .auto tool choice")
        }

        if let tool = request.tools?.first {
            #expect(tool.type == "function")
            #expect(tool.function.name == "get_weather")
            #expect(tool.function.description == "Get current weather")
        }
    }

    @Test func completionRequest_WithoutTools() throws {
        let json = """
        {
            "model": "llama-3.1-8b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello world"
                }
            ]
        }
        """

        let data = json.data(using: .utf8)!
        let request = try JSONDecoder().decode(CompletionsHandler.CompletionRequest.self, from: data)

        #expect(request.model == "llama-3.1-8b-instruct")
        #expect(request.messages.count == 1)
        #expect(request.tools == nil)
        #expect(request.tool_choice == nil)
    }

    // MARK: - Message Content Tests

    @Test func messageContent_TextOnly() throws {
        let json = """
        {
            "role": "user",
            "content": "Hello world"
        }
        """

        let data = json.data(using: .utf8)!
        let message = try JSONDecoder().decode(CompletionsHandler.Message.self, from: data)

        #expect(message.role == "user")
        #expect(message.content.textContent == "Hello world")
        #expect(message.content.imageURLs.isEmpty)
    }

    @Test func messageContent_Multimodal() throws {
        let json = """
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
        """

        let data = json.data(using: .utf8)!
        let message = try JSONDecoder().decode(CompletionsHandler.Message.self, from: data)

        #expect(message.role == "user")
        #expect(message.content.textContent == "What's in this image?")
        #expect(message.content.imageURLs.count == 1)
        #expect(message.content.imageURLs.first == "https://example.com/image.jpg")
    }

    @Test func messageContent_ToolRole() throws {
        let json = """
        {
            "role": "tool",
            "content": "The weather in Tokyo is 22°C and sunny.",
            "tool_call_id": "call_123"
        }
        """

        let data = json.data(using: .utf8)!
        let message = try JSONDecoder().decode(CompletionsHandler.Message.self, from: data)

        #expect(message.role == "tool")
        #expect(message.content.textContent == "The weather in Tokyo is 22°C and sunny.")
        #expect(message.content.imageURLs.isEmpty)
    }

    @Test func completionRequest_WithToolMessages() throws {
        let json = """
        {
            "model": "llama-3.1-8b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather like in Tokyo?"
                },
                {
                    "role": "assistant",
                    "content": "I'll check the weather for you.",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\\"location\\": \\"Tokyo\\"}"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "content": "The weather in Tokyo is 22°C and sunny.",
                    "tool_call_id": "call_123"
                },
                {
                    "role": "assistant",
                    "content": "The weather in Tokyo is currently 22°C and sunny."
                }
            ]
        }
        """

        let data = json.data(using: .utf8)!
        let request = try JSONDecoder().decode(CompletionsHandler.CompletionRequest.self, from: data)

        #expect(request.model == "llama-3.1-8b-instruct")
        #expect(request.messages.count == 4)

        // Check that we have all the expected roles
        let roles = request.messages.map(\.role)
        #expect(roles == ["user", "assistant", "tool", "assistant"])

        // Check the tool message specifically
        let toolMessage = request.messages[2]
        #expect(toolMessage.role == "tool")
        #expect(toolMessage.content.textContent == "The weather in Tokyo is 22°C and sunny.")
    }

    // MARK: - CompletionResponse Tests

    @Test func completionResponse_WithToolCalls() throws {
        let toolCall = CompletionsHandler.ResponseToolCall(
            id: "call_123",
            function: CompletionsHandler.ResponseFunction(
                name: "get_weather",
                arguments: "{\"location\": \"Tokyo\"}"
            )
        )

        let choice = CompletionsHandler.CompletionChoice(
            index: 0,
            message: CompletionsHandler.Message(
                role: "assistant",
                content: .text("I'll get the weather for you.")
            ),
            finish_reason: "tool_calls",
            tool_calls: [toolCall]
        )

        let response = CompletionsHandler.CompletionResponse(
            id: "chatcmpl-123",
            object: "chat.completion",
            created: 1_234_567_890,
            model: "llama-3.1-8b-instruct",
            choices: [choice],
            usage: CompletionsHandler.CompletionUsage(
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                response_token_s: 20.0,
                total_duration: 0.75
            )
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(response)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        #expect(json?["id"] as? String == "chatcmpl-123")
        #expect(json?["object"] as? String == "chat.completion")
        #expect(json?["model"] as? String == "llama-3.1-8b-instruct")

        if let choices = json?["choices"] as? [[String: Any]],
           let firstChoice = choices.first,
           let toolCalls = firstChoice["tool_calls"] as? [[String: Any]],
           let firstToolCall = toolCalls.first
        {
            #expect(firstToolCall["id"] as? String == "call_123")
            #expect(firstToolCall["type"] as? String == "function")
        }
    }

    @Test func completionResponse_WithoutToolCalls() throws {
        let choice = CompletionsHandler.CompletionChoice(
            index: 0,
            message: CompletionsHandler.Message(
                role: "assistant",
                content: .text("Hello! How can I help you?")
            ),
            finish_reason: "stop",
            tool_calls: nil
        )

        let response = CompletionsHandler.CompletionResponse(
            id: "chatcmpl-456",
            object: "chat.completion",
            created: 1_234_567_890,
            model: "llama-3.1-8b-instruct",
            choices: [choice],
            usage: CompletionsHandler.CompletionUsage(
                prompt_tokens: 5,
                completion_tokens: 10,
                total_tokens: 15,
                response_token_s: 15.0,
                total_duration: 1.0
            )
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(response)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        #expect(json?["id"] as? String == "chatcmpl-456")

        if let choices = json?["choices"] as? [[String: Any]],
           let firstChoice = choices.first
        {
            // tool_calls should not be present when nil/empty
            #expect(firstChoice["tool_calls"] == nil)
        }
    }
}

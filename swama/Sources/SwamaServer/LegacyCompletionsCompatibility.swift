import Foundation
@preconcurrency import MLXLMCommon
import NIOCore
import NIOHTTP1
import SwamaKit
import struct Tokenizers.ToolSpec

// MARK: - Legacy source compatibility

public extension CompletionsHandler {
    @available(*, deprecated, message: "Use the HTTP route or SwamaCore.SwamaEngine")
    static func sendNonStreamResponse(
        channel: Channel,
        modelName: String,
        chatMessages: [MLXLMCommon.Chat.Message],
        model: String,
        parameters: GenerateParameters,
        mlxTools: [ToolSpec]? = nil
    ) async throws {
        let result = try await runCancellingOnClose(channel: channel) {
            try await ServerModelPool.shared.run(modelName: modelName) { runner in
                try await runner.runChatNonStream(
                    userInput: legacyUserInput(chatMessages, modelName: modelName, tools: mlxTools),
                    parameters: parameters
                )
            }
        }
        guard channel.isActive else {
            return
        }

        let completionTokens = result.completionInfo?.generationTokenCount ?? 0
        let toolCalls: [ResponseToolCall]? = result.toolCalls.isEmpty ? nil : result.toolCalls
            .enumerated()
            .map { index, toolCall in
                ResponseToolCall(
                    index: index,
                    id: toolCall.id ?? "call_\(UUID().uuidString)",
                    type: "function",
                    function: ResponseFunction(
                        name: toolCall.function.name,
                        arguments: legacyArguments(toolCall)
                    )
                )
            }

        let response = CompletionResponse(
            id: "chatcmpl-" + UUID().uuidString,
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: model,
            choices: [.init(
                index: 0,
                message: .init(role: "assistant", content: .text(result.output), tool_calls: toolCalls),
                finish_reason: toolCalls?.isEmpty == false ? "tool_calls" : "stop"
            )],
            usage: .init(
                prompt_tokens: result.promptTokens,
                completion_tokens: completionTokens,
                total_tokens: result.promptTokens + completionTokens,
                response_token_s: result.completionInfo?.tokensPerSecond,
                total_duration: result.completionInfo.map { $0.promptTime + $0.generateTime }
            )
        )
        let data = try JSONEncoder().encode(response)
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json")
        HTTPHandler.applyCORSHeaders(&headers)
        try await channel.writeAndFlush(HTTPServerResponsePart.head(.init(
            version: .http1_1,
            status: .ok,
            headers: headers
        )))
        try await channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(.init(bytes: data))))
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
    }

    @available(*, deprecated, message: "Use the HTTP route or SwamaCore.SwamaEngine")
    static func sendStreamResponse(
        channel: Channel,
        modelName: String,
        chatMessages: [MLXLMCommon.Chat.Message],
        model: String,
        parameters: GenerateParameters,
        tools: [ToolSpec]? = nil
    ) async throws {
        actor ToolCallCounter {
            private var index = 0
            func next() -> Int {
                defer { index += 1 }
                return index
            }
        }

        let headers = HTTPHeaders([
            ("Content-Type", "text/event-stream"),
            ("Cache-Control", "no-cache"),
            ("Connection", "keep-alive")
        ])
        try await channel.writeAndFlush(HTTPServerResponsePart.head(.init(
            version: .http1_1,
            status: .ok,
            headers: headers
        )))

        let chunkID = "chatcmpl-" + UUID().uuidString
        let timestamp = Int(Date().timeIntervalSince1970)
        try await legacyWriteSSE(channel: channel, payload: [
            "id": chunkID,
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": model,
            "choices": [["index": 0, "delta": ["role": "assistant"], "finish_reason": NSNull()]]
        ])

        let result: ModelRunner.ChatRunResult
        do {
            result = try await runCancellingOnClose(channel: channel) {
                try await ServerModelPool.shared.run(modelName: modelName) { runner in
                    let counter = ToolCallCounter()
                    return try await runner.runChat(
                        userInput: legacyUserInput(chatMessages, modelName: modelName, tools: tools),
                        parameters: parameters,
                        onToken: { chunk in
                            try await legacyWriteSSE(channel: channel, payload: [
                                "id": chunkID,
                                "object": "chat.completion.chunk",
                                "created": timestamp,
                                "model": model,
                                "choices": [[
                                    "index": 0,
                                    "delta": ["content": chunk],
                                    "finish_reason": NSNull()
                                ]]
                            ])
                        },
                        onToolCall: { toolCall in
                            let index = await counter.next()
                            try await legacyWriteSSE(channel: channel, payload: [
                                "id": chunkID,
                                "object": "chat.completion.chunk",
                                "created": timestamp,
                                "model": model,
                                "choices": [[
                                    "index": 0,
                                    "delta": ["tool_calls": [[
                                        "index": index,
                                        "id": toolCall.id ?? "call_\(UUID().uuidString)",
                                        "type": "function",
                                        "function": [
                                            "name": toolCall.function.name,
                                            "arguments": legacyArguments(toolCall)
                                        ]
                                    ]]],
                                    "finish_reason": NSNull()
                                ]]
                            ])
                        }
                    )
                }
            }
        }
        catch {
            guard channel.isActive else {
                return
            }

            try await legacyWriteSSE(channel: channel, payload: [
                "id": chunkID,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model,
                "choices": [["index": 0, "delta": [:], "finish_reason": "error"]],
                "error": ["message": error.localizedDescription, "type": "request_error"]
            ])
            try await legacyWriteLine(channel: channel, line: "data: [DONE]\n\n")
            try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
            return
        }

        guard channel.isActive else {
            return
        }

        let completionTokens = result.completionInfo?.generationTokenCount ?? 0
        try await legacyWriteSSE(channel: channel, payload: [
            "id": chunkID,
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": model,
            "choices": [[
                "index": 0,
                "delta": [:],
                "finish_reason": result.toolCalls.isEmpty ? "stop" : "tool_calls"
            ]],
            "usage": [
                "prompt_tokens": result.promptTokens,
                "completion_tokens": completionTokens,
                "total_tokens": result.promptTokens + completionTokens,
                "response_token/s": result.completionInfo?.tokensPerSecond ?? 0,
                "total_duration": (result.completionInfo?.promptTime ?? 0) +
                    (result.completionInfo?.generateTime ?? 0)
            ]
        ])
        try await legacyWriteLine(channel: channel, line: "data: [DONE]\n\n")
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
    }
}

private func legacyUserInput(
    _ messages: [MLXLMCommon.Chat.Message],
    modelName: String,
    tools: [ToolSpec]?
) -> MLXLMCommon.UserInput {
    let hasMedia = messages.contains { !$0.images.isEmpty || !$0.videos.isEmpty }
    let lowered = modelName.lowercased()
    if hasMedia, lowered.contains("qwen3.5") || lowered.contains("qwen3_5") {
        return .init(
            chat: messages,
            processing: .init(resize: .init(width: 1344, height: 1344)),
            tools: tools
        )
    }
    return .init(chat: messages, tools: tools)
}

private func legacyArguments(_ toolCall: MLXLMCommon.ToolCall) -> String {
    let object = toolCall.function.arguments.mapValues(\.anyValue)
    guard let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) else {
        return "{}"
    }

    return String(decoding: data, as: UTF8.self)
}

private func legacyWriteSSE(channel: Channel, payload: [String: Any]) async throws {
    let data = try JSONSerialization.data(withJSONObject: payload)
    try await legacyWriteLine(
        channel: channel,
        line: "data: \(String(decoding: data, as: UTF8.self))\n\n"
    )
}

private func legacyWriteLine(channel: Channel, line: String) async throws {
    var buffer = channel.allocator.buffer(capacity: line.utf8.count)
    buffer.writeString(line)
    try await channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(buffer)))
}

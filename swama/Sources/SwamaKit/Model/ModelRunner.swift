import CoreImage
import Foundation
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon
import MLXVLM
import OSLog
import Tokenizers
import struct Tokenizers.ToolSpec

// MARK: - ModelRunner

/// An actor responsible for running model inference.
private let modelRunnerLogger: Logger = .init(subsystem: "SwamaKit", category: "ModelRunner")

// MARK: - ModelRunner

public actor ModelRunner {
    public struct ChatRunResult: Sendable {
        public let output: String
        public let analysis: String?
        public let promptTokens: Int
        public let completionInfo: GenerateCompletionInfo?
        public let toolCalls: [MLXLMCommon.ToolCall]
        public let rawText: String
    }

    // MARK: Lifecycle

    public init(container: ModelContainer) {
        self.container = container
    }

    // MARK: Public

    /// Runs the model with the given prompt and parameters, returning only the generated output string.
    public func run(prompt: String, images _: [Data]? = nil, parameters: GenerateParameters) async throws -> String {
        // Use new chat-based method for consistency
        let chatMessages: [MLXLMCommon.Chat.Message] = [.user(prompt)]
        let result = try await runWithChatUsage(chatMessages: chatMessages, parameters: parameters)
        return result.output
    }

    /// Runs the model with chat messages, returning the generated output and token usage.
    public nonisolated func runWithChatUsage(
        chatMessages: [MLXLMCommon.Chat.Message],
        parameters: GenerateParameters
    ) async throws -> ChatRunResult {
        let userInput = MLXLMCommon.UserInput(chat: chatMessages)
        return try await runChat(
            userInput: userInput,
            parameters: parameters
        )
    }

    /// Non-streaming chat execution - collects all output and returns at the end
    public nonisolated func runChatNonStream(
        userInput: MLXLMCommon.UserInput,
        parameters: GenerateParameters
    ) async throws -> ChatRunResult {
        // For non-streaming, we don't provide callbacks, so runChat will accumulate internally
        try await runChat(
            userInput: userInput,
            parameters: parameters
        )
    }

    /// Unified method for running chat with optional streaming and tool calls support
    public nonisolated func runChat(
        userInput: MLXLMCommon.UserInput,
        parameters: GenerateParameters,
        onToken: (@Sendable (String) -> Void)? = nil,
        onToolCall: (@Sendable (MLXLMCommon.ToolCall) -> Void)? = nil
    ) async throws -> ChatRunResult {
        try await container.perform { (context: ModelContext) in
            var output = ""
            var promptTokens = 0
            var capturedCompletionInfo: GenerateCompletionInfo?
            var toolCalls: [MLXLMCommon.ToolCall] = []

            let rawOutputStorage = RawOutputBuffer()

            var effectiveInput = userInput
            if case let .chat(messages) = userInput.prompt {
                let limit = await ContextLimitConfig.shared.currentLimit()
                let trimmedMessages = try await trimChatMessagesInternal(
                    chatMessages: messages,
                    tools: userInput.tools,
                    limit: limit,
                    context: context,
                    processing: userInput.processing,
                    additionalContext: userInput.additionalContext
                )
                effectiveInput = MLXLMCommon.UserInput(
                    chat: trimmedMessages,
                    processing: userInput.processing,
                    tools: userInput.tools,
                    additionalContext: userInput.additionalContext
                )
            }

            let lmInput = try await context.processor.prepare(input: effectiveInput)

            promptTokens = lmInput.text.tokens.count

            let generationStream = try generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )

            for await generationEvent in generationStream {
                switch generationEvent {
                case let .chunk(chunkString):
                    rawOutputStorage.append(chunkString)

                    onToken?(chunkString)
                    // Only accumulate if no onToken callback (for non-streaming)
                    if onToken == nil {
                        output += chunkString
                    }

                case let .info(info):
                    capturedCompletionInfo = info

                case let .toolCall(toolCall):
                    // Always accumulate tool calls for the return value
                    toolCalls.append(toolCall)
                    // Also send to callback if provided (for streaming)
                    onToolCall?(toolCall)
                }
            }

            let rawOutput = rawOutputStorage.consume()
            let resolvedOutput = output.isEmpty ? rawOutput : output

            return ChatRunResult(
                output: resolvedOutput,
                analysis: nil,
                promptTokens: promptTokens,
                completionInfo: capturedCompletionInfo,
                toolCalls: toolCalls,
                rawText: rawOutput
            )
        }
    }

    // MARK: - Existing methods

    // MARK: Private

    private let container: ModelContainer
}

// MARK: - RawOutputBuffer

private final class RawOutputBuffer: @unchecked Sendable {
    private var storage: String = ""

    func append(_ chunk: String) {
        storage.append(chunk)
    }

    func consume() -> String {
        defer { storage.removeAll(keepingCapacity: false) }
        return storage
    }
}

private func trimChatMessagesInternal(
    chatMessages: [MLXLMCommon.Chat.Message],
    tools: [ToolSpec]?,
    limit: Int,
    context: ModelContext,
    processing: MLXLMCommon.UserInput.Processing,
    additionalContext: [String: any Sendable]?
) async throws -> [MLXLMCommon.Chat.Message] {
    guard limit > 0 else {
        return chatMessages
    }
    guard !chatMessages.isEmpty else {
        return chatMessages
    }

    func isProtected(_ message: MLXLMCommon.Chat.Message) -> Bool {
        if message.role == .system || message.role == .tool {
            return true
        }
        return !message.images.isEmpty || !message.videos.isEmpty
    }

    func buildInput(with messages: [MLXLMCommon.Chat.Message]) -> MLXLMCommon.UserInput {
        MLXLMCommon.UserInput(
            chat: messages,
            processing: processing,
            tools: tools,
            additionalContext: additionalContext
        )
    }

    var workingMessages = chatMessages
    var trimmableIndices = workingMessages.enumerated()
        .filter { !isProtected($0.element) }
        .map(\.offset)

    var currentTokenCount = estimateTokenCount(
        messages: workingMessages,
        tools: tools,
        additionalContext: additionalContext,
        context: context
    )
    let initialTokenCount = currentTokenCount

    var trimPointer = 0

    while currentTokenCount > limit, trimPointer < trimmableIndices.count {
        let index = trimmableIndices[trimPointer]
        let originalContent = workingMessages[index].content

        if originalContent.isEmpty {
            trimPointer += 1
            continue
        }

        let tokens = context.tokenizer.encode(text: originalContent)
        if tokens.isEmpty {
            workingMessages[index].content = ""
        }
        else {
            var bestContent: String?
            var low = 0
            var high = tokens.count

            while low <= high {
                let mid = (low + high) / 2
                let prefix = Array(tokens.prefix(mid))
                let decoded = context.tokenizer.decode(tokens: prefix)
                workingMessages[index].content = decoded

                let count = estimateTokenCount(
                    messages: workingMessages,
                    tools: tools,
                    additionalContext: additionalContext,
                    context: context
                )
                if count <= limit {
                    bestContent = decoded
                    low = mid + 1
                }
                else {
                    high = mid - 1
                }
            }

            workingMessages[index].content = bestContent ?? ""
        }

        currentTokenCount = estimateTokenCount(
            messages: workingMessages,
            tools: tools,
            additionalContext: additionalContext,
            context: context
        )

        if workingMessages[index].content.isEmpty {
            workingMessages.remove(at: index)
            trimmableIndices.remove(at: trimPointer)
            trimmableIndices = trimmableIndices.map { $0 > index ? $0 - 1 : $0 }
            continue
        }

        if currentTokenCount > limit {
            trimPointer += 1
        }
    }

    let finalInput = buildInput(with: workingMessages)
    let finalTokenCount = try await tokenCount(for: finalInput, context: context)

    guard finalTokenCount <= limit else {
        modelRunnerLogger.error(
            "Context limit hit; prompt still \(finalTokenCount) tokens with limit \(limit)"
        )
        throw ContextLimitError.exceededAfterTrimming(limit: limit, promptTokens: finalTokenCount)
    }

    if finalTokenCount < initialTokenCount {
        modelRunnerLogger.info(
            "Context trimmed to \(finalTokenCount) tokens (limit \(limit))"
        )
    }

    return workingMessages
}

private func tokenCount(for input: MLXLMCommon.UserInput, context: ModelContext) async throws -> Int {
    let prepared = try await context.processor.prepare(input: input)
    return prepared.text.tokens.count
}

private func estimateTokenCount(
    messages: [MLXLMCommon.Chat.Message],
    tools: [ToolSpec]?,
    additionalContext: [String: any Sendable]?,
    context: ModelContext
) -> Int {
    let rawMessages: [MLXLMCommon.Message] = messages.map { message in
        [
            "role": message.role.rawValue,
            "content": message.content
        ]
    }

    let templateTokens: [Int]
    do {
        templateTokens = try context.tokenizer.applyChatTemplate(
            messages: rawMessages,
            tools: tools,
            additionalContext: additionalContext
        )
    }
    catch {
        let prompt = messages.map(\.content).joined(separator: "\n\n")
        templateTokens = context.tokenizer.encode(text: prompt)
    }

    let mediaItems = messages.reduce(into: 0) { count, message in
        count += message.images.count
        count += message.videos.count
    }
    let estimatedMediaTokens = mediaItems * 400

    return templateTokens.count + estimatedMediaTokens
}

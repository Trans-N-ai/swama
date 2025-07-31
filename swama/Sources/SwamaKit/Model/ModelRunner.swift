import CoreImage
import Foundation
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon
import MLXVLM
import Tokenizers

// MARK: - ModelRunner

/// An actor responsible for running model inference.
public actor ModelRunner {
    // MARK: Lifecycle

    public init(container: ModelContainer) {
        self.container = container
    }

    // MARK: Public

    /// Runs the model with the given prompt and parameters, returning only the generated output string.
    public func run(prompt: String, images _: [Data]? = nil, parameters: GenerateParameters) async throws -> String {
        // Use new chat-based method for consistency
        let chatMessages: [MLXLMCommon.Chat.Message] = [.user(prompt)]
        let (output, _, _) = try await runWithChatUsage(chatMessages: chatMessages, parameters: parameters)
        return output
    }

    /// Runs the model with chat messages, returning the generated output and token usage.
    public nonisolated func runWithChatUsage(
        chatMessages: [MLXLMCommon.Chat.Message],
        parameters: GenerateParameters
    ) async throws -> (String, Int, Int) {
        let userInput = MLXLMCommon.UserInput(chat: chatMessages)
        let result = try await runChat(
            userInput: userInput,
            parameters: parameters
        )
        let completionTokens = result.completionInfo?.generationTokenCount ?? 0
        return (result.output, result.promptTokens, completionTokens)
    }

    /// Non-streaming chat execution - collects all output and returns at the end
    public nonisolated func runChatNonStream(
        userInput: MLXLMCommon.UserInput,
        parameters: GenerateParameters
    ) async throws
        -> (
            output: String,
            promptTokens: Int,
            completionInfo: GenerateCompletionInfo?,
            toolCalls: [MLXLMCommon.ToolCall]
        )
    {
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
    ) async throws
        -> (
            output: String,
            promptTokens: Int,
            completionInfo: GenerateCompletionInfo?,
            toolCalls: [MLXLMCommon.ToolCall]
        )
    {
        try await container.perform { (context: ModelContext) in
            var output = ""
            var promptTokens = 0
            var capturedCompletionInfo: GenerateCompletionInfo?
            var toolCalls: [MLXLMCommon.ToolCall] = []

            // Use the provided UserInput directly
            let lmInput = try await context.processor.prepare(input: userInput)

            promptTokens = lmInput.text.tokens.count

            let generationStream = try generate(
                input: lmInput,
                parameters: parameters,
                context: context
            )

            for await generationEvent in generationStream {
                switch generationEvent {
                case let .chunk(chunkString):
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

            return (
                output: output,
                promptTokens: promptTokens,
                completionInfo: capturedCompletionInfo,
                toolCalls: toolCalls
            )
        }
    }

    // MARK: - Existing methods

    // MARK: Private

    private let container: ModelContainer
}

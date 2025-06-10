import CoreImage
import Foundation
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon
import MLXVLM

/// Loads a model container for the given model name.
/// This function utilizes MLXLMCommon to handle caching or downloading of the model.
public func loadModelContainer(modelName: String) async throws -> ModelContainer {
    // MLXLMCommon will interpret the modelName as a Hugging Face repo ID or check its internal cache.
    let config = ModelConfiguration(id: modelName)

    let container: ModelContainer
    do {
        container = try await LLMModelFactory.shared.loadContainer(configuration: config)
    }
    catch {
        fputs(
            "SwamaKit.ModelRunner: Error loading model container for '\(modelName)': \(error.localizedDescription)\n",
            stderr
        )
        if let nsError = error as NSError? {
            fputs("  Error Code: \(nsError.code), Domain: \(nsError.domain)\n", stderr)
            if let underlying = nsError.userInfo[NSUnderlyingErrorKey] as? Error {
                fputs("  Underlying Error: \(underlying.localizedDescription)\n", stderr)
            }
        }
        throw error
    }

    return container
}

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
        let (output, promptTokens, completionInfo) = try await runWithChatUsageAndPerformance(
            chatMessages: chatMessages,
            parameters: parameters
        )
        let completionTokens = completionInfo?.generationTokenCount ?? 0
        return (output, promptTokens, completionTokens)
    }

    /// Runs the model with chat messages, returning the generated output, token usage, and performance metrics.
    public nonisolated func runWithChatUsageAndPerformance(
        chatMessages: [MLXLMCommon.Chat.Message],
        parameters: GenerateParameters
    ) async throws -> (String, Int, GenerateCompletionInfo?) {
        try await container.perform { (context: ModelContext) in
            var output = ""
            var promptTokens = 0
            var capturedCompletionInfo: GenerateCompletionInfo?

            // Create UserInput from chat messages
            let userInput = MLXLMCommon.UserInput(chat: chatMessages)
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
                    output += chunkString
                case let .info(info):
                    capturedCompletionInfo = info
                }
            }

            return (output, promptTokens, capturedCompletionInfo)
        }
    }

    /// Runs the model with chat messages in streaming mode, calling onToken for each generated token.
    public nonisolated func runWithChatStream(
        chatMessages: [MLXLMCommon.Chat.Message],
        parameters: GenerateParameters,
        onToken: @escaping @Sendable (String) -> Void
    ) async throws -> (promptTokens: Int, completionInfo: GenerateCompletionInfo?) {
        try await container.perform { (context: ModelContext) in
            var promptTokens = 0
            var capturedCompletionInfo: GenerateCompletionInfo?

            // Create UserInput from chat messages
            let userInput = MLXLMCommon.UserInput(chat: chatMessages)
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
                    onToken(chunkString)
                case let .info(info):
                    capturedCompletionInfo = info
                }
            }

            return (promptTokens, capturedCompletionInfo)
        }
    }

    // MARK: - Existing methods

    // MARK: Private

    private let container: ModelContainer
}

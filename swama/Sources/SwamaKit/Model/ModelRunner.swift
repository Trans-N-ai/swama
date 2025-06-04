import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
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

    /// Runs the model with the given prompt, optional images, and parameters, returning the generated output and token
    /// usage.
    public func runWithUsage(
        prompt: String,
        images: [Data]? = nil,
        parameters: GenerateParameters
    ) async throws -> (output: String, promptTokens: Int, completionTokens: Int) {
        try await container.perform { (context: MLXLMCommon.ModelContext) async throws -> (
            output: String,
            promptTokens: Int,
            completionTokens: Int
        ) in
            let inputText = prompt
            var promptTokens = 0

            var ciImages: [CIImage]?
            if let imageDataArray = images, !imageDataArray.isEmpty {
                autoreleasepool {
                    ciImages = imageDataArray.compactMap { data -> CIImage? in
                        return CIImage(data: data)
                    }
                }
                if ciImages?.isEmpty == true, !imageDataArray.isEmpty {
                    ciImages = nil
                }
            }

            let userInput: MLXLMCommon.UserInput
            var lmInput: MLXLMCommon.LMInput

            if let validCIImages = ciImages, !validCIImages.isEmpty, context.model is MLXVLM.VLMModel {
                let mlxUserInputImages = validCIImages.map { MLXLMCommon.UserInput.Image.ciImage($0) }
                let chatMessages: [MLXLMCommon.Chat.Message] = [
                    .user(inputText, images: mlxUserInputImages)
                ]
                userInput = MLXLMCommon.UserInput(chat: chatMessages)
                lmInput = try await context.processor.prepare(input: userInput)
            }
            else {
                userInput = MLXLMCommon.UserInput(prompt: inputText)
                lmInput = try await context.processor.prepare(input: userInput)
            }

            promptTokens = lmInput.text.tokens.count

            var output = ""
            var completionTokens = 0
            var detokenizer = StreamingDetokenizer(tokenizer: context.tokenizer)

            _ = try generate(input: lmInput, parameters: parameters, context: context) { token in
                completionTokens += 1
                if let chunk = detokenizer.append(token: token) {
                    output += chunk
                }
                return .more
            }

            return (output, promptTokens, completionTokens)
        }
    }

    /// Runs the model with the given prompt, optional images, and parameters, streaming the output and returning token
    /// usage.
    public func runStreamWithUsage(
        prompt: String,
        images: [Data]? = nil,
        parameters: GenerateParameters,
        onToken: @escaping @Sendable (String) -> Void
    ) async throws -> (promptTokens: Int, completionInfo: GenerateCompletionInfo?) {
        try await container.perform { (context: MLXLMCommon.ModelContext) async throws -> (
            promptTokens: Int,
            completionInfo: GenerateCompletionInfo?
        ) in
            let inputText = prompt
            var promptTokens = 0
            var capturedCompletionInfo: GenerateCompletionInfo?

            var ciImages: [CIImage]?
            if let imageDataArray = images, !imageDataArray.isEmpty {
                autoreleasepool {
                    ciImages = imageDataArray.compactMap { data -> CIImage? in
                        return CIImage(data: data)
                    }
                }
                if ciImages?.isEmpty == true, !imageDataArray.isEmpty {
                    ciImages = nil
                }
            }

            let userInput: MLXLMCommon.UserInput
            var lmInput: MLXLMCommon.LMInput

            if let validCIImages = ciImages, !validCIImages.isEmpty, context.model is MLXVLM.VLMModel {
                let mlxUserInputImages = validCIImages.map { MLXLMCommon.UserInput.Image.ciImage($0) }
                let chatMessages: [MLXLMCommon.Chat.Message] = [
                    .user(inputText, images: mlxUserInputImages)
                ]
                userInput = MLXLMCommon.UserInput(chat: chatMessages)
                lmInput = try await context.processor.prepare(input: userInput)
            }
            else {
                userInput = MLXLMCommon.UserInput(prompt: inputText)
                lmInput = try await context.processor.prepare(input: userInput)
            }

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

    /// Runs the model with the given prompt and parameters, returning only the generated output string.
    public func run(prompt: String, images: [Data]? = nil, parameters: GenerateParameters) async throws -> String {
        let (output, _, _) = try await runWithUsage(prompt: prompt, images: images, parameters: parameters)
        return output
    }

    // MARK: Private

    private let container: ModelContainer
}

import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import Tokenizers

/// Loads an embedding model container for the given model name.
public func loadEmbeddingModelContainer(modelName: String) async throws -> MLXEmbedders.ModelContainer {
    let config = MLXEmbedders.ModelConfiguration(id: modelName)

    let container: MLXEmbedders.ModelContainer
    do {
        container = try await MLXEmbedders.loadModelContainer(configuration: config)
    }
    catch {
        fputs(
            "SwamaKit.EmbeddingRunner: Error loading embedding model container for '\(modelName)': \(error.localizedDescription)\n",
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

// MARK: - EmbeddingRunner

/// An actor responsible for running embedding model inference.
public actor EmbeddingRunner {
    // MARK: Lifecycle

    public init(container: MLXEmbedders.ModelContainer) {
        self.container = container
        self.isRunning = false
    }

    // MARK: Public

    /// Generates embeddings for the given input texts.
    public func generateEmbeddings(inputs: [String]) async throws -> (
        embeddings: [[Float]], usage: EmbeddingUsage
    ) {
        try await container.perform { (
            model: any EmbeddingModel,
            tokenizer: Tokenizer,
            pooler: Pooling
        ) throws -> (
            embeddings: [[Float]], usage: EmbeddingUsage
        ) in
            var totalTokens = 0

            // Tokenize all inputs
            let tokenizedInputs = inputs.map { input in
                let tokens = tokenizer.encode(text: input, addSpecialTokens: true)
                totalTokens += tokens.count
                return tokens
            }

            // Find the maximum length for padding
            let maxLength = tokenizedInputs.reduce(into: 0) { acc, tokens in
                acc = max(acc, tokens.count)
            }

            // Pad all inputs to the same length
            let padTokenId = tokenizer.eosTokenId ?? 0
            let paddedInputIds = MLX.stacked(
                tokenizedInputs.map { tokens -> MLXArray in
                    let paddingCount = maxLength - tokens.count
                    let paddedTokens = tokens + Array(repeating: padTokenId, count: paddingCount)
                    return MLXArray(paddedTokens)
                }
            )

            // Create attention mask (1 for real tokens, 0 for padding)
            let attentionMask = paddedInputIds .!= MLXArray(padTokenId)

            // Run the model with batched input
            let output = model(
                paddedInputIds,
                positionIds: nil,
                tokenTypeIds: nil,
                attentionMask: attentionMask
            )

            // Pool hidden states according to the model's configured strategy
            let pooledEmbeddings = pooler(
                output,
                mask: attentionMask,
                normalize: true,
                applyLayerNorm: true
            )

            // Convert each embedding to Float array and evaluate
            eval(pooledEmbeddings)

            let allEmbeddings = (0 ..< inputs.count).map { i in
                pooledEmbeddings[i].asArray(Float.self)
            }

            let usage = EmbeddingUsage(
                promptTokens: totalTokens,
                totalTokens: totalTokens
            )

            return (embeddings: allEmbeddings, usage: usage)
        }
    }

    /// Generates a single embedding for the given input text.
    public func generateEmbedding(input: String) async throws -> (embedding: [Float], usage: EmbeddingUsage) {
        // Wait for any ongoing inference to complete
        while isRunning {
            try await Task.sleep(nanoseconds: 10_000_000) // 10ms
        }

        isRunning = true
        defer { isRunning = false }

        let result = try await generateEmbeddings(inputs: [input])
        guard let firstEmbedding = result.embeddings.first else {
            throw EmbeddingError.noEmbeddingGenerated
        }

        return (embedding: firstEmbedding, usage: result.usage)
    }

    // MARK: Private

    private let container: MLXEmbedders.ModelContainer
    private var isRunning: Bool
}

// MARK: - EmbeddingUsage

public struct EmbeddingUsage: Sendable {
    public let promptTokens: Int
    public let totalTokens: Int

    public init(promptTokens: Int, totalTokens: Int) {
        self.promptTokens = promptTokens
        self.totalTokens = totalTokens
    }
}

// MARK: - EmbeddingError

public enum EmbeddingError: Error, LocalizedError {
    case noEmbeddingGenerated
    case invalidInput
    case modelNotSupported
    case modelLoadFailed(String, underlying: Error)
    case tokenizationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .noEmbeddingGenerated:
            "No embedding was generated"
        case .invalidInput:
            "Invalid input provided"
        case .modelNotSupported:
            "Model is not supported for embeddings"
        case let .modelLoadFailed(modelName, underlying):
            "Failed to load embedding model '\(modelName)': \(underlying.localizedDescription)"
        case let .tokenizationFailed(text):
            "Failed to tokenize text: '\(text)'"
        }
    }
}

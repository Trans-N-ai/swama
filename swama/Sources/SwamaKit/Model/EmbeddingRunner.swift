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
            let padTokenId = tokenizer.eosTokenId ?? 0
            let tokenizedInputs = inputs.enumerated().map { index, input in
                let tokens = tokenizer.encode(text: input, addSpecialTokens: true)
                totalTokens += tokens.count
                return (index: index, tokens: tokens)
            }

            let sortedInputs = tokenizedInputs.sorted { $0.tokens.count < $1.tokens.count }
            var allEmbeddings = Array(repeating: [Float](), count: inputs.count)

            var startIndex = 0
            while startIndex < sortedInputs.count {
                let endIndex = min(startIndex + maxBatchSize, sortedInputs.count)
                let batch = Array(sortedInputs[startIndex ..< endIndex])

                let maxLength = batch.reduce(into: 0) { acc, item in
                    acc = max(acc, item.tokens.count)
                }

                let paddedInputIds = MLX.stacked(
                    batch.map { item -> MLXArray in
                        let paddingCount = maxLength - item.tokens.count
                        let paddedTokens = item.tokens + Array(repeating: padTokenId, count: paddingCount)
                        return MLXArray(paddedTokens)
                    }
                )

                // Create attention mask from original token lengths.
                // This avoids treating real tokens as padding when they share the pad token id.
                let attentionMask = MLX.stacked(
                    batch.map { item -> MLXArray in
                        let paddingCount = maxLength - item.tokens.count
                        let mask = Array(repeating: 1, count: item.tokens.count)
                            + Array(repeating: 0, count: paddingCount)
                        return MLXArray(mask)
                    }
                ) .== MLXArray(1)

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

                for (offset, item) in batch.enumerated() {
                    allEmbeddings[item.index] = pooledEmbeddings[offset].asArray(Float.self)
                }

                startIndex = endIndex
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
    private let maxBatchSize = 8
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

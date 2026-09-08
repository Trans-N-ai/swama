import MLX
@testable import MLXEmbedders
import MLXLMCommon
import MLXNN
@testable import SwamaKit
@testable import SwamaRuntime
import Testing

// MARK: - RuntimeParityTests

@Suite("Legacy and package runtime parity", .serialized)
struct RuntimeParityTests {
    @Test func deterministicChatTextToolAndUsageStayEquivalent() async throws {
        let legacyContainer = makeTestContainer(period: 3).container
        let runtimeContainer = makeTestContainer(period: 3).container
        let parameters = GenerateParameters(
            maxTokens: 8,
            maxKVSize: 4096,
            temperature: 0,
            repetitionPenalty: 1.2,
            repetitionContextSize: 32
        )
        let input = UserInput(chat: [
            .system("w0"),
            .user("w1 w2 w3")
        ])

        let legacy = SwamaKit.ModelRunner(
            container: legacyContainer,
            promptCacheStore: SwamaKit.PromptCacheStore()
        )
        let runtime = SwamaRuntime.ModelRunner(
            container: runtimeContainer,
            promptCacheStore: SwamaRuntime.PromptCacheStore()
        )

        let legacyResult = try await legacy.runChat(userInput: input, parameters: parameters)
        let runtimeResult = try await runtime.runChat(userInput: input, parameters: parameters)

        #expect(runtimeResult.output == legacyResult.output)
        #expect(runtimeResult.rawText == legacyResult.rawText)
        #expect(runtimeResult.analysis == legacyResult.analysis)
        #expect(runtimeResult.promptTokens == legacyResult.promptTokens)
        #expect(runtimeResult.toolCalls == legacyResult.toolCalls)
    }

    @Test func deterministicEmbeddingAndUsageStayEquivalent() async throws {
        let tokenizer = WordVocabTokenizer(contentVocabSize: 16)
        let legacyContainer = EmbedderModelContainer(context: .init(
            configuration: .init(id: "runtime-parity-legacy"),
            model: FixedEmbeddingModel(),
            tokenizer: tokenizer,
            pooling: Pooling(strategy: .mean)
        ))
        let runtimeContainer = EmbedderModelContainer(context: .init(
            configuration: .init(id: "runtime-parity-runtime"),
            model: FixedEmbeddingModel(),
            tokenizer: tokenizer,
            pooling: Pooling(strategy: .mean)
        ))
        let inputs = ["w1 w2", "w3"]

        let legacy = SwamaKit.EmbeddingRunner(container: legacyContainer)
        let runtime = SwamaRuntime.EmbeddingRunner(container: runtimeContainer)
        let legacyResult = try await legacy.generateEmbeddings(inputs: inputs)
        let runtimeResult = try await runtime.generateEmbeddings(inputs: inputs)

        #expect(runtimeResult.embeddings == legacyResult.embeddings)
        #expect(runtimeResult.usage.promptTokens == legacyResult.usage.promptTokens)
        #expect(runtimeResult.usage.totalTokens == legacyResult.usage.totalTokens)
    }

    @Test func everyMaintainedAudioAliasIsExcludedWithoutRejectingLanguageAliases() {
        let maintainedAudioIDs =
            Array(SwamaKit.ModelAliasResolver.sttAliases.keys) +
            Array(SwamaKit.ModelAliasResolver.sttAliases.values) +
            Array(SwamaKit.ModelAliasResolver.ttsAliases.keys) +
            Array(SwamaKit.ModelAliasResolver.ttsAliases.values)
        for model in maintainedAudioIDs {
            #expect(
                SwamaRuntime.RuntimeCoreEngine.isUnsupportedAudioModelID(model),
                "audio ID escaped Core: \(model)"
            )
        }

        let maintainedLanguageIDs =
            Array(SwamaKit.ModelAliasResolver.aliases.keys) +
            Array(SwamaKit.ModelAliasResolver.aliases.values)
        for model in maintainedLanguageIDs {
            #expect(
                SwamaRuntime.RuntimeCoreEngine.isUnsupportedAudioModelID(model) == false,
                "language ID was rejected: \(model)"
            )
        }
    }
}

// MARK: - FixedEmbeddingModel

final class FixedEmbeddingModel: MLXNN.Module, EmbeddingModel {
    let vocabularySize = 32
    let poolingStrategy: Pooling.Strategy? = .mean
    let maxPositionEmbeddings: Int? = nil

    func callAsFunction(
        _ inputs: MLXArray,
        positionIds _: MLXArray?,
        tokenTypeIds _: MLXArray?,
        attentionMask _: MLXArray?
    ) -> EmbeddingModelOutput {
        let values = inputs.asType(.float32).expandedDimensions(axis: -1)
        return EmbeddingModelOutput(
            hiddenStates: MLX.concatenated([values, values + 1], axis: -1),
            pooledOutput: nil
        )
    }
}

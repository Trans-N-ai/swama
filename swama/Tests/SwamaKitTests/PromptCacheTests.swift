//
//  PromptCacheTests.swift
//  SwamaKitTests
//
//  Coverage for the prompt/KV-cache feature (claude/specs/prompt-cache/). Two complementary
//  styles are used throughout:
//
//  - Whitebox: `resolvePromptCacheReuse` is called directly against a hand-built
//    `PromptCacheStore` slot, using real `KVCacheSimple`/`RotatingKVCache` instances fed via
//    their real `update(keys:values:)`. This pins down matched-length and miss-reason
//    classification precisely and deterministically.
//  - Blackbox: a small deterministic `LanguageModel` (`WordEchoModel`) is wired into a real
//    `ModelContainer`/`ModelRunner`, so `TokenIterator`, `LLMModel.prepare`'s chunked prefill,
//    and the full `runChat` generation path are genuinely exercised end to end, with no model
//    downloads. The model's logits are a deterministic function of the token history it reads
//    back from the cache it's handed (via `KVCache.update`'s return value, which always covers
//    the full history up to the current offset) -- so if the cache/suffix stitching in
//    `ModelRunner.runChat` is wrong, cached and uncached runs of the same conversation diverge.
//
//  Every blackbox test compares a "cached" run (one `ModelRunner`/`PromptCacheStore` reused
//  across turns) against an "uncached" baseline (a fresh `PromptCacheStore` per turn, so every
//  turn is a `no_slot` miss) built from the *same* container -- any mismatch means the cache
//  path computed something different from a full, correct prefill.

import Foundation
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon
import MLXNN
@testable import SwamaKit
import Testing

// MARK: - WordVocabTokenizer

/// A tiny, fully deterministic tokenizer. Content tokens are written as "w<N>" words (e.g.
/// "w3 w1 w7"); `encode` parses them directly to `N % contentVocabSize`, so tests can construct
/// exact token sequences without going through real BPE. `decode` is the exact inverse (each id
/// maps back to "w<N> "), so text a model generates can be re-encoded losslessly if fed back in
/// as a later turn's context -- as a real client resending conversation history would.
private struct WordVocabTokenizer: MLXLMCommon.Tokenizer {
    static let systemToken = 16
    static let userToken = 17
    static let assistantToken = 18
    static let sepToken = 19

    let contentVocabSize: Int

    var bosToken: String? {
        nil
    }

    var eosToken: String? {
        nil
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        text.split(separator: " ").compactMap { piece -> Int? in
            guard piece.hasPrefix("w"), let value = Int(piece.dropFirst()) else {
                return nil
            }

            return ((value % contentVocabSize) + contentVocabSize) % contentVocabSize
        }
    }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIds.map { id in
            switch id {
            case 0 ..< contentVocabSize:
                "w\(id) "
            case Self.systemToken:
                "<sys> "
            case Self.userToken:
                "<user> "
            case Self.assistantToken:
                "<asst> "
            case Self.sepToken:
                "<sep> "
            default:
                "<unk\(id)> "
            }
        }
        .joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        if token.hasPrefix("w"), let value = Int(token.dropFirst()) {
            return value
        }
        switch token {
        case "<sys>":
            return Self.systemToken
        case "<user>":
            return Self.userToken
        case "<asst>":
            return Self.assistantToken
        case "<sep>":
            return Self.sepToken
        default:
            return nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        decode(tokenIds: [id], skipSpecialTokens: false).trimmingCharacters(in: .whitespaces)
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] {
        var tokens = [Int]()
        for message in messages {
            let role = message["role"] as? String ?? "user"
            switch role {
            case "system":
                tokens.append(Self.systemToken)
            case "assistant":
                tokens.append(Self.assistantToken)
            default:
                tokens.append(Self.userToken)
            }
            if let content = message["content"] as? String {
                tokens.append(contentsOf: encode(text: content, addSpecialTokens: false))
            }
            tokens.append(Self.sepToken)
        }
        return tokens
    }
}

// MARK: - WordVocabUserInputProcessor

/// Mirrors `LLMModelFactory`'s real (but non-public) `LLMUserInputProcessor`: render the chat
/// messages with `DefaultMessageGenerator`, then run them through the tokenizer's chat template.
private struct WordVocabUserInputProcessor: UserInputProcessor {
    let tokenizer: WordVocabTokenizer
    private let messageGenerator = MLXLMCommon.DefaultMessageGenerator()

    func prepare(input: MLXLMCommon.UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext
        )
        return LMInput(tokens: MLXArray(tokens))
    }
}

// MARK: - WordEchoModel

/// A deterministic `LanguageModel`: its "prediction" for the next token is the content token
/// `period` positions back in the *full* history, read back from the cache's own
/// `update(keys:values:)` return value (which always covers everything up to the new offset --
/// exactly like a real attention cache). This creates a period-`period` repeating pattern that
/// repetition penalties have real, visible leverage over, making equivalence a meaningful
/// check rather than a trivial one: the wrong window (e.g. suffix-only instead of full history)
/// changes which token wins.
private final class WordEchoModel: MLXNN.Module, LLMModel, KVCacheDimensionProvider {
    let kvHeads: [Int]
    private let outputVocabSize: Int
    private let contentVocabSize: Int
    private let period: Int
    private let predictedLogit: Float
    private let alternativeLogit: Float

    var loraLayers: [MLXNN.Module] {
        []
    }

    init(
        numLayers: Int,
        outputVocabSize: Int,
        contentVocabSize: Int,
        period: Int = 2,
        predictedLogit: Float = 10,
        alternativeLogit: Float = 8
    ) {
        self.kvHeads = Array(repeating: 1, count: numLayers)
        self.outputVocabSize = outputVocabSize
        self.contentVocabSize = contentVocabSize
        self.period = period
        self.predictedLogit = predictedLogit
        self.alternativeLogit = alternativeLogit
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let sequenceLength = inputs.dim(1)
        let keysIn = inputs.asType(.float32).reshaped([1, 1, sequenceLength, 1])

        var history = [Float]()
        if let cache, cache.isEmpty == false {
            for (index, layer) in cache.enumerated() {
                let (k, _) = layer.update(keys: keysIn, values: keysIn)
                if index == 0 {
                    history = k.reshaped([-1]).asArray(Float32.self)
                }
            }
        }
        else {
            history = keysIn.reshaped([-1]).asArray(Float32.self)
        }

        let historyTokens = history.map { Int($0.rounded()) }
        let predicted = Self.predictNextToken(
            history: historyTokens, period: period, contentVocabSize: contentVocabSize
        )
        let alternative = (predicted + 1) % contentVocabSize

        var row = [Float](repeating: 0, count: outputVocabSize)
        row[predicted] = predictedLogit
        row[alternative] = max(row[alternative], alternativeLogit)

        let flat = Array(repeating: row, count: sequenceLength).flatMap(\.self)
        return MLXArray(flat, [1, sequenceLength, outputVocabSize])
    }

    static func predictNextToken(history: [Int], period: Int, contentVocabSize: Int) -> Int {
        guard history.count >= period else {
            return 0
        }

        let candidate = history[history.count - period]
        return ((candidate % contentVocabSize) + contentVocabSize) % contentVocabSize
    }
}

// MARK: - Test fixtures

/// Space-separated "w<N>" content words, matching `WordVocabTokenizer.encode`.
private func words(_ ids: [Int]) -> String {
    ids.map { "w\($0)" }.joined(separator: " ")
}

/// Builds a fresh, isolated (never-before-used) model container: a unique model name (so
/// `PromptCacheStore.shared` could never leak between tests even if used), a small deterministic
/// vocabulary, and `WordEchoModel` as the `LanguageModel`.
private func makeTestContainer(
    numLayers: Int = 2,
    contentVocabSize: Int = 16,
    period: Int = 2
) -> (container: MLXLMCommon.ModelContainer, tokenizer: WordVocabTokenizer) {
    let tokenizer = WordVocabTokenizer(contentVocabSize: contentVocabSize)
    let model = WordEchoModel(
        numLayers: numLayers,
        outputVocabSize: contentVocabSize + 4,
        contentVocabSize: contentVocabSize,
        period: period
    )
    let configuration = MLXLMCommon.ModelConfiguration(id: "prompt-cache-test-\(UUID().uuidString)")
    let context = MLXLMCommon.ModelContext(
        configuration: configuration,
        model: model,
        processor: WordVocabUserInputProcessor(tokenizer: tokenizer),
        tokenizer: tokenizer
    )
    return (MLXLMCommon.ModelContainer(context: context), tokenizer)
}

/// Feeds `tokens` through fresh `KVCacheSimple` layers in one `update` call each, as a stand-in
/// for "a slot stored after some earlier, already-completed turn". Unbounded (no rotation), so
/// suitable for prefix-matching tests that aren't about rotation.
private func makeSimpleCacheFedWithTokens(_ tokens: [Int], layerCount: Int) -> [KVCache] {
    let cache: [KVCache] = (0 ..< layerCount).map { _ in KVCacheSimple() }
    guard tokens.isEmpty == false else {
        return cache
    }

    let keys = MLXArray(tokens.map(Float.init), [1, 1, tokens.count, 1])
    for layer in cache {
        _ = layer.update(keys: keys, values: keys)
    }
    return cache
}

/// Feeds `tokens` through fresh `RotatingKVCache` layers **one token at a time**, matching real
/// autoregressive decode (`updateInPlace`) rather than a bulk multi-token prefill
/// (`updateConcat`, which has its own, unrelated temporary-growth trimming). With
/// `tokens.count > maxKVSize` this reliably drives the cache past rotation
/// (`offset >= maxKVSize`, `isTrimmable == false`).
private func makeRotatedCache(tokens: [Int], layerCount: Int, maxKVSize: Int, keep: Int = 4) -> [KVCache] {
    let cache: [KVCache] = (0 ..< layerCount).map { _ in RotatingKVCache(maxSize: maxKVSize, keep: keep) }
    for token in tokens {
        let keys = MLXArray([Float(token)], [1, 1, 1, 1])
        for layer in cache {
            _ = layer.update(keys: keys, values: keys)
        }
    }
    return cache
}

/// Runs an uncached baseline: a brand-new `PromptCacheStore` per call means every turn is a
/// `no_slot` miss, so this is equivalent to the feature not existing, while still sharing the
/// exact same container/model as whatever cached run it's compared against.
private func runUncached(
    container: MLXLMCommon.ModelContainer,
    messages: [MLXLMCommon.Chat.Message],
    parameters: GenerateParameters
) async throws -> ModelRunner.ChatRunResult {
    let runner = ModelRunner(container: container, promptCacheStore: PromptCacheStore())
    return try await runner.runChat(
        userInput: MLXLMCommon.UserInput(chat: messages), parameters: parameters
    )
}

// MARK: - PromptCacheReuseClassificationTests

@Suite("Prompt cache: reuse classification")
struct PromptCacheReuseClassificationTests {
    @Test func hitOnAppendMatchesEntirePreviousPrompt() {
        let store = PromptCacheStore()
        let modelName = "model-append"
        let previousTokens = [16, 1, 2, 19, 17, 3, 4, 19]
        let cache = makeSimpleCacheFedWithTokens(previousTokens, layerCount: 2)
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(tokens: previousTokens, cache: cache, kvConfig: PromptCacheKVConfig(maxKVSize: 4096)),
            epoch: store.currentEpoch(modelName: modelName)
        )

        let newTokens = previousTokens + [18, 5, 6, 19, 17, 7, 8, 19]
        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: newTokens, kvConfig: PromptCacheKVConfig(maxKVSize: 4096), store: store
        )

        guard case let .reuse(matchedLength, reusedCache, reason, _) = decision else {
            Issue.record("expected .reuse, got \(decision)")
            return
        }

        #expect(matchedLength == previousTokens.count)
        #expect(reason == nil)
        #expect(reusedCache[0].offset == previousTokens.count)
        #expect(store.checkout(modelName: modelName) == nil, "slot must be consumed by checkout")
    }

    @Test func partialMismatchOnMidHistoryEditStillReusesTheSharedPrefix() {
        let store = PromptCacheStore()
        let modelName = "model-edit"
        let priorTokens = [16, 1, 19, 17, 2, 19, 18, 3, 19]
        let cache = makeSimpleCacheFedWithTokens(priorTokens, layerCount: 2)
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(tokens: priorTokens, cache: cache, kvConfig: PromptCacheKVConfig(maxKVSize: 4096)),
            epoch: store.currentEpoch(modelName: modelName)
        )

        var editedTokens = priorTokens
        editedTokens[4] = 99 // edit deep inside the already-cached "user" content
        editedTokens.append(contentsOf: [17, 5, 19])

        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: editedTokens, kvConfig: PromptCacheKVConfig(maxKVSize: 4096), store: store
        )

        guard case let .reuse(matchedLength, _, reason, _) = decision else {
            Issue.record("expected .reuse (partial), got \(decision)")
            return
        }

        #expect(matchedLength == 4)
        #expect(reason == .partialMismatch)
    }

    @Test func mismatchAtZeroWhenNothingInCommon() {
        let store = PromptCacheStore()
        let modelName = "model-system-change"
        let priorTokens = [16, 1, 19, 17, 2, 19]
        let cache = makeSimpleCacheFedWithTokens(priorTokens, layerCount: 2)
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(tokens: priorTokens, cache: cache, kvConfig: PromptCacheKVConfig(maxKVSize: 4096)),
            epoch: store.currentEpoch(modelName: modelName)
        )

        // No system message this time: the sequence starts with the user-role token instead of
        // the system-role token, so position 0 itself differs.
        let newTokens = [17, 5, 19]
        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: newTokens, kvConfig: PromptCacheKVConfig(maxKVSize: 4096), store: store
        )

        guard case let .miss(reason, _) = decision else {
            Issue.record("expected .miss, got \(decision)")
            return
        }

        #expect(reason == .mismatchAtZero)
    }

    @Test func fullPrefixMatchCapsAtNMinusOneForTheIterator() {
        let store = PromptCacheStore()
        let modelName = "model-resend"
        let previousTokens = [16, 1, 19, 17, 2, 3, 19]
        let cache = makeSimpleCacheFedWithTokens(previousTokens, layerCount: 2)
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(tokens: previousTokens, cache: cache, kvConfig: PromptCacheKVConfig(maxKVSize: 4096)),
            epoch: store.currentEpoch(modelName: modelName)
        )

        // The exact same prompt, resent verbatim.
        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: previousTokens, kvConfig: PromptCacheKVConfig(maxKVSize: 4096), store: store
        )

        guard case let .reuse(matchedLength, reusedCache, reason, _) = decision else {
            Issue.record("expected .reuse, got \(decision)")
            return
        }

        #expect(matchedLength == previousTokens.count - 1)
        #expect(reason == nil)
        #expect(reusedCache[0].offset == previousTokens.count - 1)
    }

    @Test func rotatedCacheRefusesReuseRegardlessOfMatchingPrefix() {
        let store = PromptCacheStore()
        let modelName = "model-rotated"
        let maxKVSize = 6
        let priorTokens = Array(1 ... 10) // fed one at a time, exceeds maxKVSize -> rotates
        let cache = makeRotatedCache(tokens: priorTokens, layerCount: 2, maxKVSize: maxKVSize)
        #expect(promptCacheIsReusable(cache) == false, "setup sanity: cache should have rotated")

        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(
                tokens: priorTokens,
                cache: cache,
                kvConfig: PromptCacheKVConfig(maxKVSize: maxKVSize)
            ),
            epoch: store.currentEpoch(modelName: modelName)
        )

        let newTokens = priorTokens + [11, 12] // would share a full prefix if reuse were attempted
        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: newTokens, kvConfig: PromptCacheKVConfig(maxKVSize: maxKVSize), store: store
        )

        guard case let .miss(reason, _) = decision else {
            Issue.record("expected .miss, got \(decision)")
            return
        }

        #expect(reason == .rotated)
        #expect(store.checkout(modelName: modelName) == nil, "rotated slot must be dropped, not reused")
    }

    @Test func kvConfigChangeIsAMiss() {
        let store = PromptCacheStore()
        let modelName = "model-kv-change"
        let priorTokens = [16, 1, 19, 17, 2, 19]
        let cache = makeSimpleCacheFedWithTokens(priorTokens, layerCount: 1)
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(tokens: priorTokens, cache: cache, kvConfig: PromptCacheKVConfig(maxKVSize: 2048)),
            epoch: store.currentEpoch(modelName: modelName)
        )

        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: priorTokens + [18, 3, 19], kvConfig: PromptCacheKVConfig(maxKVSize: 4096), store: store
        )

        guard case let .miss(reason, _) = decision else {
            Issue.record("expected .miss, got \(decision)")
            return
        }

        #expect(reason == .kvConfigChanged)
    }

    // MARK: KV-config fingerprint (kvBits / kvGroupSize / quantizedKVStart)

    /// Unlike their siblings above, the tests in this section build slots from a bare, never-fed
    /// `KVCacheSimple()` instead of `makeSimpleCacheFedWithTokens`. `resolvePromptCacheReuse`'s
    /// `PromptCacheKVConfig` comparison runs *before* anything touches `KVCache.update(keys:
    /// values:)` (the real MLX tensor op that needs a Metal runtime this environment lacks -- see
    /// the harness's hazard note), so a never-fed cache classifies identically for these purposes
    /// while keeping these specific tests runnable without one.
    @Test func kvBitsChangeIsAMiss() {
        let store = PromptCacheStore()
        let modelName = "model-kvbits-change"
        let priorTokens = [16, 1, 19, 17, 2, 19]
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(
                tokens: priorTokens, cache: [KVCacheSimple()],
                kvConfig: PromptCacheKVConfig(maxKVSize: 4096, kvBits: 8, kvGroupSize: 64, quantizedKVStart: 0)
            ),
            epoch: store.currentEpoch(modelName: modelName)
        )

        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: priorTokens + [18, 3, 19],
            kvConfig: PromptCacheKVConfig(maxKVSize: 4096, kvBits: 4, kvGroupSize: 64, quantizedKVStart: 0),
            store: store
        )

        guard case let .miss(reason, _) = decision else {
            Issue.record("expected .miss, got \(decision)")
            return
        }

        #expect(reason == .kvConfigChanged)
    }

    @Test func kvGroupSizeChangeIsAMiss() {
        let store = PromptCacheStore()
        let modelName = "model-kvgroupsize-change"
        let priorTokens = [16, 1, 19, 17, 2, 19]
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(
                tokens: priorTokens, cache: [KVCacheSimple()],
                kvConfig: PromptCacheKVConfig(maxKVSize: 4096, kvBits: 8, kvGroupSize: 64, quantizedKVStart: 0)
            ),
            epoch: store.currentEpoch(modelName: modelName)
        )

        // kvBits held fixed at 8 on both sides so kvGroupSize is the only field under test.
        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: priorTokens + [18, 3, 19],
            kvConfig: PromptCacheKVConfig(maxKVSize: 4096, kvBits: 8, kvGroupSize: 32, quantizedKVStart: 0),
            store: store
        )

        guard case let .miss(reason, _) = decision else {
            Issue.record("expected .miss, got \(decision)")
            return
        }

        #expect(reason == .kvConfigChanged)
    }

    @Test func quantizedKVStartChangeIsAMiss() {
        let store = PromptCacheStore()
        let modelName = "model-quantizedkvstart-change"
        let priorTokens = [16, 1, 19, 17, 2, 19]
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(
                tokens: priorTokens, cache: [KVCacheSimple()],
                kvConfig: PromptCacheKVConfig(maxKVSize: 4096, kvBits: 8, kvGroupSize: 64, quantizedKVStart: 0)
            ),
            epoch: store.currentEpoch(modelName: modelName)
        )

        // kvBits/kvGroupSize held fixed on both sides so quantizedKVStart is the only field under
        // test.
        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: priorTokens + [18, 3, 19],
            kvConfig: PromptCacheKVConfig(maxKVSize: 4096, kvBits: 8, kvGroupSize: 64, quantizedKVStart: 8),
            store: store
        )

        guard case let .miss(reason, _) = decision else {
            Issue.record("expected .miss, got \(decision)")
            return
        }

        #expect(reason == .kvConfigChanged)
    }

    @Test func identicalKVConfigIncludingQuantizationSettingsStillHits() {
        let store = PromptCacheStore()
        let modelName = "model-kvconfig-match"
        let priorTokens = [16, 1, 19, 17, 2, 19]
        let config = PromptCacheKVConfig(maxKVSize: 4096, kvBits: 8, kvGroupSize: 32, quantizedKVStart: 4)
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(tokens: priorTokens, cache: [KVCacheSimple()], kvConfig: config),
            epoch: store.currentEpoch(modelName: modelName)
        )

        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: priorTokens + [18, 3, 19], kvConfig: config, store: store
        )

        guard case .reuse = decision else {
            Issue.record("expected .reuse, got \(decision)")
            return
        }
    }

    @Test func noSlotWhenModelHasNeverBeenSeen() {
        let store = PromptCacheStore()
        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: "never-seen",
            newTokens: [1, 2, 3], kvConfig: PromptCacheKVConfig(maxKVSize: 4096), store: store
        )
        guard case let .miss(reason, _) = decision else {
            Issue.record("expected .miss, got \(decision)")
            return
        }

        #expect(reason == .noSlot)
    }

    @Test func disabledNeverTouchesTheStore() {
        let store = PromptCacheStore()
        let modelName = "model-disabled"
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(
                tokens: [1, 2], cache: makeSimpleCacheFedWithTokens([1, 2], layerCount: 1),
                kvConfig: PromptCacheKVConfig(maxKVSize: 100)
            ),
            epoch: store.currentEpoch(modelName: modelName)
        )

        let decision = resolvePromptCacheReuse(
            enabled: false, hasMediaInput: false, modelName: modelName,
            newTokens: [1, 2, 3], kvConfig: PromptCacheKVConfig(maxKVSize: 100), store: store
        )
        guard case let .miss(reason, _) = decision else {
            Issue.record("expected .miss, got \(decision)")
            return
        }

        #expect(reason == .disabled)
        #expect(store.peek(modelName: modelName) != nil, "disabled must not disturb an existing slot")
    }

    @Test func multimodalNeverTouchesTheStore() {
        let store = PromptCacheStore()
        let modelName = "model-multimodal"
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(
                tokens: [1, 2], cache: makeSimpleCacheFedWithTokens([1, 2], layerCount: 1),
                kvConfig: PromptCacheKVConfig(maxKVSize: 100)
            ),
            epoch: store.currentEpoch(modelName: modelName)
        )

        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: true, modelName: modelName,
            newTokens: [1, 2, 3], kvConfig: PromptCacheKVConfig(maxKVSize: 100), store: store
        )
        guard case let .miss(reason, _) = decision else {
            Issue.record("expected .miss, got \(decision)")
            return
        }

        #expect(reason == .multimodal)
        #expect(store.peek(modelName: modelName) != nil, "multimodal must not disturb an existing slot")
    }
}

// MARK: - PromptCacheStoreTests

@Suite("Prompt cache: store lifecycle")
struct PromptCacheStoreTests {
    @Test func dropRemovesOneSlotAndDropAllClearsEverything() {
        let store = PromptCacheStore()
        store.checkin(
            modelName: "a",
            slot: PromptCacheSlot(
                tokens: [1, 2, 3],
                cache: [KVCacheSimple()],
                kvConfig: PromptCacheKVConfig(maxKVSize: 100)
            ),
            epoch: store.currentEpoch(modelName: "a")
        )
        store.checkin(
            modelName: "b",
            slot: PromptCacheSlot(
                tokens: [4, 5],
                cache: [KVCacheSimple()],
                kvConfig: PromptCacheKVConfig(maxKVSize: 100)
            ),
            epoch: store.currentEpoch(modelName: "b")
        )

        store.drop(modelName: "a")
        #expect(store.peek(modelName: "a") == nil)
        #expect(store.peek(modelName: "b") != nil)

        store.dropAll()
        #expect(store.peek(modelName: "b") == nil)
    }

    @Test func checkoutIsDestructiveButPeekIsNot() {
        let store = PromptCacheStore()
        store.checkin(
            modelName: "m",
            slot: PromptCacheSlot(
                tokens: [1, 2],
                cache: [KVCacheSimple()],
                kvConfig: PromptCacheKVConfig(maxKVSize: 100)
            ),
            epoch: store.currentEpoch(modelName: "m")
        )

        #expect(store.peek(modelName: "m") != nil)
        #expect(store.peek(modelName: "m") != nil, "peek must not remove the slot")
        #expect(store.checkout(modelName: "m") != nil)
        #expect(store.peek(modelName: "m") == nil, "checkout must remove the slot")
    }

    @Test func dropBumpsThatModelsEpochEvenWithNoLiveSlotAndLeavesOtherModelsAlone() {
        let store = PromptCacheStore()
        let epochBefore = store.currentEpoch(modelName: "never-had-a-slot")

        // `drop` must invalidate a future write-back for this model even though no slot exists
        // to remove right now -- see `PromptCacheStore.modelEpochs`'s "never cleared" note.
        store.drop(modelName: "never-had-a-slot")
        let epochAfter = store.currentEpoch(modelName: "never-had-a-slot")
        #expect(epochAfter != epochBefore, "drop must bump the per-model epoch even with no live slot")

        let untouchedBefore = store.currentEpoch(modelName: "untouched")
        store.drop(modelName: "never-had-a-slot")
        let untouchedAfter = store.currentEpoch(modelName: "untouched")
        #expect(untouchedAfter == untouchedBefore, "dropping one model must not bump another model's epoch")
    }

    @Test func dropAllBumpsTheGlobalEpochForEveryModelIncludingOnesNeverSeen() {
        let store = PromptCacheStore()
        let neverSeenBefore = store.currentEpoch(modelName: "never-seen")
        let seededBefore = store.currentEpoch(modelName: "seeded")

        store.dropAll()

        #expect(
            store.currentEpoch(modelName: "never-seen") != neverSeenBefore,
            "dropAll must invalidate even a model with no slot and no prior per-model drop"
        )
        #expect(store.currentEpoch(modelName: "seeded") != seededBefore)
    }
}

// MARK: - PromptCacheEpochInvalidationTests

/// Covers the callout on `ModelPool.clearCache()`'s `PromptCacheStore.shared.dropAll()` call:
/// `ModelPool.run` suspends while an inference is in flight, so `clearCache()`/`remove(modelName:)`
/// can mutate the store while a still-running `ModelRunner` already holds (or is about to build) a
/// slot for the model being cleared, and its eventual `checkin` must not resurrect what was just
/// invalidated. `ModelPool` itself is never constructed here (see the harness's hazard note) --
/// every test drives `PromptCacheStore` directly, exactly as `ModelPool.clearCache()`/
/// `remove(modelName:)` do via their own `dropAll()`/`drop(modelName:)` calls.
@Suite("Prompt cache: epoch invalidation")
struct PromptCacheEpochInvalidationTests {
    @Test func checkinIsRejectedAfterDropAllEvenWithTheTokenCapturedAtCheckout() {
        let store = PromptCacheStore()
        let modelName = "model-dropall-race"
        let slot = PromptCacheSlot(
            tokens: [1, 2, 3],
            cache: [KVCacheSimple()],
            kvConfig: PromptCacheKVConfig(maxKVSize: 100)
        )
        store.checkin(modelName: modelName, slot: slot, epoch: store.currentEpoch(modelName: modelName))

        guard let (checkedOutSlot, epoch) = store.checkout(modelName: modelName) else {
            Issue.record("expected a slot to check out")
            return
        }

        // Simulates `ModelPool.run` being suspended awaiting an in-flight inference's operation
        // while `clearCache()` runs on the actor and calls `dropAll()`.
        store.dropAll()

        let accepted = store.checkin(modelName: modelName, slot: checkedOutSlot, epoch: epoch)
        #expect(accepted == false, "checkin must be rejected once the global epoch has moved on")
        #expect(store.peek(modelName: modelName) == nil, "a stale check-in must not resurrect a slot")
    }

    @Test func checkinIsRejectedAfterDropOfThatModelButNotAfterDroppingAnotherModel() {
        let store = PromptCacheStore()
        let droppedModel = "model-dropped"
        let otherModel = "model-untouched"
        let droppedSlot = PromptCacheSlot(
            tokens: [1, 2],
            cache: [KVCacheSimple()],
            kvConfig: PromptCacheKVConfig(maxKVSize: 100)
        )
        let otherSlot = PromptCacheSlot(
            tokens: [3, 4],
            cache: [KVCacheSimple()],
            kvConfig: PromptCacheKVConfig(maxKVSize: 100)
        )
        store.checkin(modelName: droppedModel, slot: droppedSlot, epoch: store.currentEpoch(modelName: droppedModel))
        store.checkin(modelName: otherModel, slot: otherSlot, epoch: store.currentEpoch(modelName: otherModel))

        guard let (checkedOutDropped, droppedEpoch) = store.checkout(modelName: droppedModel) else {
            Issue.record("expected \(droppedModel)'s slot to check out")
            return
        }
        guard let (checkedOutOther, otherEpoch) = store.checkout(modelName: otherModel) else {
            Issue.record("expected \(otherModel)'s slot to check out")
            return
        }

        // Simulates `ModelPool.remove(modelName:)` racing with just `droppedModel`'s in-flight
        // inference -- `otherModel`'s concurrently in-flight inference must be unaffected.
        store.drop(modelName: droppedModel)

        let droppedAccepted = store.checkin(modelName: droppedModel, slot: checkedOutDropped, epoch: droppedEpoch)
        #expect(droppedAccepted == false, "checkin for the dropped model must be rejected")
        #expect(store.peek(modelName: droppedModel) == nil)

        // Control: an unrelated model's checkin must still succeed -- proves the rejection above
        // is targeted at `droppedModel`'s epoch, not some global side effect of calling `drop`.
        let otherAccepted = store.checkin(modelName: otherModel, slot: checkedOutOther, epoch: otherEpoch)
        #expect(otherAccepted, "checkin for an untouched model must still succeed")
        #expect(store.peek(modelName: otherModel) != nil)
    }

    @Test func currentEpochCapturedOnTheMissPathIsInvalidatedByADropAllBeforeCheckin() {
        let store = PromptCacheStore()
        let modelName = "model-never-seen-yet"

        // Mirrors `resolvePromptCacheReuse`'s `.noSlot` branch: no slot exists yet, so the caller
        // captures `currentEpoch` up front, intending to write a freshly-built cache back at the
        // end of generation.
        let epoch = store.currentEpoch(modelName: modelName)

        store.dropAll()

        let freshSlot = PromptCacheSlot(
            tokens: [1, 2, 3],
            cache: [KVCacheSimple()],
            kvConfig: PromptCacheKVConfig(maxKVSize: 100)
        )
        let accepted = store.checkin(modelName: modelName, slot: freshSlot, epoch: epoch)
        #expect(accepted == false, "a global epoch bump before checkin must reject even a miss-path write-back")
        #expect(store.peek(modelName: modelName) == nil)
    }

    @Test func ordinaryCheckoutCheckinRoundTripWithNoInterveningDropStillStores() {
        // Guards against the epoch check rejecting everything -- that failure mode would look
        // like a passing suite sitting on top of a cache that never actually stores anything.
        let store = PromptCacheStore()
        let modelName = "model-clean-path"
        let slot = PromptCacheSlot(
            tokens: [1, 2],
            cache: [KVCacheSimple()],
            kvConfig: PromptCacheKVConfig(maxKVSize: 100)
        )
        store.checkin(modelName: modelName, slot: slot, epoch: store.currentEpoch(modelName: modelName))

        guard let (checkedOut, epoch) = store.checkout(modelName: modelName) else {
            Issue.record("expected a slot to check out")
            return
        }

        let accepted = store.checkin(modelName: modelName, slot: checkedOut, epoch: epoch)
        #expect(accepted, "an ordinary checkout/checkin round trip with no intervening drop must still store")
        #expect(store.peek(modelName: modelName) != nil)
    }
}

// MARK: - PromptCacheIteratorStrategyTests

/// `promptCacheIteratorStrategy` is a pure function over three plain values (no `KVCache`,
/// `MLXArray`, or `TokenIterator` involved), so these tests exercise the initializer-selection
/// decision directly rather than trying to assert on a constructed `TokenIterator` (whose
/// `kvBits`/`kvGroupSize`/`quantizedKVStart` are internal `let`s SwamaKit cannot read back out).
/// Also useful in this sandbox specifically: unlike almost everything else in this file, none of
/// these tests can ever hit the missing-metallib crash, since nothing here touches MLX.
@Suite("Prompt cache: iterator strategy")
struct PromptCacheIteratorStrategyTests {
    @Test func missPathAlwaysUsesParametersRegardlessOfProcessorOrKVBits() {
        // needsFullHistoryWrapper: false covers every cacheable-miss path -- `iterInput` already
        // is the full prompt there, so there is never anything to wrap, independent of whether a
        // penalty processor or KV quantization is configured.
        #expect(
            promptCacheIteratorStrategy(needsFullHistoryWrapper: false, hasPenaltyProcessor: false, kvBits: nil)
                == .parameters
        )
        #expect(
            promptCacheIteratorStrategy(needsFullHistoryWrapper: false, hasPenaltyProcessor: true, kvBits: nil)
                == .parameters
        )
        #expect(
            promptCacheIteratorStrategy(needsFullHistoryWrapper: false, hasPenaltyProcessor: true, kvBits: 4)
                == .parameters
        )
    }

    @Test func cacheHitWithNoPenaltyProcessorUsesParameters() {
        // A cache hit with nothing to wrap: no reason to give up the `parameters:` initializer's
        // full KV-config fidelity just because this is a suffix-only prefill.
        #expect(
            promptCacheIteratorStrategy(needsFullHistoryWrapper: true, hasPenaltyProcessor: false, kvBits: nil)
                == .parameters
        )
        #expect(
            promptCacheIteratorStrategy(needsFullHistoryWrapper: true, hasPenaltyProcessor: false, kvBits: 8)
                == .parameters
        )
    }

    @Test func cacheHitWithPenaltyProcessorAndNoKVQuantizationUsesProcessorWrapper() {
        #expect(
            promptCacheIteratorStrategy(needsFullHistoryWrapper: true, hasPenaltyProcessor: true, kvBits: nil)
                == .processorWrapper
        )
    }

    @Test func cacheHitWithPenaltyProcessorAndKVQuantizationBypasses() {
        // The one combination neither upstream initializer can serve correctly: a full-history
        // wrapper is needed (drops to the processor:sampler: overload) but the caller also wants
        // KV quantization (which that overload's hardcoded kvBits: nil would silently disable).
        #expect(
            promptCacheIteratorStrategy(needsFullHistoryWrapper: true, hasPenaltyProcessor: true, kvBits: 8)
                == .bypass
        )
        #expect(
            promptCacheIteratorStrategy(needsFullHistoryWrapper: true, hasPenaltyProcessor: true, kvBits: 0)
                == .bypass,
            "kvBits: 0 is still non-nil -- a caller asking for 0-bit quantization must not be treated as unset"
        )
    }
}

// MARK: - PromptCacheEndToEndTests

@Suite("Prompt cache: end to end")
struct PromptCacheEndToEndTests {
    @Test func multiTurnEquivalenceWithRepetitionPenalty() async throws {
        let (container, _) = makeTestContainer(period: 5)
        // Small per-turn generations keep each turn's *suffix* (previous reply + new user
        // message -- the only tokens a cache-hit turn actually prefills) short, while
        // `repetitionContextSize` is set far larger than any single suffix but
        // comfortably within the full conversation length. That makes this a real regression
        // check rather than a vacuous one: a processor primed with only the suffix (the bug this
        // wrapper exists to prevent) sees a materially smaller window than one primed with the
        // full history, which changes which tokens get penalized enough to change the argmax --
        // confirmed by deliberately breaking `FullHistoryLogitProcessor.prompt` locally and
        // re-running this test, which fails as expected.
        let parameters = GenerateParameters(
            maxTokens: 3, maxKVSize: 4096, temperature: 0,
            repetitionPenalty: 1.3, repetitionContextSize: 40
        )

        let userTurns = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]] // >= 3 turns

        let cachedRunner = ModelRunner(container: container, promptCacheStore: PromptCacheStore())
        var cachedMessages: [MLXLMCommon.Chat.Message] = [.system(words([0]))]
        var cachedOutputs = [String]()
        for turnWords in userTurns {
            cachedMessages.append(.user(words(turnWords)))
            let result = try await cachedRunner.runChat(
                userInput: .init(chat: cachedMessages), parameters: parameters
            )
            cachedOutputs.append(result.output)
            cachedMessages.append(.assistant(result.output))
        }

        var uncachedMessages: [MLXLMCommon.Chat.Message] = [.system(words([0]))]
        var uncachedOutputs = [String]()
        for turnWords in userTurns {
            uncachedMessages.append(.user(words(turnWords)))
            let result = try await runUncached(
                container: container, messages: uncachedMessages, parameters: parameters
            )
            uncachedOutputs.append(result.output)
            uncachedMessages.append(.assistant(result.output))
        }

        #expect(cachedOutputs.count == userTurns.count)
        #expect(cachedOutputs.allSatisfy { $0.isEmpty == false })
        #expect(cachedOutputs == uncachedOutputs, "cached and uncached generation must be tokenwise identical")
    }

    @Test func cacheHitAfterAppendingOneMessageReusesTheWholePreviousPrompt() async throws {
        let (container, _) = makeTestContainer()
        let modelName = await container.configuration.name
        let parameters = GenerateParameters(maxTokens: 6, maxKVSize: 4096, temperature: 0)
        let store = PromptCacheStore()
        let runner = ModelRunner(container: container, promptCacheStore: store)

        var messages: [MLXLMCommon.Chat.Message] = [.system(words([0])), .user(words([1, 2]))]
        let turn1 = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)

        let slotAfterTurn1 = try #require(
            store.peek(modelName: modelName),
            "expected a slot to be stored after a clean turn"
        )

        let turn1PromptTokenCount = slotAfterTurn1.tokens.count

        messages.append(.assistant(turn1.output))
        messages.append(.user(words([3, 4])))
        let turn2 = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)

        let baseline = try await runUncached(container: container, messages: messages, parameters: parameters)
        #expect(turn2.output == baseline.output)

        let slotAfterTurn2 = try #require(store.peek(modelName: modelName), "expected a slot to be stored after turn 2")

        // The stored slot always holds exactly the prompt tokens for the last clean turn (no
        // generated tokens); turn 2's prompt is turn 1's prompt plus the appended messages, so
        // its stored length must be strictly greater than turn 1's.
        #expect(slotAfterTurn2.tokens.count > turn1PromptTokenCount)
        #expect(Array(slotAfterTurn2.tokens.prefix(turn1PromptTokenCount)) == slotAfterTurn1.tokens)
    }

    @Test func missOnMidHistoryEditStillProducesCorrectOutput() async throws {
        let (container, _) = makeTestContainer()
        let parameters = GenerateParameters(maxTokens: 5, maxKVSize: 4096, temperature: 0)
        let store = PromptCacheStore()
        let runner = ModelRunner(container: container, promptCacheStore: store)

        _ = try await runner.runChat(
            userInput: .init(chat: [.system(words([0])), .user(words([1, 2]))]), parameters: parameters
        )

        // Same system message (shared prefix), but the user content changed -- a mid-history
        // edit relative to what's cached.
        let editedMessages: [MLXLMCommon.Chat.Message] = [.system(words([0])), .user(words([9, 9, 9]))]
        let edited = try await runner.runChat(userInput: .init(chat: editedMessages), parameters: parameters)
        let baseline = try await runUncached(container: container, messages: editedMessages, parameters: parameters)

        #expect(edited.output == baseline.output)
    }

    @Test func missOnSystemPromptChangeStillProducesCorrectOutput() async throws {
        let (container, _) = makeTestContainer()
        let parameters = GenerateParameters(maxTokens: 5, maxKVSize: 4096, temperature: 0)
        let store = PromptCacheStore()
        let runner = ModelRunner(container: container, promptCacheStore: store)

        _ = try await runner.runChat(
            userInput: .init(chat: [.system(words([0])), .user(words([1]))]), parameters: parameters
        )

        // No system message this time -- the sequence starts with a different role token, a
        // mismatch at position 0.
        let newMessages: [MLXLMCommon.Chat.Message] = [.user(words([2]))]
        let missResult = try await runner.runChat(userInput: .init(chat: newMessages), parameters: parameters)
        let baseline = try await runUncached(container: container, messages: newMessages, parameters: parameters)

        #expect(missResult.output == baseline.output)
    }

    @Test func fullPrefixMatchTrimsToNMinusOneAndStillProducesCorrectOutput() async throws {
        let (container, _) = makeTestContainer()
        let parameters = GenerateParameters(maxTokens: 5, maxKVSize: 4096, temperature: 0)
        let store = PromptCacheStore()
        let runner = ModelRunner(container: container, promptCacheStore: store)

        let messages: [MLXLMCommon.Chat.Message] = [.system(words([0])), .user(words([1, 2, 3]))]
        let turn1 = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)

        // Resend the exact same prompt verbatim: matched length == the entire cached prefix ==
        // the entire new prompt, which the iterator-needs->=1-token cap must trim to N-1.
        let turn2 = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)

        let baseline = try await runUncached(container: container, messages: messages, parameters: parameters)
        #expect(turn1.output == baseline.output)
        #expect(turn2.output == baseline.output)
    }

    @Test func rotationFallbackProducesCorrectOutput() async throws {
        let (container, tokenizer) = makeTestContainer()
        let modelName = await container.configuration.name
        let maxKVSize = 8
        let store = PromptCacheStore()

        let priorMessages: [MLXLMCommon.Chat.Message] = [.system(words([0])), .user(words([1, 2, 3, 4, 5]))]
        let priorRawMessages = priorMessages.map { ["role": $0.role.rawValue, "content": $0.content] }
        let priorTokens = try tokenizer.applyChatTemplate(
            messages: priorRawMessages, tools: nil, additionalContext: nil
        )

        // Simulate "a slot was stored earlier and has since rotated past maxKVSize" directly,
        // the same way `PromptCacheReuseClassificationTests.rotatedCacheRefusesReuseRegardlessOfMatchingPrefix`
        // does, but wired into a real end-to-end run this time.
        let cache = makeRotatedCache(tokens: priorTokens, layerCount: 2, maxKVSize: maxKVSize)
        #expect(promptCacheIsReusable(cache) == false, "setup sanity: cache should have rotated")
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(
                tokens: priorTokens,
                cache: cache,
                kvConfig: PromptCacheKVConfig(maxKVSize: maxKVSize)
            ),
            epoch: store.currentEpoch(modelName: modelName)
        )

        let runner = ModelRunner(container: container, promptCacheStore: store)
        var messages = priorMessages
        messages.append(.user(words([6, 7])))
        let parameters = GenerateParameters(maxTokens: 5, maxKVSize: maxKVSize, temperature: 0)

        let result = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)
        let baseline = try await runUncached(container: container, messages: messages, parameters: parameters)

        #expect(result.output == baseline.output)
    }

    /// Covers the callout on `ModelRunner.swift`'s `buildPromptCacheGenerationRun`: a cache hit
    /// whose `GenerateParameters` configure both a penalty processor (forcing the
    /// `FullHistoryLogitProcessor` wrapper) *and* KV quantization (`kvBits != nil`) must not
    /// silently reuse the slot through the wrapper's hardcoded `kvBits: nil` -- see
    /// `PromptCacheIteratorStrategy.bypass`.
    ///
    /// Unlike this suite's other tests, this one cannot be distinguished from the pre-fix
    /// behavior by a compile error: it only calls upstream's own, pre-existing
    /// `GenerateParameters(kvBits:)`, not any new SwamaKit symbol. The only way to observe the
    /// difference is behaviorally -- turn 2 must decline the cache hit `resolvePromptCacheReuse`
    /// found and store nothing back, where the pre-fix code would have silently reused it via the
    /// processor:sampler: overload (dropping kvBits) and stored the result. That requires
    /// actually running this test, which this sandbox cannot do for any test that reaches a real
    /// MLXArray tensor op (see the harness's hazard note) -- turn 1's prefill alone is already
    /// enough to trigger `maybeQuantizeKVCache`'s real quantization math once `kvBits` is
    /// non-nil, on top of the pre-existing missing-metallib limitation every other blackbox test
    /// in this file already has.
    @Test func kvBitsWithPenaltyProcessorOnCacheHitBypassesReuseAndProducesCorrectOutput() async throws {
        let (container, _) = makeTestContainer(period: 5)
        let modelName = await container.configuration.name
        let store = PromptCacheStore()
        let runner = ModelRunner(container: container, promptCacheStore: store)
        let parameters = GenerateParameters(
            maxTokens: 4, maxKVSize: 4096, kvBits: 8, temperature: 0,
            repetitionPenalty: 1.3, repetitionContextSize: 40
        )

        var messages: [MLXLMCommon.Chat.Message] = [.system(words([0])), .user(words([1, 2]))]
        let turn1 = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)
        #expect(store.peek(modelName: modelName) != nil, "setup sanity: turn 1 must have primed a slot")

        messages.append(.assistant(turn1.output))
        messages.append(.user(words([3, 4, 5])))

        // Turn 2 shares turn 1's exact kvConfig and appends onto its token sequence, so
        // `resolvePromptCacheReuse` would ordinarily report `.reuse` here -- the bypass must
        // intercept that decision before any `TokenIterator` gets built for it.
        let turn2 = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)
        let baseline = try await runUncached(container: container, messages: messages, parameters: parameters)

        #expect(turn2.output == baseline.output)
        #expect(
            store.peek(modelName: modelName) == nil,
            """
            a bypassed cache hit must neither resurrect the slot resolvePromptCacheReuse already \
            checked out nor store a new one -- run.cache is nil on the .kvQuantizationUnsupported path
            """
        )
    }

    @Test func cancelMidGenerationThenNextRequestIsCleanAndCorrect() async throws {
        let (container, _) = makeTestContainer()
        let modelName = await container.configuration.name
        let store = PromptCacheStore()
        let runner = ModelRunner(container: container, promptCacheStore: store)
        let parameters = GenerateParameters(maxTokens: 40, maxKVSize: 4096, temperature: 0)

        /// Rebuilt at each call site (rather than shared via one `let`) so the non-Sendable
        /// `[Chat.Message]` value doesn't get flagged as sent into the cancellable Task below
        /// while still "in use" by the follow-up calls afterward -- each call site owns its own
        /// independent value.
        func buildMessages() -> [MLXLMCommon.Chat.Message] {
            [.system(words([0])), .user(words([1, 2, 3]))]
        }

        let chunkCount = Counter()
        // Run on its own Task (as production does via `runCancellingOnClose`) and cancel *that*
        // task from within the callback -- deterministic, no scheduling race -- rather than
        // cancelling whatever task happens to be running this test function itself, which would
        // also derail the "clean next request" assertions below.
        let cancellableTask = Task<ModelRunner.ChatRunResult, Error> {
            try await runner.runChat(
                userInput: .init(chat: buildMessages()),
                parameters: parameters,
                onToken: { _ in
                    chunkCount.increment()
                    if chunkCount.value == 3 {
                        withUnsafeCurrentTask { $0?.cancel() }
                    }
                }
            )
        }
        // runChat does not throw on cancellation; it returns whatever was accumulated so far.
        _ = try await cancellableTask.value

        #expect(store.peek(modelName: modelName) == nil, "an aborted generation must never be stored")

        // The next request on the same model must be a clean run with correct output -- never
        // corrupted state from the cancelled generation.
        let followUp = try await runner.runChat(userInput: .init(chat: buildMessages()), parameters: parameters)
        let baseline = try await runUncached(container: container, messages: buildMessages(), parameters: parameters)
        #expect(followUp.output == baseline.output)
    }

    @Test func dropAllMidGenerationDoesNotResurrectAStaleSlotAndNextRequestIsCorrect() async throws {
        let (container, _) = makeTestContainer()
        let modelName = await container.configuration.name
        let store = PromptCacheStore()
        let runner = ModelRunner(container: container, promptCacheStore: store)
        let parameters = GenerateParameters(maxTokens: 40, maxKVSize: 4096, temperature: 0)

        // Prime a slot with a first, ordinary turn so turn 2 below is a genuine cache *hit*
        // (checkout, not just `currentEpoch` on a miss) -- the exact shape of the callout: an
        // in-flight inference that has already checked a slot out for reuse.
        var messages: [MLXLMCommon.Chat.Message] = [.system(words([0])), .user(words([1, 2]))]
        let turn1 = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)
        #expect(store.peek(modelName: modelName) != nil, "setup sanity: turn 1 must have primed a slot")

        messages.append(.assistant(turn1.output))
        messages.append(.user(words([3, 4, 5])))

        let chunkCount = Counter()
        let turn2 = try await runner.runChat(
            userInput: .init(chat: messages),
            parameters: parameters,
            onToken: { _ in
                chunkCount.increment()
                if chunkCount.value == 3 {
                    // Simulates `ModelPool.run` suspended awaiting turn 2's still-in-flight
                    // operation while `clearCache()` runs on the actor and calls `dropAll()` --
                    // the callout this fix addresses.
                    store.dropAll()
                }
            }
        )

        #expect(turn2.output.isEmpty == false)
        #expect(
            store.peek(modelName: modelName) == nil,
            "turn 2's check-in must not resurrect a slot dropAll() already cleared"
        )

        // The next request must still be a clean, correct run -- never stale/corrupted state
        // left over from the invalidated (but still-completed) turn 2.
        let turn3 = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)
        let baseline = try await runUncached(container: container, messages: messages, parameters: parameters)
        #expect(turn3.output == baseline.output)
    }

    @Test func dropModelNameMidGenerationDoesNotResurrectAStaleSlotAndNextRequestIsCorrect() async throws {
        let (container, _) = makeTestContainer()
        let modelName = await container.configuration.name
        let store = PromptCacheStore()
        let runner = ModelRunner(container: container, promptCacheStore: store)
        let parameters = GenerateParameters(maxTokens: 40, maxKVSize: 4096, temperature: 0)

        var messages: [MLXLMCommon.Chat.Message] = [.system(words([0])), .user(words([1, 2]))]
        let turn1 = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)
        #expect(store.peek(modelName: modelName) != nil, "setup sanity: turn 1 must have primed a slot")

        messages.append(.assistant(turn1.output))
        messages.append(.user(words([3, 4, 5])))

        let chunkCount = Counter()
        let turn2 = try await runner.runChat(
            userInput: .init(chat: messages),
            parameters: parameters,
            onToken: { _ in
                chunkCount.increment()
                if chunkCount.value == 3 {
                    // Sibling of the `dropAll` test above, but for the targeted
                    // `ModelPool.remove(modelName:)` path via `drop(modelName:)`.
                    store.drop(modelName: modelName)
                }
            }
        )

        #expect(turn2.output.isEmpty == false)
        #expect(
            store.peek(modelName: modelName) == nil,
            "turn 2's check-in must not resurrect a slot drop(modelName:) already cleared"
        )

        let turn3 = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)
        let baseline = try await runUncached(container: container, messages: messages, parameters: parameters)
        #expect(turn3.output == baseline.output)
    }

    @Test func abortViaThrownErrorDuringStreamingNeverStoresAPartialSlot() async throws {
        struct SimulatedDisconnect: Error {}

        let (container, _) = makeTestContainer()
        let modelName = await container.configuration.name
        let store = PromptCacheStore()
        let runner = ModelRunner(container: container, promptCacheStore: store)
        let parameters = GenerateParameters(maxTokens: 40, maxKVSize: 4096, temperature: 0)
        let messages: [MLXLMCommon.Chat.Message] = [.system(words([0])), .user(words([1, 2, 3]))]

        let chunkCount = Counter()
        do {
            _ = try await runner.runChat(
                userInput: .init(chat: messages),
                parameters: parameters,
                onToken: { _ in
                    chunkCount.increment()
                    if chunkCount.value == 3 {
                        throw SimulatedDisconnect()
                    }
                }
            )
            Issue.record("expected runChat to rethrow the onToken failure")
        }
        catch is SimulatedDisconnect {
            // expected
        }

        #expect(store.peek(modelName: modelName) == nil, "an aborted generation must never be stored")

        let followUp = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)
        let baseline = try await runUncached(container: container, messages: messages, parameters: parameters)
        #expect(followUp.output == baseline.output)
    }

    @Test func storesOnCleanCompletionOnly() async throws {
        let (container, _) = makeTestContainer()
        let modelName = await container.configuration.name
        let store = PromptCacheStore()
        let runner = ModelRunner(container: container, promptCacheStore: store)
        let parameters = GenerateParameters(maxTokens: 4, maxKVSize: 4096, temperature: 0)

        #expect(store.peek(modelName: modelName) == nil)

        let messages: [MLXLMCommon.Chat.Message] = [.system(words([0])), .user(words([1]))]
        let result = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)

        let slot = try #require(store.peek(modelName: modelName), "expected a slot after a clean completion")

        #expect(slot.tokens.isEmpty == false)
        #expect(slot.kvConfig.maxKVSize == 4096)
        // Precise store-side trim check: the cache must be trimmed back to *exactly* the
        // request's prompt token count (discarding this turn's generated tokens), matching
        // `ChatRunResult.promptTokens` -- not off by one in either direction. `next()` feeds
        // exactly one token into the cache per yielded token, and none are fed beyond the last
        // one yielded, so after `result.completionInfo!.generationTokenCount` tokens were
        // generated, `offset == promptTokens + generationTokenCount`; trimming by
        // `generationTokenCount` must land exactly back on `promptTokens`.
        #expect(slot.tokens.count == result.promptTokens)
        #expect(slot.cache[0].offset == result.promptTokens)
        #expect(slot.cache.allSatisfy { $0.offset == result.promptTokens })
    }
}

// MARK: - Counter

/// A plain, unsynchronized mutable box for test callbacks that are invoked serially (as
/// `runChat`'s `onToken` documents itself to be) but whose closure type is `@Sendable`.
private final class Counter: @unchecked Sendable {
    private(set) var value = 0

    func increment() {
        value += 1
    }
}

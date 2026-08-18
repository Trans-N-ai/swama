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

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

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
            case 0 ..< contentVocabSize: "w\(id) "
            case Self.systemToken: "<sys> "
            case Self.userToken: "<user> "
            case Self.assistantToken: "<asst> "
            case Self.sepToken: "<sep> "
            default: "<unk\(id)> "
            }
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        if token.hasPrefix("w"), let value = Int(token.dropFirst()) {
            return value
        }
        switch token {
        case "<sys>": return Self.systemToken
        case "<user>": return Self.userToken
        case "<asst>": return Self.assistantToken
        case "<sep>": return Self.sepToken
        default: return nil
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
        var tokens: [Int] = []
        for message in messages {
            let role = message["role"] as? String ?? "user"
            switch role {
            case "system": tokens.append(Self.systemToken)
            case "assistant": tokens.append(Self.assistantToken)
            default: tokens.append(Self.userToken)
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

    var loraLayers: [MLXNN.Module] { [] }

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

        var history: [Float] = []
        if let cache, !cache.isEmpty {
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

        let flat = Array(repeating: row, count: sequenceLength).flatMap { $0 }
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
    guard !tokens.isEmpty else {
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

// MARK: - PromptCacheReuseClassificationTests (whitebox)

@Suite("Prompt cache: reuse classification")
struct PromptCacheReuseClassificationTests {
    @Test func hitOnAppendMatchesEntirePreviousPrompt() {
        let store = PromptCacheStore()
        let modelName = "model-append"
        let previousTokens = [16, 1, 2, 19, 17, 3, 4, 19]
        let cache = makeSimpleCacheFedWithTokens(previousTokens, layerCount: 2)
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(tokens: previousTokens, cache: cache, maxKVSize: 4096)
        )

        let newTokens = previousTokens + [18, 5, 6, 19, 17, 7, 8, 19]
        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: newTokens, maxKVSize: 4096, store: store
        )

        guard case let .reuse(matchedLength, reusedCache, reason) = decision else {
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
            slot: PromptCacheSlot(tokens: priorTokens, cache: cache, maxKVSize: 4096)
        )

        var editedTokens = priorTokens
        editedTokens[4] = 99 // edit deep inside the already-cached "user" content
        editedTokens.append(contentsOf: [17, 5, 19])

        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: editedTokens, maxKVSize: 4096, store: store
        )

        guard case let .reuse(matchedLength, _, reason) = decision else {
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
            slot: PromptCacheSlot(tokens: priorTokens, cache: cache, maxKVSize: 4096)
        )

        // No system message this time: the sequence starts with the user-role token instead of
        // the system-role token, so position 0 itself differs.
        let newTokens = [17, 5, 19]
        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: newTokens, maxKVSize: 4096, store: store
        )

        guard case let .miss(reason) = decision else {
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
            slot: PromptCacheSlot(tokens: previousTokens, cache: cache, maxKVSize: 4096)
        )

        // The exact same prompt, resent verbatim.
        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: previousTokens, maxKVSize: 4096, store: store
        )

        guard case let .reuse(matchedLength, reusedCache, reason) = decision else {
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
        #expect(!promptCacheIsReusable(cache), "setup sanity: cache should have rotated")

        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(tokens: priorTokens, cache: cache, maxKVSize: maxKVSize)
        )

        let newTokens = priorTokens + [11, 12] // would share a full prefix if reuse were attempted
        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: newTokens, maxKVSize: maxKVSize, store: store
        )

        guard case let .miss(reason) = decision else {
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
            slot: PromptCacheSlot(tokens: priorTokens, cache: cache, maxKVSize: 2048)
        )

        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: modelName,
            newTokens: priorTokens + [18, 3, 19], maxKVSize: 4096, store: store
        )

        guard case let .miss(reason) = decision else {
            Issue.record("expected .miss, got \(decision)")
            return
        }
        #expect(reason == .kvConfigChanged)
    }

    @Test func noSlotWhenModelHasNeverBeenSeen() {
        let store = PromptCacheStore()
        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: false, modelName: "never-seen",
            newTokens: [1, 2, 3], maxKVSize: 4096, store: store
        )
        guard case let .miss(reason) = decision else {
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
                tokens: [1, 2], cache: makeSimpleCacheFedWithTokens([1, 2], layerCount: 1), maxKVSize: 100
            )
        )

        let decision = resolvePromptCacheReuse(
            enabled: false, hasMediaInput: false, modelName: modelName,
            newTokens: [1, 2, 3], maxKVSize: 100, store: store
        )
        guard case let .miss(reason) = decision else {
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
                tokens: [1, 2], cache: makeSimpleCacheFedWithTokens([1, 2], layerCount: 1), maxKVSize: 100
            )
        )

        let decision = resolvePromptCacheReuse(
            enabled: true, hasMediaInput: true, modelName: modelName,
            newTokens: [1, 2, 3], maxKVSize: 100, store: store
        )
        guard case let .miss(reason) = decision else {
            Issue.record("expected .miss, got \(decision)")
            return
        }
        #expect(reason == .multimodal)
        #expect(store.peek(modelName: modelName) != nil, "multimodal must not disturb an existing slot")
    }
}

// MARK: - PromptCacheStoreTests (slot lifecycle, pure)

@Suite("Prompt cache: store lifecycle")
struct PromptCacheStoreTests {
    @Test func dropRemovesOneSlotAndDropAllClearsEverything() {
        let store = PromptCacheStore()
        store.checkin(
            modelName: "a",
            slot: PromptCacheSlot(tokens: [1, 2, 3], cache: [KVCacheSimple()], maxKVSize: 100)
        )
        store.checkin(
            modelName: "b",
            slot: PromptCacheSlot(tokens: [4, 5], cache: [KVCacheSimple()], maxKVSize: 100)
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
            slot: PromptCacheSlot(tokens: [1, 2], cache: [KVCacheSimple()], maxKVSize: 100)
        )

        #expect(store.peek(modelName: "m") != nil)
        #expect(store.peek(modelName: "m") != nil, "peek must not remove the slot")
        #expect(store.checkout(modelName: "m") != nil)
        #expect(store.peek(modelName: "m") == nil, "checkout must remove the slot")
    }
}

// MARK: - PromptCacheEndToEndTests (blackbox: real ModelRunner.runChat round trips)

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
        var cachedOutputs: [String] = []
        for turnWords in userTurns {
            cachedMessages.append(.user(words(turnWords)))
            let result = try await cachedRunner.runChat(
                userInput: .init(chat: cachedMessages), parameters: parameters
            )
            cachedOutputs.append(result.output)
            cachedMessages.append(.assistant(result.output))
        }

        var uncachedMessages: [MLXLMCommon.Chat.Message] = [.system(words([0]))]
        var uncachedOutputs: [String] = []
        for turnWords in userTurns {
            uncachedMessages.append(.user(words(turnWords)))
            let result = try await runUncached(
                container: container, messages: uncachedMessages, parameters: parameters
            )
            uncachedOutputs.append(result.output)
            uncachedMessages.append(.assistant(result.output))
        }

        #expect(cachedOutputs.count == userTurns.count)
        #expect(cachedOutputs.allSatisfy { !$0.isEmpty })
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

        guard let slotAfterTurn1 = store.peek(modelName: modelName) else {
            Issue.record("expected a slot to be stored after a clean turn")
            return
        }
        let turn1PromptTokenCount = slotAfterTurn1.tokens.count

        messages.append(.assistant(turn1.output))
        messages.append(.user(words([3, 4])))
        let turn2 = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)

        let baseline = try await runUncached(container: container, messages: messages, parameters: parameters)
        #expect(turn2.output == baseline.output)

        guard let slotAfterTurn2 = store.peek(modelName: modelName) else {
            Issue.record("expected a slot to be stored after turn 2")
            return
        }
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
        #expect(!promptCacheIsReusable(cache), "setup sanity: cache should have rotated")
        store.checkin(
            modelName: modelName,
            slot: PromptCacheSlot(tokens: priorTokens, cache: cache, maxKVSize: maxKVSize)
        )

        let runner = ModelRunner(container: container, promptCacheStore: store)
        var messages = priorMessages
        messages.append(.user(words([6, 7])))
        let parameters = GenerateParameters(maxTokens: 5, maxKVSize: maxKVSize, temperature: 0)

        let result = try await runner.runChat(userInput: .init(chat: messages), parameters: parameters)
        let baseline = try await runUncached(container: container, messages: messages, parameters: parameters)

        #expect(result.output == baseline.output)
    }

    @Test func cancelMidGenerationThenNextRequestIsCleanAndCorrect() async throws {
        let (container, _) = makeTestContainer()
        let modelName = await container.configuration.name
        let store = PromptCacheStore()
        let runner = ModelRunner(container: container, promptCacheStore: store)
        let parameters = GenerateParameters(maxTokens: 40, maxKVSize: 4096, temperature: 0)

        // Rebuilt at each call site (rather than shared via one `let`) so the non-Sendable
        // `[Chat.Message]` value doesn't get flagged as sent into the cancellable Task below
        // while still "in use" by the follow-up calls afterward -- each call site owns its own
        // independent value.
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

        guard let slot = store.peek(modelName: modelName) else {
            Issue.record("expected a slot after a clean completion")
            return
        }
        #expect(slot.tokens.count > 0)
        #expect(slot.maxKVSize == 4096)
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

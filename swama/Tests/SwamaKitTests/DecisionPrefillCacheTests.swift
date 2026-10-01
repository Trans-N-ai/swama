import MLXLMCommon
@testable import SwamaKit
import Testing

// MARK: - DecisionPrefillCacheTests

@Suite("Decision prefix plan")
struct DecisionPrefillCacheTests {
    @Test func warmRangesEqualTheOriginalBalancedRemainder() throws {
        for total in [513, 514, 799, 800, 801, 1024, 1025, 1538, 4095] {
            var cold: [Range<Int>] = []
            let processed = try PrefillParameters().forEachChunk(total: total) { cold.append($0) }
            #expect(processed == total - 1)
            let size = try #require(DecisionPrefillCache.chunkSize(total: total))
            for cut in cold.indices.dropLast() {
                let offset = cold[cut].upperBound
                var warm: [Range<Int>] = []
                _ = try PrefillParameters(stepSize: size, chunking: .remainder)
                    .forEachChunk(total: total - offset) { range in
                        warm.append((range.lowerBound + offset) ..< (range.upperBound + offset))
                    }
                #expect(warm == Array(cold.dropFirst(cut + 1)))
            }
        }
    }

    @Test func shortPromptsHaveNoArtificialPrefillBoundary() {
        for total in [1, 20, 511, 512] {
            #expect(DecisionPrefillCache.chunkSize(total: total) == nil)
        }
    }

    @Test func bytesRequireLiteralTemplateContentAndCompleteUTF8() {
        let content = "中文 instruction and changing suffix"
        let prompt = "<user>" + content + "</user><assistant>"
        #expect(DecisionPrefillCache.prefixBytes(
            content: content,
            prompt: prompt,
            decodedPrefix: "<user>中文 instruction"
        ) != nil)
        #expect(DecisionPrefillCache.prefixBytes(
            content: "different bytes",
            prompt: prompt,
            decodedPrefix: "<user>中文 instruction"
        ) == nil)
        #expect(DecisionPrefillCache.prefixBytes(
            content: content,
            prompt: prompt,
            decodedPrefix: "<user>中\u{fffd}"
        ) == nil)
        #expect(DecisionPrefillCache.prefixBytes(
            content: content,
            prompt: prompt,
            decodedPrefix: "wrong template"
        ) == nil)
    }

    @Test func templateEnvelopeChangesRemainDistinct() {
        let a = DecisionPrefillCache.templateIdentity(content: "input A", prompt: "<user>input A</user>A:")
        let b = DecisionPrefillCache.templateIdentity(content: "input B", prompt: "<user>input B</user>A:")
        let different = DecisionPrefillCache.templateIdentity(content: "input B", prompt: "<user>input B</user>B:")
        #expect(a == b)
        #expect(a != different)
    }
}

// MARK: - DecisionCacheLifetimeTests

@Suite("Decision cache lifetime")
struct DecisionCacheLifetimeTests {
    @Test func transientRunnersShareOnlyTheirOwnContainerState() async {
        let one = makeLifetimeTestContainer()
        let two = makeLifetimeTestContainer()
        let first = ModelRunner(container: one)
        let second = ModelRunner(container: one)
        let other = ModelRunner(container: two)
        let a = await first.decisionPrefillCache
        let b = await second.decisionPrefillCache
        let c = await other.decisionPrefillCache
        #expect(a === b)
        #expect(a !== c)
        DecisionPrefillStore.shared.remove(one)
        DecisionPrefillStore.shared.remove(two)
    }

    @Test func registryCannotKeepModelWeightsAlive() {
        var container: ModelContainer? = makeLifetimeTestContainer()
        weak var witness = container
        let cache = DecisionPrefillStore.shared.cache(for: container!)
        container = nil
        #expect(witness == nil)
        withExtendedLifetime(cache) {}
    }

    @Test func clearingThePoolReleasesIdleDecisionState() async throws {
        let pool = ModelPool(memoryHooks: .init(activeMemory: { 0 }, clearCache: {}))
        let container = makeLifetimeTestContainer()
        try await pool.cacheContainerForTesting(container, modelName: "idle-decision-cache")
        weak var witness: DecisionPrefillCache?
        do {
            let state = DecisionPrefillStore.shared.cache(for: container)
            witness = state
        }
        #expect(witness != nil)
        await pool.clearCache()
        #expect(witness == nil)
        withExtendedLifetime(container) {}
    }

    @Test func teardownCannotBeUndoneByAnInflightContainer() async throws {
        let pool = ModelPool(memoryHooks: .init(activeMemory: { 0 }, clearCache: {}))
        let container = makeLifetimeTestContainer()
        try await pool.cacheContainerForTesting(container, modelName: "decision-cache")
        let first = DecisionPrefillStore.shared.cache(for: container)
        await pool.remove(modelName: "decision-cache")
        let staleOne = DecisionPrefillStore.shared.cache(for: container)
        let staleTwo = DecisionPrefillStore.shared.cache(for: container)
        #expect(first !== staleOne)
        #expect(staleOne !== staleTwo)
    }
}

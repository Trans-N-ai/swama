//
//  DecisionReasoningOpenerCacheTests.swift
//  SwamaRuntimeTests
//

import Foundation
@preconcurrency import MLXLMCommon
@testable import SwamaRuntime
import Testing

// MARK: - DecisionReasoningOpenerCacheTests

/// Reasoning-opener ids are computed once per model container and must never leak to another
/// container (a different model can have a different tokenizer).
@Suite("Decision reasoning-opener cache")
struct DecisionReasoningOpenerCacheTests {
    @Test func openersAreCachedPerContainer() async {
        let cache = DecisionBoundaryCache()
        let first = makeTestContainer().container
        let second = makeTestContainer().container

        #expect(await cache.reasoningOpeners(for: first) == nil)
        await cache.storeReasoningOpeners([11, 12], for: first)
        #expect(await cache.reasoningOpeners(for: first) == [11, 12])
        #expect(await cache.reasoningOpeners(for: second) == nil)

        await cache.storeReasoningOpeners([], for: second)
        #expect(await cache.reasoningOpeners(for: second) == [])
        #expect(await cache.reasoningOpeners(for: first) == [11, 12])
    }

    @Test func aReleasedContainerDoesNotHandItsOpenersToANewOne() async {
        let cache = DecisionBoundaryCache()
        do {
            let gone = makeTestContainer().container
            await cache.storeReasoningOpeners([7], for: gone)
        }
        // Whatever reuses the old address must start empty: the entry is keyed weakly.
        for _ in 0 ..< 8 {
            let fresh = makeTestContainer().container
            #expect(await cache.reasoningOpeners(for: fresh) == nil)
        }
    }
}

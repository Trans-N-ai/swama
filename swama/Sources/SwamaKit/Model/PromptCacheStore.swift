//
//  PromptCacheStore.swift
//  SwamaKit
//
//  In-memory, one-slot-per-model prompt/KV-cache used by `ModelRunner.runChat` to avoid
//  re-prefilling the shared prefix of a multi-turn conversation. See
//  claude/specs/prompt-cache/plan.md ("Design (v1)" and "Phase-1 amendments") for the design
//  this implements.
//

import Foundation
@preconcurrency import MLXLMCommon
import Synchronization

// MARK: - PromptCacheMissReason

/// Why a request could not reuse cached KV state. Logged on every miss (including a *partial*
/// reuse, which still counts as one for diagnostic purposes -- see `PromptCacheResolution`)
/// so a silently-zero-hit cache is diagnosable rather than looking identical to a working one.
public enum PromptCacheMissReason: String, Sendable {
    /// `SWAMA_PROMPT_CACHE=0`.
    case disabled

    /// No slot exists for this model yet (first request, or the previous slot was dropped by
    /// eviction/memory pressure/cancellation). Also covers a model swap, since slots are keyed
    /// by model name.
    case noSlot = "no_slot"

    /// The request carries image/video content; multimodal requests bypass the cache entirely
    /// (both restore and store), since image embeddings are not represented in the token
    /// sequence a KV cache indexes by.
    case multimodal

    /// The new token sequence shares no prefix at all with the cached one (position 0 already
    /// differs) -- typically a system-prompt change or an unrelated conversation.
    case mismatchAtZero = "mismatch_at_0"

    /// The new token sequence shares a non-empty prefix with the cached one, but not the whole
    /// cached prefix -- typically a mid-history edit. The shared prefix is still reused (causal
    /// attention means cache state at position i depends only on tokens[0..<i], so this remains
    /// exact), this reason exists purely to flag that the reuse was partial.
    case partialMismatch = "partial_mismatch"

    /// The cached `KVCache` has rotated (a `RotatingKVCache` at or past `maxSize`) or is
    /// otherwise not trimmable; its layout no longer corresponds to a simple prefix of the
    /// original sequence, so it cannot be reused at all.
    case rotated

    /// The slot exists but was built with a different `maxKVSize` than this request wants.
    case kvConfigChanged = "kv_config_changed"
}

// MARK: - PromptCacheResolution

/// The outcome of trying to reuse a model's prompt cache for one request's token sequence.
///
/// `@unchecked Sendable`: carries `[KVCache]` (see `PromptCacheSlot`); this value is built once
/// per request, handed across a single `container.perform` boundary, and not shared afterward.
enum PromptCacheResolution: @unchecked Sendable {
    /// Reuse the cache starting at `matchedLength` tokens in (already capped so the caller has
    /// at least one suffix token to feed the `TokenIterator`, and already trimmed to that
    /// offset). `reason` is `.partialMismatch` when the match was shorter than the entire
    /// cached prefix, `nil` for a clean full-prefix hit (a plain append).
    case reuse(matchedLength: Int, cache: [KVCache], reason: PromptCacheMissReason?)

    /// No reuse at all; the caller must prefill the full prompt.
    case miss(reason: PromptCacheMissReason)
}

// MARK: - PromptCacheSlot

/// One model's cached prompt/KV state: the exact token sequence the cache represents, the live
/// `KVCache` layers (one per model layer), and the `maxKVSize` the cache was built with (a
/// mismatch there means a different `RotatingKVCache` window, so the cache cannot be reused
/// as-is even if the tokens match).
///
/// `@unchecked Sendable`: `KVCache` is a mutable, non-Sendable MLX type. Safety comes from
/// `PromptCacheStore`'s checkout discipline, not from the type system -- a slot has exactly one
/// owner at a time (see `PromptCacheStore`).
public struct PromptCacheSlot: @unchecked Sendable {
    public var tokens: [Int]
    public var cache: [KVCache]
    public var maxKVSize: Int

    public init(tokens: [Int], cache: [KVCache], maxKVSize: Int) {
        self.tokens = tokens
        self.cache = cache
        self.maxKVSize = maxKVSize
    }
}

// MARK: - PromptCacheStore

/// One prompt/KV-cache slot per model name, with checkout semantics: `checkout` atomically
/// removes and returns a model's slot, so the caller has exclusive ownership of it until it
/// explicitly `checkin`s it. This makes the cancellation-safe lifecycle free -- a caller that
/// hits an error, is cancelled, or decides the resulting cache is unusable (rotated, etc.)
/// simply never calls `checkin`, and the slot stays absent. There is no separate "invalidate"
/// step and no way to leave a half-written slot behind: the slot the caller checked out is
/// already gone from the store the moment `checkout` returns it.
///
/// Access is a simple mutex rather than an actor: `ModelPool.run` already serializes inference
/// per model name, so contention is not expected, but the store itself makes no assumption
/// about that -- `checkout`/`checkin`/`drop` are individually atomic regardless.
public final class PromptCacheStore: Sendable {
    // MARK: Lifecycle

    public init() {}

    // MARK: Public

    public static let shared = PromptCacheStore()

    /// Removes and returns the slot for `modelName`, if any -- see the checkout-semantics note
    /// on the type itself.
    public func checkout(modelName: String) -> PromptCacheSlot? {
        slots.withLock { $0.removeValue(forKey: modelName) }
    }

    /// Non-destructive lookup of a model's slot, for diagnostics and tests -- does not affect
    /// checkout state (unlike `checkout`, this does not remove the slot).
    public func peek(modelName: String) -> PromptCacheSlot? {
        slots.withLock { $0[modelName] }
    }

    /// Stores `slot` for `modelName`, checking it back in after clean completion.
    public func checkin(modelName: String, slot: PromptCacheSlot) {
        slots.withLock { $0[modelName] = slot }
    }

    /// Drops any slot for `modelName` -- used on model eviction and under memory pressure,
    /// where the cache's underlying weights/buffers are going away regardless of the slot's
    /// content, and available for explicit invalidation elsewhere.
    public func drop(modelName: String) {
        slots.withLock { _ = $0.removeValue(forKey: modelName) }
    }

    /// Drops every slot -- used when the whole model cache is cleared.
    public func dropAll() {
        slots.withLock { $0.removeAll() }
    }

    // MARK: Private

    private let slots = Mutex<[String: PromptCacheSlot]>([:])
}

// MARK: - Pure helpers (free functions for direct unit testing via @testable import)

/// Longest common prefix length of two token sequences.
func promptCacheLongestCommonPrefix(_ a: [Int], _ b: [Int]) -> Int {
    var index = 0
    let limit = min(a.count, b.count)
    while index < limit, a[index] == b[index] {
        index += 1
    }
    return index
}

/// Whether a single cache layer can still be trimmed and treated as an exact prefix of the
/// original sequence. `RotatingKVCache.isTrimmable` already encodes `offset < maxSize`, but the
/// `maxSize` check is repeated explicitly here (rather than trusting `isTrimmable` alone) per
/// the design's "assert, don't assume" rotation guard: the failure mode on getting this wrong is
/// corrupt output, not extra latency.
func promptCacheLayerIsReusable(_ layer: KVCache) -> Bool {
    guard layer.isTrimmable else {
        return false
    }
    if let maxSize = layer.maxSize {
        return layer.offset < maxSize
    }
    return true
}

/// Whether every layer of a cache can still be trimmed and reused.
func promptCacheIsReusable(_ cache: [KVCache]) -> Bool {
    !cache.isEmpty && cache.allSatisfy(promptCacheLayerIsReusable)
}

/// Resolves whether `newTokens` can reuse `store`'s slot for `modelName`, checking the slot out
/// of the store in the process whenever one exists and is even considered (see the type docs on
/// `PromptCacheStore` for why not checking a considered-but-rejected slot back in is sufficient
/// to invalidate it).
///
/// `hasMediaInput` and `!enabled` short-circuit before any checkout, so a multimodal or
/// cache-disabled request never disturbs a slot a later, ordinary request could still use.
func resolvePromptCacheReuse(
    enabled: Bool,
    hasMediaInput: Bool,
    modelName: String,
    newTokens: [Int],
    maxKVSize: Int,
    store: PromptCacheStore
) -> PromptCacheResolution {
    guard enabled else {
        return .miss(reason: .disabled)
    }
    guard !hasMediaInput else {
        return .miss(reason: .multimodal)
    }
    guard let slot = store.checkout(modelName: modelName) else {
        return .miss(reason: .noSlot)
    }
    guard slot.maxKVSize == maxKVSize else {
        return .miss(reason: .kvConfigChanged)
    }
    guard promptCacheIsReusable(slot.cache) else {
        return .miss(reason: .rotated)
    }

    let matched = promptCacheLongestCommonPrefix(slot.tokens, newTokens)
    guard matched > 0 else {
        return .miss(reason: .mismatchAtZero)
    }

    // The TokenIterator needs at least one prompt token to prime the pump, so a full-prefix
    // match (matched == newTokens.count) still leaves the last token to be fed as the suffix.
    let trimTarget = min(matched, max(newTokens.count - 1, 0))
    for layer in slot.cache {
        layer.trim(layer.offset - trimTarget)
    }

    let reason: PromptCacheMissReason? = matched < slot.tokens.count ? .partialMismatch : nil
    return .reuse(matchedLength: trimTarget, cache: slot.cache, reason: reason)
}

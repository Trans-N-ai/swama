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

    /// The slot exists but was built with a different `PromptCacheKVConfig` than this request
    /// wants -- `maxKVSize`, `kvBits`, `kvGroupSize`, or `quantizedKVStart` differs.
    case kvConfigChanged = "kv_config_changed"

    /// This request's `PromptCacheEpoch` (captured at `checkout`, or at `currentEpoch` on a
    /// miss) no longer matched the store's current epoch by the time the request tried to
    /// check its slot back in -- `drop(modelName:)` or `dropAll()` ran while this request's
    /// generation was still in flight. The freshly-built or reused-and-trimmed cache this
    /// request produced is discarded rather than resurrecting state the store had already
    /// invalidated (see `PromptCacheEpoch`).
    case invalidated

    /// A cache hit whose reuse would need `FullHistoryLogitProcessor` (a penalty processor is
    /// configured, so the wrapper must re-prime it with the full history) together with a
    /// caller-requested quantized KV cache (`GenerateParameters.kvBits != nil`). The
    /// `TokenIterator` overload that accepts a custom processor hardcodes
    /// `kvBits: nil`/`kvGroupSize: 64`/`quantizedKVStart: 0` and cannot be told otherwise --
    /// those are internal `let`s, set only by the sibling initializer that derives them from
    /// `GenerateParameters`, which does not itself accept a custom processor. Rather than
    /// silently drop the caller's quantization request, `ModelRunner.runChat` declines this
    /// cache hit entirely and falls back to an ordinary, uncached, fully-quantization-correct
    /// prefill for this one request. See `PromptCacheIteratorStrategy.bypass`.
    case kvQuantizationUnsupported = "kv_quantization_unsupported"
}

// MARK: - PromptCacheEpoch

/// A snapshot of `PromptCacheStore`'s invalidation counters, captured up front (at `checkout` on
/// a reuse attempt, or at `currentEpoch` on a miss) and re-checked at `checkin`.
///
/// This closes the race described on `ModelPool.clearCache()`'s call to `PromptCacheStore
/// .shared.dropAll()`: `ModelPool.run` suspends while an inference is in flight, so
/// `clearCache()`/`remove(modelName:)` can run -- and mutate the store -- while a still-running
/// `ModelRunner` already holds (or is about to build) a slot for the very model being cleared.
/// Without this token, that runner's eventual `checkin` would silently resurrect a slot the store
/// had already dropped, undoing the clear/remove and leaking the resurrected slot's KV memory
/// forever. `checkin` only stores when the token presented still equals the store's current one;
/// otherwise the slot the caller built is simply discarded.
///
/// This is a supplement to, not a replacement for, the checkout-is-destructive discipline
/// documented on `PromptCacheStore` itself: checkout still atomically removes a slot so at most
/// one caller ever owns it, and the epoch only guards the additional window where the store was
/// mutated *while* that caller held the slot (or, on a miss, before it wrote a fresh one back).
public struct PromptCacheEpoch: Equatable, Sendable {
    fileprivate let global: UInt64
    fileprivate let perModel: UInt64
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
    /// cached prefix, `nil` for a clean full-prefix hit (a plain append). `epoch` is the token
    /// captured at the `checkout` that produced this slot -- carried by the caller all the way
    /// through generation and presented back to `checkin`.
    case reuse(matchedLength: Int, cache: [KVCache], reason: PromptCacheMissReason?, epoch: PromptCacheEpoch)

    /// No reuse at all; the caller must prefill the full prompt. `epoch` is `nil` for
    /// `.disabled`/`.multimodal`, which never touch the store at all (nothing will ever be
    /// checked in for this request), and otherwise the token from `currentEpoch(modelName:)`
    /// (`.noSlot`) or from the checkout that was attempted and rejected (`.kvConfigChanged`,
    /// `.rotated`, `.mismatchAtZero`) -- in every case, captured before generation starts.
    case miss(reason: PromptCacheMissReason, epoch: PromptCacheEpoch?)
}

// MARK: - PromptCacheKVConfig

/// The KV-cache *layout* a slot's `KVCache` layers were built with -- everything that changes the
/// shape or numeric representation of the cache tensors themselves, as opposed to the token
/// content they represent. A mismatch on any field here means the stored `KVCache` layers are
/// simply the wrong shape/type for this request's iterator to keep appending to, independent of
/// whether the token prefix matches: `maxKVSize` selects `RotatingKVCache` vs `KVCacheSimple` (and
/// its window size), while `kvBits`/`kvGroupSize`/`quantizedKVStart` select whether/when the
/// cache gets transparently swapped for a `QuantizedKVCache` mid-generation (see
/// `maybeQuantizeKVCache`). Compared as a whole struct in `resolvePromptCacheReuse` so a change to
/// any one field -- not just `maxKVSize` -- correctly forces a `.kvConfigChanged` miss rather than
/// reusing a cache whose layout no longer matches what this request's `GenerateParameters` asked
/// for.
public struct PromptCacheKVConfig: Equatable, Sendable {
    public var maxKVSize: Int
    public var kvBits: Int?
    public var kvGroupSize: Int
    public var quantizedKVStart: Int

    public init(maxKVSize: Int, kvBits: Int? = nil, kvGroupSize: Int = 64, quantizedKVStart: Int = 0) {
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
    }

    /// Builds a config from `GenerateParameters`, with `maxKVSize` supplied separately since
    /// `ModelRunner.runChat` computes the *effective* value (falling back to the context limit
    /// when the caller left `parameters.maxKVSize` unset) rather than trusting
    /// `parameters.maxKVSize` directly.
    public init(parameters: GenerateParameters, maxKVSize: Int) {
        self.init(
            maxKVSize: maxKVSize,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart
        )
    }
}

// MARK: - PromptCacheSlot

/// One model's cached prompt/KV state: the exact token sequence the cache represents, the live
/// `KVCache` layers (one per model layer), and the `PromptCacheKVConfig` the cache was built with
/// (a mismatch there means the stored layers are the wrong shape/layout, so the cache cannot be
/// reused as-is even if the tokens match).
///
/// `@unchecked Sendable`: `KVCache` is a mutable, non-Sendable MLX type. Safety comes from
/// `PromptCacheStore`'s checkout discipline, not from the type system -- a slot has exactly one
/// owner at a time (see `PromptCacheStore`).
public struct PromptCacheSlot: @unchecked Sendable {
    public var tokens: [Int]
    public var cache: [KVCache]
    public var kvConfig: PromptCacheKVConfig

    public init(tokens: [Int], cache: [KVCache], kvConfig: PromptCacheKVConfig) {
        self.tokens = tokens
        self.cache = cache
        self.kvConfig = kvConfig
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
/// about that -- `checkout`/`checkin`/`drop` are individually atomic regardless, and the epoch
/// counters live under the same lock as the slots so "is this token still current" and "mutate
/// the slots" can never observe each other torn.
public final class PromptCacheStore: Sendable {
    // MARK: Lifecycle

    public init() {}

    // MARK: Public

    public static let shared: PromptCacheStore = .init()

    /// Removes and returns the slot for `modelName`, if any, together with the `PromptCacheEpoch`
    /// in force at that same instant -- see the checkout-semantics note on the type itself, and
    /// `PromptCacheEpoch`'s doc comment for why the epoch has to be captured atomically with the
    /// removal rather than read separately afterward.
    public func checkout(modelName: String) -> (slot: PromptCacheSlot, epoch: PromptCacheEpoch)? {
        let (slot, epoch) = checkoutOrCurrentEpoch(modelName: modelName)
        guard let slot else {
            return nil
        }

        return (slot, epoch)
    }

    /// Checks a slot out if there is one, and reports the `PromptCacheEpoch` in force either way
    /// -- both under a single acquisition of the lock.
    ///
    /// The atomicity is the point. Reading the epoch in a second, separate call after a checkout
    /// that found nothing would leave a window in which `dropAll()` lands between the two: the
    /// caller would then capture the *post*-drop epoch, its later `checkin` would be accepted,
    /// and the slot it built while holding a container that `clearCache()` has since evicted
    /// would survive the clear -- which is the very resurrection `PromptCacheEpoch` exists to
    /// prevent, just through a narrower window.
    public func checkoutOrCurrentEpoch(modelName: String) -> (slot: PromptCacheSlot?, epoch: PromptCacheEpoch) {
        state.withLock { state in
            (state.slots.removeValue(forKey: modelName), state.currentEpoch(modelName: modelName))
        }
    }

    /// The `PromptCacheEpoch` in force for `modelName` right now, without touching any slot --
    /// for the miss path, where `checkout` found nothing to remove but a slot will still be
    /// written back at the end of generation, so a token still has to be captured before that
    /// generation starts.
    public func currentEpoch(modelName: String) -> PromptCacheEpoch {
        state.withLock { $0.currentEpoch(modelName: modelName) }
    }

    /// Non-destructive lookup of a model's slot, for diagnostics and tests -- does not affect
    /// checkout state (unlike `checkout`, this does not remove the slot).
    public func peek(modelName: String) -> PromptCacheSlot? {
        state.withLock { $0.slots[modelName] }
    }

    /// Stores `slot` for `modelName`, checking it back in after clean completion, but only if
    /// `epoch` still equals the store's current epoch for `modelName`. A mismatch means
    /// `drop(modelName:)` or `dropAll()` ran while this slot's generation was in flight; the
    /// slot is dropped on the floor instead (which also releases its KV arrays) rather than
    /// resurrecting state the store had already invalidated. Returns whether the slot was
    /// actually stored, so the caller can log a rejection for diagnosability.
    @discardableResult
    public func checkin(modelName: String, slot: PromptCacheSlot, epoch: PromptCacheEpoch) -> Bool {
        state.withLock { state in
            guard state.currentEpoch(modelName: modelName) == epoch else {
                return false
            }

            state.slots[modelName] = slot
            return true
        }
    }

    /// Drops any slot for `modelName` and bumps that model's epoch -- used on model eviction and
    /// under memory pressure, where the cache's underlying weights/buffers are going away
    /// regardless of the slot's content, and available for explicit invalidation elsewhere. The
    /// epoch bump happens unconditionally, even when no slot is currently present, so a
    /// generation that is mid-flight for this model right now (and so already checked its slot
    /// out, or never found one) is still correctly invalidated when it tries to check in later.
    public func drop(modelName: String) {
        state.withLock { state in
            _ = state.slots.removeValue(forKey: modelName)
            state.modelEpochs[modelName, default: 0] += 1
        }
    }

    /// Drops every slot and bumps the global epoch -- used when the whole model cache is
    /// cleared. Per-model epoch counters are left untouched (bumping the global epoch already
    /// invalidates every model), and are never reset by this call either -- see `State
    /// .modelEpochs`.
    public func dropAll() {
        state.withLock { state in
            state.slots.removeAll()
            state.globalEpoch += 1
        }
    }

    // MARK: Private

    /// Bundles the slots with the epoch counters that guard them so both live under one lock.
    private struct State {
        var slots: [String: PromptCacheSlot] = [:]

        /// Per-model invalidation counters, bumped by `drop(modelName:)`. Deliberately **not**
        /// cleared when a model's slot is removed (by `checkout`, `drop`, or eviction) -- a model
        /// with no live slot right now must still correctly invalidate a write-back a past
        /// generation for that model is about to attempt.
        var modelEpochs: [String: UInt64] = [:]

        /// Global invalidation counter, bumped by `dropAll()`.
        var globalEpoch: UInt64 = 0

        func currentEpoch(modelName: String) -> PromptCacheEpoch {
            PromptCacheEpoch(global: globalEpoch, perModel: modelEpochs[modelName] ?? 0)
        }
    }

    private let state: Mutex<State> = .init(State())
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
    cache.isEmpty == false && cache.allSatisfy(promptCacheLayerIsReusable)
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
    kvConfig: PromptCacheKVConfig,
    store: PromptCacheStore
) -> PromptCacheResolution {
    guard enabled else {
        return .miss(reason: .disabled, epoch: nil)
    }
    guard hasMediaInput == false else {
        return .miss(reason: .multimodal, epoch: nil)
    }

    // Nothing may be checked out here, but a fresh cache will still be built and written back at
    // the end of generation, so the token it will need is captured now -- in the same locked
    // step as the checkout attempt, never as a separate read afterward.
    let (checkedOutSlot, epoch) = store.checkoutOrCurrentEpoch(modelName: modelName)
    guard let slot = checkedOutSlot else {
        return .miss(reason: .noSlot, epoch: epoch)
    }
    guard slot.kvConfig == kvConfig else {
        return .miss(reason: .kvConfigChanged, epoch: epoch)
    }
    guard promptCacheIsReusable(slot.cache) else {
        return .miss(reason: .rotated, epoch: epoch)
    }

    let matched = promptCacheLongestCommonPrefix(slot.tokens, newTokens)
    guard matched > 0 else {
        return .miss(reason: .mismatchAtZero, epoch: epoch)
    }

    // The TokenIterator needs at least one prompt token to prime the pump, so a full-prefix
    // match (matched == newTokens.count) still leaves the last token to be fed as the suffix.
    let trimTarget = min(matched, max(newTokens.count - 1, 0))
    for layer in slot.cache {
        layer.trim(layer.offset - trimTarget)
    }

    let reason: PromptCacheMissReason? = matched < slot.tokens.count ? .partialMismatch : nil
    return .reuse(matchedLength: trimTarget, cache: slot.cache, reason: reason, epoch: epoch)
}

import Foundation
import MLX
import MLXLMCommon
import MLXVLM

// MARK: - DecisionPrefillCache

/// Shared by transient runners for one immutable container. Tensor access occurs only inside its
/// serial `perform` operation; the recursive lock also protects diagnostic counters.
/// Stored tensors are evaluated before a scoring operation returns.
final class DecisionPrefillCache: @unchecked Sendable {
    struct Session {
        let output: LMOutput
        let pending: [PendingPrefix]
        let reusedTokens: Int
        let hit: String
    }

    struct PendingPrefix {
        let count: Int
        let tokens: [Int]
        let bytes: Data
        let chunk: Int
        let template: String
        let caches: [any KVCache]
        let cost: Int
    }

    private struct Prefix {
        let value: PendingPrefix
        let state: LMOutput.State
    }

    private struct Complete {
        let tokens: [Int]
        let content: String
        let template: String
        let last: MLXArray
    }

    private let lock: NSRecursiveLock = .init()
    private let enabled: Bool
    // One immutable container binds the model, loaded tokenizer and chat-template
    // identity; entries are never transferred to another container.
    private let modelIdentity: ObjectIdentifier
    private let trace: Bool
    private let budget: Int
    private var prefixes: [Prefix] = .init()
    private var complete: [Complete] = .init()
    private var requests = 0
    private var hits = 0
    private var reused = 0
    private var logical = 0
    private var evictions = 0
    private let ropeKey = LMOutput.Key<MLXArray>("qwen35.ropeDeltas")

    init(
        modelIdentity: ObjectIdentifier,
        enabled: Bool = ProcessInfo.processInfo.environment["SWAMA_DECISION_PREFIX_CACHE"] != "0",
        budget: Int = 128 * 1024 * 1024,
        trace: Bool = ProcessInfo.processInfo.environment["SWAMA_DECISION_CACHE_TRACE"] == "1"
    ) {
        self.modelIdentity = modelIdentity
        self.enabled = enabled
        self.budget = max(0, budget)
        self.trace = trace
    }

    /// The full cold prompt determines balanced boundaries. A shorter warm remainder
    /// must not choose its own balanced schedule.
    static func chunkSize(total: Int) -> Int? {
        guard total > PrefillParameters.defaultStepSize else {
            return nil
        }

        return PrefillParameters().chunkLength(forChunking: total - 1)
    }

    /// Require literal bytes as well as IDs. Decline normalization/partial-UTF8 cases
    /// whose original content cannot be mapped back to a complete template prefix.
    static func prefixBytes(content: String, prompt: String, decodedPrefix: String) -> Data? {
        guard !content.isEmpty, prompt.hasPrefix(decodedPrefix),
              !decodedPrefix.contains("\u{fffd}"),
              let location = prompt.range(of: content),
              prompt.range(of: content, range: location.upperBound ..< prompt.endIndex) == nil
        else {
            return nil
        }

        let header = String(prompt[..<location.lowerBound])
        guard decodedPrefix.utf8.count >= header.utf8.count else {
            return nil
        }

        let count = min(content.utf8.count, decodedPrefix.utf8.count - header.utf8.count)
        return Data(header.utf8) + Data(content.utf8.prefix(count))
    }

    static func templateIdentity(content: String, prompt: String) -> String {
        guard !content.isEmpty, let location = prompt.range(of: content),
              prompt.range(of: content, range: location.upperBound ..< prompt.endIndex) == nil
        else {
            return prompt
        }

        return String(prompt[..<location.lowerBound]) + "\u{0000}" + String(prompt[location.upperBound...])
    }

    func forward(
        context: ModelContext,
        input: LMInput,
        tokens: [Int],
        content: String,
        prompt: String,
        cold: () throws -> LMOutput
    ) throws -> Session {
        guard enabled else {
            return try Session(output: cold(), pending: [], reusedTokens: 0, hit: "disabled")
        }

        lock.lock()
        defer { lock.unlock() }
        requests += 1
        logical += tokens.count
        let template = Self.templateIdentity(content: content, prompt: prompt)
        guard enabled else {
            return try Session(output: cold(), pending: [], reusedTokens: 0, hit: "disabled")
        }

        if let index = complete
            .firstIndex(where: { $0.tokens == tokens && $0.content == content && $0.template == template })
        {
            let entry = complete.remove(at: index)
            complete.append(entry)
            hits += 1
            reused += tokens.count
            return Session(
                output: LMOutput(logits: entry.last.reshaped(1, 1, -1)),
                pending: [],
                reusedTokens: tokens.count,
                hit: "complete"
            )
        }

        // Only the checked text-only Qwen35 path has public continuation semantics
        // matching its cold plan. Other architectures retain the exact cold call.
        guard type(of: context.model) == MLXVLM.Qwen35.self,
              let chunk = Self.chunkSize(total: tokens.count)
        else {
            return try Session(output: cold(), pending: [], reusedTokens: 0, hit: "cold")
        }

        var selected: Prefix?
        for entry in prefixes.reversed() {
            let p = entry.value
            guard p.chunk == chunk, p.template == template, p.count < tokens.count,
                  Array(tokens.prefix(p.count)) == p.tokens,
                  let bytes = Self.prefixBytes(
                      content: content,
                      prompt: prompt,
                      decodedPrefix: context.tokenizer.decode(
                          tokenIds: p.tokens,
                          skipSpecialTokens: false
                      )
                  ),
                  bytes == p.bytes
            else {
                continue
            }

            if selected == nil || p.count > selected!.value.count {
                selected = entry
            }
        }
        let offset = selected?.value.count ?? 0
        let caches = selected?.value.caches.map { $0.copy() } ?? context.model.newCache(parameters: nil)
        guard caches.allSatisfy({ type(of: $0) == KVCacheSimple.self || type(of: $0) == MambaCache.self }) else {
            return try Session(output: cold(), pending: [], reusedTokens: 0, hit: "cold")
        }

        // A snapshot's arrays are never passed directly to a mutating model.
        var pending = [PendingPrefix]()
        var pendingCost = 0
        let collector = DecisionPrefixCollector { [self] processed, total in
            let count = offset + processed
            // The final prefill boundary has only the fixed answer-prefix token
            // left. Exact repetition is already served by the raw-logit memo;
            // retaining this large state crowds out genuinely shared prefixes.
            guard processed < total, count < tokens.count - 1, count > 0,
                  let bytes = Self.prefixBytes(
                      content: content,
                      prompt: prompt,
                      decodedPrefix: context.tokenizer.decode(
                          tokenIds: Array(tokens.prefix(count)),
                          skipSpecialTokens: false
                      )
                  )
            else {
                return
            }

            let cost = caches.reduce(0) { $0 + $1.state.reduce(0) { $0 + $1.nbytes } }
            guard cost <= budget else {
                return
            }

            while pendingCost + cost > budget, !pending.isEmpty {
                pendingCost -= pending.removeFirst().cost
            }
            let copies = caches.map { $0.copy() }
            pending.append(PendingPrefix(
                count: count,
                tokens: Array(tokens.prefix(count)),
                bytes: bytes,
                chunk: chunk,
                template: template,
                caches: copies,
                cost: cost
            ))
            pendingCost += cost
        }
        let parameters = PrefillParameters(
            stepSize: selected == nil ? nil : chunk,
            chunking: selected == nil ? .balanced : .remainder,
            progress: collector.record
        )
        let remainder = offset == 0 ? input :
            LMInput(tokens: MLXArray(Array(tokens.dropFirst(offset))).expandedDimensions(axis: 0))
        let output: LMOutput =
            switch try context.model.prepare(
                remainder,
                cache: caches,
                state: selected?.state,
                prefill: parameters
            ) {
            case let .logits(value): value
            case let .tokens(value):
                // Qwen35 currently returns logits. Keep the normal fallback shape if
                // an upstream implementation changes; no intermediate state is adopted.
                withPreparedCache(caches, lengths: value.sequenceLengths) {
                    context.model(value[text: .newAxis], cache: caches, state: selected?.state)
                }
            }
        if offset > 0 {
            hits += 1; reused += offset
        }
        return Session(
            output: output,
            pending: pending,
            reusedTokens: offset,
            hit: offset > 0 ? "prefix" : "cold"
        )
    }

    /// Called only after label/logit/reasoning validation and cancellation checks.
    func commit(_ session: Session, tokens: [Int], content: String, prompt: String, last: MLXArray) {
        guard enabled else {
            report(session, count: tokens.count)
            return
        }

        lock.lock()
        defer { lock.unlock() }
        complete.removeAll { $0.tokens == tokens && $0.content == content }
        let stored = last[.ellipsis]
        eval(stored)
        complete.append(Complete(
            tokens: tokens,
            content: content,
            template: Self.templateIdentity(content: content, prompt: prompt),
            last: stored
        ))
        if complete.count > 2 {
            complete.removeFirst(); evictions += 1
        }
        if let source = session.output.state?[ropeKey], source.size == 1,
           source.item(Int.self) == 0
        {
            // Checked against the pinned Qwen35 text-only positional state. An
            // image-bearing/unknown anchor cannot be adopted as a text prefix.
            var state = LMOutput.State()
            state[ropeKey] = source[.ellipsis]
            for value in session.pending {
                eval(value.caches)
                prefixes
                    .removeAll {
                        $0.value.tokens == value.tokens && $0.value.bytes == value.bytes && $0.value.chunk == value
                            .chunk
                    }
                prefixes.append(Prefix(value: value, state: state))
            }
        }
        while bytes > budget, !prefixes.isEmpty {
            prefixes.removeFirst(); evictions += 1
        }
        while bytes > budget, !complete.isEmpty {
            complete.removeFirst(); evictions += 1
        }
        report(session, count: tokens.count)
    }

    private func report(_ session: Session, count: Int) {
        lock.lock()
        defer { lock.unlock() }
        if trace {
            let record: [String: Any] = [
                "event": "decision_prefill_cache",
                "model_identity": String(describing: modelIdentity),
                "hit": session.hit,
                "prompt_tokens": count,
                "reused_tokens": session.reusedTokens,
                "requests": requests,
                "hits": hits,
                "total_prompt_tokens": logical,
                "total_reused_tokens": reused,
                "resident_bytes": bytes,
                "evictions": evictions
            ]
            if let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) {
                FileHandle.standardError.write(data + Data([10]))
            }
        }
    }

    private var bytes: Int {
        prefixes.reduce(0) { $0 + $1.value.cost } + complete.reduce(0) { $0 + $1.last.nbytes }
    }
}

// MARK: - DecisionPrefixCollector

/// The prefill API marks its callback Sendable, but calls it synchronously inside
/// one model-container operation. The collector never leaves that operation.
private final class DecisionPrefixCollector: @unchecked Sendable {
    private let body: (Int, Int) -> Void
    init(_ body: @escaping (Int, Int) -> Void) { self.body = body }
    func record(_ processed: Int, _ total: Int) { body(processed, total) }
}

// MARK: - DecisionPrefillStore

/// ModelPool creates a runner for each operation. Keep reuse with the actual model
/// container, without retaining model weights through this registry.
final class DecisionPrefillStore: @unchecked Sendable {
    static let shared: DecisionPrefillStore = .init()
    private final class Entry {
        weak var container: ModelContainer?
        let cache: DecisionPrefillCache
        init(_ container: ModelContainer) {
            self.container = container
            self.cache = DecisionPrefillCache(modelIdentity: ObjectIdentifier(container))
        }
    }

    private let lock: NSLock = .init()
    private var entries: [ObjectIdentifier: Entry] = [:]
    private var retired: [ObjectIdentifier: WeakContainer] = [:]

    private final class WeakContainer {
        weak var value: ModelContainer?
        init(_ value: ModelContainer) { self.value = value }
    }

    func cache(for container: ModelContainer) -> DecisionPrefillCache {
        lock.lock()
        defer { lock.unlock() }
        entries = entries.filter { $0.value.container != nil }
        retired = retired.filter { $0.value.value != nil }
        let id = ObjectIdentifier(container)
        if retired[id]?.value === container {
            // An in-flight operation can outlive a pool eviction. Its remaining
            // work must not repopulate the process registry after that teardown.
            return DecisionPrefillCache(modelIdentity: id)
        }
        if let entry = entries[id], entry.container === container {
            return entry.cache
        }
        let entry = Entry(container)
        entries[id] = entry
        return entry.cache
    }

    func remove(_ container: ModelContainer) {
        lock.lock()
        defer { lock.unlock() }
        entries.removeValue(forKey: ObjectIdentifier(container))
        retired[ObjectIdentifier(container)] = WeakContainer(container)
    }
}

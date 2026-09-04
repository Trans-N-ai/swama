import CryptoKit
import Foundation
import MLXLMCommon

// MARK: - TokenizerInputFingerprint

/// Content identity for every local file the pinned tokenizer loader currently reads, plus the
/// common split-vocabulary files that a future compatible loader may consult. File names,
/// presence/absence, sizes, and bytes all participate except for `config.json`: the pinned loader
/// uses only its parsed `model_type` to select a tokenizer fallback, so unrelated model-shape fields
/// are deliberately excluded. The model name and directory path do not participate, allowing equal
/// tokenizer inputs to be shared safely across model directories.
struct TokenizerInputFingerprint: Equatable, Hashable, Sendable {
    static let fileNames = [
        "added_tokens.json",
        "chat_template.jinja",
        "chat_template.json",
        "config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json"
    ]

    let digest: String
    let byteCount: Int64

    static func calculate(in directory: URL) throws -> Self {
        var hasher = SHA256()
        var byteCount: Int64 = 0

        for fileName in fileNames {
            hasher.update(data: Data(fileName.utf8))
            hasher.update(data: Data([0]))
            let file = directory.appendingPathComponent(fileName)
            guard FileManager.default.fileExists(atPath: file.path) else {
                hasher.update(data: Data([0]))
                continue
            }

            let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
            guard attributes[.type] as? FileAttributeType == .typeRegular,
                  let size = attributes[.size] as? NSNumber
            else {
                throw TokenizerFingerprintError.inputIsNotARegularFile(fileName)
            }

            hasher.update(data: Data([1]))
            byteCount = try adding(byteCount, Int64(bitPattern: size.uint64Value))
            if fileName == "config.json" {
                let data = try Data(contentsOf: file)
                guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    throw TokenizerFingerprintError.invalidModelConfiguration
                }

                if let modelType = object["model_type"] as? String {
                    hasher.update(data: Data([1]))
                    hasher.update(data: Data(modelType.utf8))
                }
                else {
                    hasher.update(data: Data([0]))
                }
                continue
            }

            var bigEndianSize = size.uint64Value.bigEndian
            withUnsafeBytes(of: &bigEndianSize) { hasher.update(bufferPointer: $0) }

            let handle = try FileHandle(forReadingFrom: file)
            defer { try? handle.close() }
            while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty {
                hasher.update(data: data)
            }
        }

        return .init(
            digest: hasher.finalize().map { String(format: "%02x", $0) }.joined(),
            byteCount: byteCount
        )
    }

    static func rejectionDigest(for directory: URL) -> String {
        let payload = Data("tokenizer-rejected\0\(directory.standardizedFileURL.path)".utf8)
        return SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined()
    }

    private static func adding(_ lhs: Int64, _ rhs: Int64) throws -> Int64 {
        let (sum, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow, rhs >= 0 else {
            throw TokenizerFingerprintError.inputSizeOverflow
        }

        return sum
    }
}

// MARK: - TokenizerFingerprintError

private enum TokenizerFingerprintError: Error {
    case invalidModelConfiguration
    case inputIsNotARegularFile(String)
    case inputSizeOverflow
}

// MARK: - TokenizerCacheConfiguration

struct TokenizerCacheConfiguration: Sendable {
    static let live: Self = .init(
        maximumEntries: 4,
        maximumInputBytes: 64 * 1024 * 1024
    )

    let maximumEntries: Int
    let maximumInputBytes: Int64
}

// MARK: - TokenizerCacheDiagnostics

struct TokenizerCacheDiagnostics: Sendable {
    static let live: Self = .init(
        hit: { SwamaDiagnostics.hitTokenizerCache(fingerprint: $0) },
        miss: { SwamaDiagnostics.missTokenizerCache(fingerprint: $0) },
        evict: { SwamaDiagnostics.evictTokenizerCache(fingerprint: $0, reason: $1) },
        reject: { SwamaDiagnostics.rejectTokenizerCache(fingerprint: $0, durationMilliseconds: $1) }
    )

    static let disabled: Self = .init(
        hit: { _ in },
        miss: { _ in },
        evict: { _, _ in },
        reject: { _, _ in }
    )

    let hit: @Sendable (String) -> Void
    let miss: @Sendable (String) -> Void
    let evict: @Sendable (String, String) -> Void
    let reject: @Sendable (String, Double) -> Void
}

// MARK: - TokenizerCacheOwner

struct TokenizerCacheOwner: Hashable, Sendable {
    static let standalone: Self = .init()

    init() {
        id = .init()
    }

    private let id: UUID
}

// MARK: - TokenizerCache

final class TokenizerCache: @unchecked Sendable {
    static let shared: TokenizerCache = .init(configuration: .live, diagnostics: .live)

    init(
        configuration: TokenizerCacheConfiguration,
        diagnostics: TokenizerCacheDiagnostics = .live
    ) {
        self.configuration = configuration
        self.diagnostics = diagnostics
    }

    /// Loads through a content-addressed, process-local cache. A failed fingerprint or an
    /// over-budget input fails open to the upstream loader without entering either the ready cache
    /// or the coalescing table.
    func load(
        from directory: URL,
        using upstream: any TokenizerLoader,
        owner: TokenizerCacheOwner = .standalone
    ) async throws -> any Tokenizer {
        try Task.checkCancellation()
        let startedAt = DispatchTime.now().uptimeNanoseconds
        let fingerprint: TokenizerInputFingerprint
        do {
            fingerprint = try TokenizerInputFingerprint.calculate(in: directory)
        }
        catch {
            diagnostics.reject(
                TokenizerInputFingerprint.rejectionDigest(for: directory),
                Self.elapsedMilliseconds(since: startedAt)
            )
            try Task.checkCancellation()
            return try await upstream.load(from: directory)
        }
        try Task.checkCancellation()

        guard configuration.maximumEntries > 0,
              configuration.maximumInputBytes > 0,
              fingerprint.byteCount <= configuration.maximumInputBytes
        else {
            diagnostics.reject(fingerprint.digest, Self.elapsedMilliseconds(since: startedAt))
            try Task.checkCancellation()
            return try await upstream.load(from: directory)
        }

        let (acquisition, newTask) = acquire(
            fingerprint: fingerprint,
            directory: directory,
            upstream: upstream,
            owner: owner
        )
        switch acquisition {
        case let .ready(tokenizer):
            diagnostics.hit(fingerprint.digest)
            return tokenizer

        case let .waiting(sequence, waiter):
            if let newTask {
                diagnostics.miss(fingerprint.digest)
                observe(
                    newTask,
                    fingerprint: fingerprint,
                    sequence: sequence,
                    startedAt: startedAt
                )
            }
            else {
                diagnostics.hit(fingerprint.digest)
            }

            let completed = try await waiter.value {
                self.cancelWaiter(
                    fingerprint: fingerprint.digest,
                    sequence: sequence,
                    waiterID: waiter.id
                )
            }
            try Task.checkCancellation()
            let fingerprintAfterWaiting = try? TokenizerInputFingerprint.calculate(in: directory)
            try Task.checkCancellation()
            guard fingerprintAfterWaiting == fingerprint else {
                diagnostics.reject(
                    fingerprint.digest,
                    Self.elapsedMilliseconds(since: startedAt)
                )
                throw TokenizerCacheError.inputsChangedDuringLoad
            }

            return completed.tokenizer
        }
    }

    func purge(owner: TokenizerCacheOwner = .standalone) {
        let (fingerprints, tasks, waiters) = lock.withLock {
            () -> ([String], [Task<CompletedLoad, Error>], [Waiter]) in
            var fingerprints: [String] = []
            for fingerprint in Array(ready.keys) {
                guard var entry = ready[fingerprint] else {
                    continue
                }

                entry.owners.remove(owner)
                if entry.owners.isEmpty {
                    ready.removeValue(forKey: fingerprint)
                    fingerprints.append(fingerprint)
                }
                else {
                    ready[fingerprint] = entry
                }
            }

            var tasks: [Task<CompletedLoad, Error>] = []
            var removedWaiters: [Waiter] = []
            for fingerprint in Array(pending.keys) {
                guard var pendingLoad = pending[fingerprint] else {
                    continue
                }

                let ownerWaiters = pendingLoad.waiters.values.filter { $0.owner == owner }
                removedWaiters.append(contentsOf: ownerWaiters)
                for waiter in ownerWaiters {
                    pendingLoad.waiters.removeValue(forKey: waiter.id)
                }
                if pendingLoad.waiters.isEmpty {
                    pending.removeValue(forKey: fingerprint)
                    tasks.append(pendingLoad.task)
                }
                else {
                    pending[fingerprint] = pendingLoad
                }
            }

            return (fingerprints.sorted(), tasks, removedWaiters)
        }
        for fingerprint in fingerprints {
            diagnostics.evict(fingerprint, "explicit")
        }
        for task in tasks {
            task.cancel()
        }
        for waiter in waiters {
            waiter.resolve(.failure(CancellationError()))
        }
    }

    #if DEBUG
        func entryCountForTesting() -> Int {
            lock.withLock { ready.count }
        }

        func pendingCountForTesting() -> Int {
            lock.withLock { pending.count }
        }

        func waiterCountForTesting() -> Int {
            lock.withLock { pending.values.reduce(0) { $0 + $1.waiters.count } }
        }
    #endif

    private enum Acquisition {
        case ready(any Tokenizer)
        case waiting(sequence: UInt64, waiter: Waiter)
    }

    private struct ReadyEntry: Sendable {
        let tokenizer: any Tokenizer
        var lastAccess: UInt64
        var owners: Set<TokenizerCacheOwner>
    }

    private struct CompletedLoad: Sendable {
        let tokenizer: any Tokenizer
        let fingerprintAfterLoad: TokenizerInputFingerprint?
    }

    private struct PendingLoad: Sendable {
        let sequence: UInt64
        let task: Task<CompletedLoad, Error>
        var waiters: [UUID: Waiter]
    }

    private let configuration: TokenizerCacheConfiguration
    private let diagnostics: TokenizerCacheDiagnostics
    private let lock: NSLock = .init()
    private var ready: [String: ReadyEntry] = [:]
    private var pending: [String: PendingLoad] = [:]
    private var accessSequence: UInt64 = 0
    private var loadSequence: UInt64 = 0

    private func acquire(
        fingerprint: TokenizerInputFingerprint,
        directory: URL,
        upstream: any TokenizerLoader,
        owner: TokenizerCacheOwner
    ) -> (Acquisition, Task<CompletedLoad, Error>?) {
        lock.withLock {
            if var entry = ready[fingerprint.digest] {
                accessSequence &+= 1
                entry.lastAccess = accessSequence
                entry.owners.insert(owner)
                ready[fingerprint.digest] = entry
                return (.ready(entry.tokenizer), nil)
            }

            let waiter = Waiter(owner: owner)
            if var pendingLoad = pending[fingerprint.digest] {
                pendingLoad.waiters[waiter.id] = waiter
                pending[fingerprint.digest] = pendingLoad
                return (.waiting(sequence: pendingLoad.sequence, waiter: waiter), nil)
            }

            loadSequence &+= 1
            let sequence = loadSequence
            let task = Task.detached { () throws -> CompletedLoad in
                let tokenizer = try await upstream.load(from: directory)
                let fingerprintAfterLoad = try? TokenizerInputFingerprint.calculate(in: directory)
                return .init(tokenizer: tokenizer, fingerprintAfterLoad: fingerprintAfterLoad)
            }
            pending[fingerprint.digest] = .init(
                sequence: sequence,
                task: task,
                waiters: [waiter.id: waiter]
            )
            return (.waiting(sequence: sequence, waiter: waiter), task)
        }
    }

    private func observe(
        _ task: Task<CompletedLoad, Error>,
        fingerprint: TokenizerInputFingerprint,
        sequence: UInt64,
        startedAt: UInt64
    ) {
        Task.detached { [self] in
            let result = await task.result
            complete(
                result,
                fingerprint: fingerprint,
                sequence: sequence,
                startedAt: startedAt
            )
        }
    }

    private func complete(
        _ result: Result<CompletedLoad, Error>,
        fingerprint: TokenizerInputFingerprint,
        sequence: UInt64,
        startedAt: UInt64
    ) {
        var finalResult = result
        var waiters: [Waiter] = []
        var evictedFingerprints: [String] = []
        var rejected = false

        lock.lock()
        guard let pendingLoad = pending[fingerprint.digest], pendingLoad.sequence == sequence else {
            lock.unlock()
            return
        }

        pending.removeValue(forKey: fingerprint.digest)
        waiters = Array(pendingLoad.waiters.values)

        if case let .success(completed) = result {
            if completed.fingerprintAfterLoad == fingerprint {
                accessSequence &+= 1
                ready[fingerprint.digest] = .init(
                    tokenizer: completed.tokenizer,
                    lastAccess: accessSequence,
                    owners: Set(waiters.map(\.owner))
                )
                evictedFingerprints = evictToBudgetLocked()
            }
            else {
                rejected = true
                finalResult = .failure(TokenizerCacheError.inputsChangedDuringLoad)
            }
        }
        lock.unlock()

        if rejected {
            diagnostics.reject(
                fingerprint.digest,
                Self.elapsedMilliseconds(since: startedAt)
            )
        }
        for evictedFingerprint in evictedFingerprints {
            diagnostics.evict(evictedFingerprint, "budget")
        }
        for waiter in waiters {
            waiter.resolve(finalResult)
        }
    }

    private func cancelWaiter(
        fingerprint: String,
        sequence: UInt64,
        waiterID: UUID
    ) {
        var taskToCancel: Task<CompletedLoad, Error>?
        lock.withLock {
            guard var pendingLoad = pending[fingerprint], pendingLoad.sequence == sequence else {
                return
            }

            pendingLoad.waiters.removeValue(forKey: waiterID)
            if pendingLoad.waiters.isEmpty {
                pending.removeValue(forKey: fingerprint)
                taskToCancel = pendingLoad.task
            }
            else {
                pending[fingerprint] = pendingLoad
            }
        }
        taskToCancel?.cancel()
    }

    private func evictToBudgetLocked() -> [String] {
        var fingerprints: [String] = []
        while ready.count > configuration.maximumEntries,
              let victim = ready.min(by: { lhs, rhs in
                  if lhs.value.lastAccess == rhs.value.lastAccess {
                      lhs.key < rhs.key
                  }
                  else {
                      lhs.value.lastAccess < rhs.value.lastAccess
                  }
              })
        {
            ready.removeValue(forKey: victim.key)
            fingerprints.append(victim.key)
        }
        return fingerprints
    }

    private static func elapsedMilliseconds(since startedAt: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - startedAt) / 1_000_000
    }

    private final class Waiter: @unchecked Sendable {
        init(owner: TokenizerCacheOwner) {
            self.owner = owner
        }

        let id: UUID = .init()
        let owner: TokenizerCacheOwner

        func value(onCancel: @escaping @Sendable () -> Void) async throws -> CompletedLoad {
            do {
                try Task.checkCancellation()
                return try await withTaskCancellationHandler {
                    try await withCheckedThrowingContinuation { continuation in
                        let immediate = lock.withLock { () -> Result<CompletedLoad, Error>? in
                            if let result {
                                return result
                            }
                            if cancelled {
                                return .failure(CancellationError())
                            }
                            self.continuation = continuation
                            return nil
                        }
                        if let immediate {
                            continuation.resume(with: immediate)
                        }
                    }
                } onCancel: {
                    if self.cancel() {
                        onCancel()
                    }
                }
            }
            catch is CancellationError {
                if cancel() {
                    onCancel()
                }
                throw CancellationError()
            }
        }

        func resolve(_ result: Result<CompletedLoad, Error>) {
            let continuation = lock.withLock { () -> CheckedContinuation<CompletedLoad, Error>? in
                guard self.result == nil, !cancelled else {
                    return nil
                }

                self.result = result
                let continuation = self.continuation
                self.continuation = nil
                return continuation
            }
            continuation?.resume(with: result)
        }

        private func cancel() -> Bool {
            let (didCancel, continuation) = lock.withLock {
                () -> (Bool, CheckedContinuation<CompletedLoad, Error>?) in
                guard result == nil, !cancelled else {
                    return (false, nil)
                }

                cancelled = true
                let continuation = self.continuation
                self.continuation = nil
                return (true, continuation)
            }
            continuation?.resume(throwing: CancellationError())
            return didCancel
        }

        private let lock: NSLock = .init()
        private var continuation: CheckedContinuation<CompletedLoad, Error>?
        private var result: Result<CompletedLoad, Error>?
        private var cancelled = false
    }
}

// MARK: - TokenizerCacheError

enum TokenizerCacheError: Error {
    case inputsChangedDuringLoad
}

// MARK: - CachedTokenizerLoader

struct CachedTokenizerLoader: TokenizerLoader {
    init(
        upstream: any TokenizerLoader,
        cache: TokenizerCache = .shared,
        owner: TokenizerCacheOwner = .standalone
    ) {
        self.upstream = upstream
        self.cache = cache
        self.owner = owner
    }

    func load(from directory: URL) async throws -> any Tokenizer {
        try await cache.load(from: directory, using: upstream, owner: owner)
    }

    private let upstream: any TokenizerLoader
    private let cache: TokenizerCache
    private let owner: TokenizerCacheOwner
}

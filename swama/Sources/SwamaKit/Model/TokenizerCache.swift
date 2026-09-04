import CryptoKit
import Foundation
import MLXLMCommon

// MARK: - TokenizerInputFingerprint

/// Content identity for every local file the pinned tokenizer loader currently reads, plus the
/// common split-vocabulary files that a future compatible loader may consult. File names,
/// presence/absence, sizes, and bytes all participate; the model name and directory path do not,
/// so byte-identical tokenizers can be shared safely across model directories.
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
            var bigEndianSize = size.uint64Value.bigEndian
            withUnsafeBytes(of: &bigEndianSize) { hasher.update(bufferPointer: $0) }
            byteCount = try adding(byteCount, Int64(bitPattern: size.uint64Value))

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

// MARK: - TokenizerCache

actor TokenizerCache {
    static let shared: TokenizerCache = .init(configuration: .live, diagnostics: .live)

    init(
        configuration: TokenizerCacheConfiguration,
        diagnostics: TokenizerCacheDiagnostics = .live
    ) {
        self.configuration = configuration
        self.diagnostics = diagnostics
    }

    /// Loads through a content-addressed, process-local cache. Fingerprinting intentionally occurs
    /// before entering the actor so callers hashing distinct directories do not serialize behind
    /// one another. A failed fingerprint or an over-budget input fails open to the upstream loader
    /// without entering either the ready cache or the coalescing table.
    nonisolated func load(
        from directory: URL,
        using upstream: any TokenizerLoader
    ) async throws -> any Tokenizer {
        let startedAt = DispatchTime.now().uptimeNanoseconds
        let fingerprint: TokenizerInputFingerprint
        do {
            fingerprint = try TokenizerInputFingerprint.calculate(in: directory)
        }
        catch {
            diagnostics.reject(
                TokenizerInputFingerprint.rejectionDigest(for: directory),
                elapsedMilliseconds(since: startedAt)
            )
            return try await upstream.load(from: directory)
        }

        return try await load(
            fingerprint: fingerprint,
            directory: directory,
            upstream: upstream,
            startedAt: startedAt
        )
    }

    func purge() {
        for fingerprint in ready.keys.sorted() {
            diagnostics.evict(fingerprint, "explicit")
        }
        ready.removeAll()
        for pendingLoad in pending.values {
            pendingLoad.task.cancel()
        }
        pending.removeAll()
    }

    #if DEBUG
        func entryCountForTesting() -> Int {
            ready.count
        }
    #endif

    private struct ReadyEntry: Sendable {
        let tokenizer: any Tokenizer
        var lastAccess: UInt64
    }

    private struct CompletedLoad: Sendable {
        let tokenizer: any Tokenizer
        let fingerprintAfterLoad: TokenizerInputFingerprint?
    }

    private struct PendingLoad: Sendable {
        let sequence: UInt64
        let task: Task<CompletedLoad, Error>
        let startedAt: UInt64
    }

    private let configuration: TokenizerCacheConfiguration
    private let diagnostics: TokenizerCacheDiagnostics
    private var ready: [String: ReadyEntry] = [:]
    private var pending: [String: PendingLoad] = [:]
    private var accessSequence: UInt64 = 0
    private var loadSequence: UInt64 = 0

    private func load(
        fingerprint: TokenizerInputFingerprint,
        directory: URL,
        upstream: any TokenizerLoader,
        startedAt: UInt64
    ) async throws -> any Tokenizer {
        guard configuration.maximumEntries > 0,
              configuration.maximumInputBytes > 0,
              fingerprint.byteCount <= configuration.maximumInputBytes
        else {
            diagnostics.reject(fingerprint.digest, elapsedMilliseconds(since: startedAt))
            return try await upstream.load(from: directory)
        }

        if var entry = ready[fingerprint.digest] {
            accessSequence &+= 1
            entry.lastAccess = accessSequence
            ready[fingerprint.digest] = entry
            diagnostics.hit(fingerprint.digest)
            return entry.tokenizer
        }

        if let pendingLoad = pending[fingerprint.digest] {
            diagnostics.hit(fingerprint.digest)
            return try await pendingLoad.task.value.tokenizer
        }

        loadSequence &+= 1
        let sequence = loadSequence
        let task = Task.detached { () throws -> CompletedLoad in
            let tokenizer = try await upstream.load(from: directory)
            let fingerprintAfterLoad = try? TokenizerInputFingerprint.calculate(in: directory)
            return .init(tokenizer: tokenizer, fingerprintAfterLoad: fingerprintAfterLoad)
        }
        let pendingLoad = PendingLoad(
            sequence: sequence,
            task: task,
            startedAt: startedAt
        )
        pending[fingerprint.digest] = pendingLoad
        diagnostics.miss(fingerprint.digest)

        do {
            let completed = try await task.value
            guard pending[fingerprint.digest]?.sequence == sequence else {
                return completed.tokenizer
            }

            pending.removeValue(forKey: fingerprint.digest)

            guard completed.fingerprintAfterLoad == fingerprint else {
                diagnostics.reject(
                    fingerprint.digest,
                    elapsedMilliseconds(since: pendingLoad.startedAt)
                )
                return completed.tokenizer
            }

            accessSequence &+= 1
            ready[fingerprint.digest] = .init(
                tokenizer: completed.tokenizer,
                lastAccess: accessSequence
            )
            evictToBudget()
            return completed.tokenizer
        }
        catch {
            if pending[fingerprint.digest]?.sequence == sequence {
                pending.removeValue(forKey: fingerprint.digest)
            }
            throw error
        }
    }

    private func evictToBudget() {
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
            diagnostics.evict(victim.key, "budget")
        }
    }

    private nonisolated func elapsedMilliseconds(since startedAt: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - startedAt) / 1_000_000
    }
}

// MARK: - CachedTokenizerLoader

struct CachedTokenizerLoader: TokenizerLoader {
    init(
        upstream: any TokenizerLoader,
        cache: TokenizerCache = .shared
    ) {
        self.upstream = upstream
        self.cache = cache
    }

    func load(from directory: URL) async throws -> any Tokenizer {
        try await cache.load(from: directory, using: upstream)
    }

    private let upstream: any TokenizerLoader
    private let cache: TokenizerCache
}

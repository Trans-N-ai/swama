import Foundation
import MLXLMCommon

// MARK: - ModelLoadPhaseRecorder

final class ModelLoadPhaseRecorder: @unchecked Sendable {
    var phases: SwamaModelLoadPhases {
        lock.withLock {
            .init(tokenizerMs: tokenizerMilliseconds)
        }
    }

    func recordTokenizer(milliseconds: Double) {
        lock.withLock {
            tokenizerMilliseconds = milliseconds
        }
    }

    private let lock: NSLock = .init()
    private var tokenizerMilliseconds: Double?
}

// MARK: - DiagnosticTokenizerLoader

struct DiagnosticTokenizerLoader: TokenizerLoader {
    let upstream: any TokenizerLoader
    let phases: ModelLoadPhaseRecorder
    let cache: TokenizerCache

    init(
        upstream: any TokenizerLoader,
        phases: ModelLoadPhaseRecorder,
        cache: TokenizerCache = .shared
    ) {
        self.upstream = upstream
        self.phases = phases
        self.cache = cache
    }

    func load(from directory: URL) async throws -> any Tokenizer {
        let startedAt = DispatchTime.now().uptimeNanoseconds
        do {
            let tokenizer = try await cache.load(from: directory, using: upstream)
            phases.recordTokenizer(milliseconds: elapsedMilliseconds(since: startedAt))
            return tokenizer
        }
        catch {
            phases.recordTokenizer(milliseconds: elapsedMilliseconds(since: startedAt))
            throw error
        }
    }

    private func elapsedMilliseconds(since startedAt: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - startedAt) / 1_000_000
    }
}

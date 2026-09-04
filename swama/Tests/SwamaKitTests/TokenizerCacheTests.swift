import Foundation
import MLXLMCommon
@testable import SwamaKit
import Testing

// MARK: - TokenizerCacheTests

@Suite("Bounded tokenizer cache", .serialized)
struct TokenizerCacheTests {
    @Test func identicalInputsAcrossDirectoriesCoalesceAndShare() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let first = try fixture.directory(named: "first")
        let second = try fixture.directory(named: "second")
        let loader = CountingTokenizerLoader(delay: .milliseconds(100))
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )

        async let firstValue = cache.load(from: first, using: loader)
        async let secondValue = cache.load(from: second, using: loader)
        let values = try await [firstValue, secondValue]

        #expect(await loader.calls == 1)
        #expect(values.compactMap { ($0 as? StubTokenizer)?.identifier } == ["load-1", "load-1"])

        _ = try await cache.load(from: first, using: loader)
        #expect(await loader.calls == 1)
    }

    @Test func diagnosticLoaderUsesTheCacheAndTimesHitsAsWellAsMisses() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let directory = try fixture.directory(named: "diagnostic-loader")
        let upstream = CountingTokenizerLoader()
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )
        let missPhases = ModelLoadPhaseRecorder()
        let hitPhases = ModelLoadPhaseRecorder()

        _ = try await DiagnosticTokenizerLoader(
            upstream: upstream,
            phases: missPhases,
            cache: cache
        ).load(from: directory)
        _ = try await DiagnosticTokenizerLoader(
            upstream: upstream,
            phases: hitPhases,
            cache: cache
        ).load(from: directory)

        #expect(await upstream.calls == 1)
        #expect(missPhases.phases.tokenizerMs != nil)
        #expect(hitPhases.phases.tokenizerMs != nil)
    }

    @Test func everyDeclaredTokenizerInputChangesIdentity() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let loader = CountingTokenizerLoader()
        let cache = TokenizerCache(
            configuration: .init(
                maximumEntries: expectedTokenizerInputFileNames.count + 1,
                maximumInputBytes: 1_000_000
            ),
            diagnostics: .disabled
        )
        let baseline = try fixture.directory(named: "baseline")
        _ = try await cache.load(from: baseline, using: loader)

        #expect(TokenizerInputFingerprint.fileNames == expectedTokenizerInputFileNames)
        for fileName in expectedTokenizerInputFileNames {
            let changed = try fixture.directory(named: "changed-\(fileName.replacingOccurrences(of: ".", with: "-"))")
            let data = fileName == "config.json"
                ? Data(#"{"model_type":"changed-model"}"#.utf8)
                : Data("changed \(fileName)".utf8)
            try data.write(to: changed.appendingPathComponent(fileName))
            _ = try await cache.load(from: changed, using: loader)
        }

        #expect(await loader.calls == expectedTokenizerInputFileNames.count + 1)
    }

    @Test func missingAndEmptyOptionalInputsHaveDifferentIdentities() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let missing = try fixture.directory(named: "missing")
        let empty = try fixture.directory(named: "empty")
        try FileManager.default.removeItem(at: missing.appendingPathComponent("merges.txt"))
        try Data().write(to: empty.appendingPathComponent("merges.txt"))
        let loader = CountingTokenizerLoader()
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )

        _ = try await cache.load(from: missing, using: loader)
        _ = try await cache.load(from: empty, using: loader)

        #expect(await loader.calls == 2)
    }

    @Test func sameSizeContentChangeCannotReuseThePriorEntry() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let directory = try fixture.directory(named: "same-size-change")
        let tokenizerFile = directory.appendingPathComponent("tokenizer.json")
        let original = try Data(contentsOf: tokenizerFile)
        let loader = CountingTokenizerLoader()
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )

        _ = try await cache.load(from: directory, using: loader)
        try Data(repeating: 0x78, count: original.count).write(to: tokenizerFile)
        _ = try await cache.load(from: directory, using: loader)

        #expect(await loader.calls == 2)
    }

    @Test func invalidModelConfigurationNeverFallsBackToAnOldEntry() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let directory = try fixture.directory(named: "invalid-config")
        let loader = ConfigurationValidatingTokenizerLoader()
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )

        _ = try await cache.load(from: directory, using: loader)
        try Data("not valid json".utf8).write(to: directory.appendingPathComponent("config.json"))
        await #expect(throws: StubLoaderError.self) {
            _ = try await cache.load(from: directory, using: loader)
        }
        await #expect(throws: StubLoaderError.self) {
            _ = try await cache.load(from: directory, using: loader)
        }

        #expect(await loader.calls == 3)
        #expect(cache.entryCountForTesting() == 1)
    }

    @Test func modelWeightsAndGenerationPolicyDoNotSplitIdenticalTokenizers() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let first = try fixture.directory(named: "first")
        let second = try fixture.directory(named: "second")
        for (fileName, firstValue, secondValue) in [
            ("generation_config.json", "first-stop-policy", "second-stop-policy"),
            ("model.safetensors", "first-weights", "second-weights"),
            ("preprocessor_config.json", "first-vision", "second-vision"),
            (".swama-meta.json", "first-metadata", "second-metadata")
        ] {
            try Data(firstValue.utf8).write(to: first.appendingPathComponent(fileName))
            try Data(secondValue.utf8).write(to: second.appendingPathComponent(fileName))
        }
        let loader = CountingTokenizerLoader(delay: .milliseconds(100))
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )

        async let firstValue = cache.load(from: first, using: loader)
        async let secondValue = cache.load(from: second, using: loader)
        _ = try await [firstValue, secondValue]

        #expect(await loader.calls == 1)
    }

    @Test func unrelatedModelConfigurationFieldsDoNotSplitTheSameModelType() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let first = try fixture.directory(named: "first")
        let second = try fixture.directory(named: "second")
        try Data(#"{"model_type":"shared-model","hidden_size":128}"#.utf8)
            .write(to: first.appendingPathComponent("config.json"))
        try Data(#"{"model_type":"shared-model","hidden_size":4096}"#.utf8)
            .write(to: second.appendingPathComponent("config.json"))
        let loader = CountingTokenizerLoader(delay: .milliseconds(100))
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )

        async let firstValue = cache.load(from: first, using: loader)
        async let secondValue = cache.load(from: second, using: loader)
        _ = try await [firstValue, secondValue]

        #expect(await loader.calls == 1)
    }

    @Test func failedLoadsAreNeverCached() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let directory = try fixture.directory(named: "failure")
        let loader = CountingTokenizerLoader(failuresRemaining: 1)
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )

        await #expect(throws: StubLoaderError.self) {
            _ = try await cache.load(from: directory, using: loader)
        }
        _ = try await cache.load(from: directory, using: loader)
        _ = try await cache.load(from: directory, using: loader)

        #expect(await loader.calls == 2)
    }

    @Test func leastRecentlyUsedEntryIsEvictedAndExplicitPurgeClearsTheRest() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let first = try fixture.directory(named: "first", seed: "first")
        let second = try fixture.directory(named: "second", seed: "second")
        let third = try fixture.directory(named: "third", seed: "third")
        let loader = CountingTokenizerLoader()
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 2, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )

        _ = try await cache.load(from: first, using: loader)
        _ = try await cache.load(from: second, using: loader)
        _ = try await cache.load(from: first, using: loader) // first is now most-recently used
        _ = try await cache.load(from: third, using: loader) // second is evicted
        _ = try await cache.load(from: second, using: loader)
        #expect(await loader.calls == 4)
        #expect(cache.entryCountForTesting() == 2)

        cache.purge()
        #expect(cache.entryCountForTesting() == 0)
        _ = try await cache.load(from: second, using: loader)
        #expect(await loader.calls == 5)
    }

    @Test func purgeInvalidatesAnInFlightLoadInsteadOfResurrectingIt() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let directory = try fixture.directory(named: "in-flight")
        let loader = CountingTokenizerLoader(delay: .milliseconds(200))
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )

        let first = Task { try await cache.load(from: directory, using: loader) }
        while await loader.calls == 0 {
            await Task.yield()
        }
        cache.purge()
        await #expect(throws: CancellationError.self) { _ = try await first.value }
        #expect(cache.entryCountForTesting() == 0)

        _ = try await cache.load(from: directory, using: loader)
        #expect(await loader.calls == 2)
    }

    @Test func producerInputDriftRejectsEveryCrossDirectoryWaiter() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let firstDirectory = try fixture.directory(named: "first")
        let secondDirectory = try fixture.directory(named: "second")
        let loader = SuspendedTokenizerLoader(mutateFileName: "tokenizer_config.json")
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )

        let first = Task { try await cache.load(from: firstDirectory, using: loader) }
        await loader.waitUntilStarted()
        let second = Task { try await cache.load(from: secondDirectory, using: loader) }
        while cache.waiterCountForTesting() < 2 {
            await Task.yield()
        }
        await loader.release()

        await #expect(throws: TokenizerCacheError.self) { _ = try await first.value }
        await #expect(throws: TokenizerCacheError.self) { _ = try await second.value }
        #expect(cache.entryCountForTesting() == 0)
    }

    @Test func waiterInputDriftRejectsOnlyThatWaiter() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let producerDirectory = try fixture.directory(named: "producer")
        let changedWaiterDirectory = try fixture.directory(named: "changed-waiter")
        let loader = SuspendedTokenizerLoader()
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )

        let producer = Task { try await cache.load(from: producerDirectory, using: loader) }
        await loader.waitUntilStarted()
        let changedWaiter = Task { try await cache.load(from: changedWaiterDirectory, using: loader) }
        while cache.waiterCountForTesting() < 2 {
            await Task.yield()
        }
        try Data("changed-waiter-input".utf8)
            .write(to: changedWaiterDirectory.appendingPathComponent("tokenizer_config.json"))
        await loader.release()

        _ = try await producer.value
        await #expect(throws: TokenizerCacheError.self) { _ = try await changedWaiter.value }
        #expect(cache.entryCountForTesting() == 1)
    }

    @Test func cancellingTheOnlyWaiterReturnsPromptlyAndDropsThePendingLoad() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let directory = try fixture.directory(named: "cancel-only")
        let loader = SuspendedTokenizerLoader()
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )
        let outcome = TokenizerLoadOutcome()
        let task = Task {
            do {
                _ = try await cache.load(from: directory, using: loader)
                await outcome.record(.value)
            }
            catch is CancellationError {
                await outcome.record(.cancelled)
            }
            catch {
                await outcome.record(.otherError)
            }
        }
        await loader.waitUntilStarted()

        task.cancel()
        try await waitForOutcome(outcome, timeout: .milliseconds(100))
        #expect(await outcome.value == .cancelled)
        #expect(cache.pendingCountForTesting() == 0)
        await loader.release()
        await loader.waitUntilFinished()
        #expect(await loader.wasCancelled)
        await task.value
    }

    @Test func cancellingOneWaiterDoesNotKillTheSharedLoadForAnotherWaiter() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let firstDirectory = try fixture.directory(named: "first")
        let secondDirectory = try fixture.directory(named: "second")
        let loader = SuspendedTokenizerLoader()
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )
        let cancelledOutcome = TokenizerLoadOutcome()
        let first = Task {
            do {
                _ = try await cache.load(from: firstDirectory, using: loader)
                await cancelledOutcome.record(.value)
            }
            catch is CancellationError {
                await cancelledOutcome.record(.cancelled)
            }
            catch {
                await cancelledOutcome.record(.otherError)
            }
        }
        await loader.waitUntilStarted()
        let second = Task { try await cache.load(from: secondDirectory, using: loader) }
        while cache.waiterCountForTesting() < 2 {
            await Task.yield()
        }

        first.cancel()
        try await waitForOutcome(cancelledOutcome, timeout: .milliseconds(100))
        #expect(await cancelledOutcome.value == .cancelled)
        #expect(cache.pendingCountForTesting() == 1)
        #expect(cache.waiterCountForTesting() == 1)
        await loader.release()
        _ = try await second.value
        await first.value

        #expect(await loader.calls == 1)
        #expect(await !(loader.wasCancelled))
        #expect(cache.entryCountForTesting() == 1)
    }

    @Test func inputsChangedDuringLoadAreRejectedAndNeverCachedUnderTheOldFingerprint() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let directory = try fixture.directory(named: "changing")
        let loader = MutatingTokenizerLoader(fileName: "tokenizer_config.json")
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )

        await #expect(throws: TokenizerCacheError.self) {
            _ = try await cache.load(from: directory, using: loader)
        }
        #expect(cache.entryCountForTesting() == 0)
        _ = try await cache.load(from: directory, using: loader)
        _ = try await cache.load(from: directory, using: loader)

        #expect(await loader.calls == 2)
        #expect(cache.entryCountForTesting() == 1)
    }

    @Test func oversizedInputsLoadWithoutEnteringTheCache() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let directory = try fixture.directory(named: "oversized")
        let loader = CountingTokenizerLoader()
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 4, maximumInputBytes: 1),
            diagnostics: .disabled
        )

        _ = try await cache.load(from: directory, using: loader)
        _ = try await cache.load(from: directory, using: loader)

        #expect(await loader.calls == 2)
        #expect(cache.entryCountForTesting() == 0)
    }

    @Test func purgeReleasesTheTokenizerOwner() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let directory = try fixture.directory(named: "lifetime")
        var lifetime: TokenizerLifetime? = TokenizerLifetime()
        let loader = try LifetimeTokenizerLoader(lifetime: #require(lifetime))
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 1, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )
        weak let weakLifetime = lifetime
        lifetime = nil

        var tokenizer: (any MLXLMCommon.Tokenizer)? = try await cache.load(from: directory, using: loader)
        #expect(weakLifetime != nil)
        tokenizer = nil
        _ = tokenizer
        cache.purge()

        #expect(weakLifetime == nil)
    }

    @Test func modelPoolGlobalClearReleasesTokenizerBeforeMLXCleanup() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let directory = try fixture.directory(named: "pool-clear")
        var lifetime: TokenizerLifetime? = TokenizerLifetime()
        let observation = try TokenizerCleanupObservation(lifetime: #require(lifetime))
        let loader = try LifetimeTokenizerLoader(lifetime: #require(lifetime))
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 1, maximumInputBytes: 1_000_000),
            diagnostics: .disabled
        )
        var tokenizer: (any MLXLMCommon.Tokenizer)? = try await cache.load(from: directory, using: loader)
        lifetime = nil
        tokenizer = nil
        _ = tokenizer
        #expect(observation.isAlive)

        let pool = ModelPool(
            memoryHooks: .init(
                activeMemory: { 0 },
                clearCache: { observation.recordCleanup() }
            ),
            tokenizerCache: cache
        )
        await pool.clearCache()

        #expect(observation.wasReleasedBeforeCleanup)
        #expect(!observation.isAlive)
    }

    @Test func diagnosticsEmitHitMissEvictionAndRejectionWithoutPaths() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let first = try fixture.directory(named: "first", seed: "first")
        let second = try fixture.directory(named: "second", seed: "second")
        let sink = TokenizerDiagnosticSink()
        let recorder = SwamaDiagnosticRecorder(
            enabled: true,
            primaryWrite: { sink.append($0) },
            fallbackWrite: { sink.append($0) }
        )
        recorder.start(mode: .cli)
        let previous = SwamaDiagnostics.installRecorderForTesting(recorder)
        defer { SwamaDiagnostics.restoreRecorderForTesting(previous) }

        let loader = CountingTokenizerLoader()
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 1, maximumInputBytes: 1_000_000),
            diagnostics: .live
        )
        _ = try await cache.load(from: first, using: loader) // miss
        _ = try await cache.load(from: first, using: loader) // hit
        _ = try await cache.load(from: second, using: loader) // miss + budget eviction
        cache.purge() // explicit eviction

        let rejectingCache = TokenizerCache(
            configuration: .init(maximumEntries: 1, maximumInputBytes: 1),
            diagnostics: .live
        )
        _ = try await rejectingCache.load(from: first, using: loader)
        recorder.stop(outcome: .ok)
        recorder.flush()

        let events = try diagnosticEvents(from: sink.data)
        let tokenizerEvents = events.filter { $0.subsystem == "tokenizer" }
        #expect(tokenizerEvents.map(\.event) == [
            .tokenizerCacheMiss,
            .tokenizerCacheHit,
            .tokenizerCacheMiss,
            .tokenizerCacheEvicted,
            .tokenizerCacheEvicted,
            .tokenizerCacheRejected
        ])
        #expect(tokenizerEvents.allSatisfy { event in
            guard case let .string(fingerprint)? = event.data?["fingerprint"] else {
                return false
            }

            return fingerprint.count == 64 && !sink.text.contains(fixture.root.path)
        })
        #expect(SwamaDiagnostics.isValidSnapshot(sink.data))
    }

    @Test func diagnosticSinkFailureNeverChangesTheTokenizerResult() async throws {
        let fixture = try TokenizerFixture()
        defer { fixture.remove() }
        let directory = try fixture.directory(named: "diagnostic-failure")
        let fallback = TokenizerDiagnosticSink()
        let recorder = SwamaDiagnosticRecorder(
            enabled: true,
            primaryWrite: { _ in throw StubLoaderError.injected },
            fallbackWrite: { fallback.append($0) }
        )
        recorder.start(mode: .cli)
        let previous = SwamaDiagnostics.installRecorderForTesting(recorder)
        defer { SwamaDiagnostics.restoreRecorderForTesting(previous) }
        let loader = CountingTokenizerLoader()
        let cache = TokenizerCache(
            configuration: .init(maximumEntries: 1, maximumInputBytes: 1_000_000),
            diagnostics: .live
        )

        let miss = try await cache.load(from: directory, using: loader)
        let hit = try await cache.load(from: directory, using: loader)
        recorder.stop(outcome: .ok)
        recorder.flush()

        #expect((miss as? StubTokenizer)?.identifier == "load-1")
        #expect((hit as? StubTokenizer)?.identifier == "load-1")
        #expect(await loader.calls == 1)
        #expect(!fallback.data.isEmpty)
    }
}

// MARK: - StubLoaderError

private enum StubLoaderError: Error {
    case injected
    case invalidConfiguration
}

// MARK: - StubTokenizer

private struct StubTokenizer: MLXLMCommon.Tokenizer {
    let identifier: String

    func encode(text _: String, addSpecialTokens _: Bool) -> [Int] { [identifier.count] }
    func decode(tokenIds _: [Int], skipSpecialTokens _: Bool) -> String { identifier }
    func convertTokenToId(_: String) -> Int? { nil }
    func convertIdToToken(_: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

// MARK: - CountingTokenizerLoader

private actor CountingTokenizerLoader: TokenizerLoader {
    init(failuresRemaining: Int = 0, delay: Duration? = nil) {
        self.failuresRemaining = failuresRemaining
        self.delay = delay
    }

    var calls: Int { callCount }

    func load(from _: URL) async throws -> any MLXLMCommon.Tokenizer {
        callCount += 1
        let current = callCount
        if failuresRemaining > 0 {
            failuresRemaining -= 1
            throw StubLoaderError.injected
        }
        if let delay {
            try? await Task.sleep(for: delay)
        }
        return StubTokenizer(identifier: "load-\(current)")
    }

    private var callCount = 0
    private var failuresRemaining: Int
    private let delay: Duration?
}

// MARK: - ConfigurationValidatingTokenizerLoader

private actor ConfigurationValidatingTokenizerLoader: TokenizerLoader {
    var calls: Int { callCount }

    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        callCount += 1
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard (try? JSONSerialization.jsonObject(with: data)) != nil else {
            throw StubLoaderError.invalidConfiguration
        }

        return StubTokenizer(identifier: "load-\(callCount)")
    }

    private var callCount = 0
}

// MARK: - MutatingTokenizerLoader

private actor MutatingTokenizerLoader: TokenizerLoader {
    init(fileName: String) {
        self.fileName = fileName
    }

    var calls: Int { callCount }

    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        callCount += 1
        if callCount == 1 {
            try Data("mutated-during-load".utf8).write(to: directory.appendingPathComponent(fileName))
        }
        return StubTokenizer(identifier: "load-\(callCount)")
    }

    private let fileName: String
    private var callCount = 0
}

// MARK: - SuspendedTokenizerLoader

private actor SuspendedTokenizerLoader: TokenizerLoader {
    init(mutateFileName: String? = nil) {
        self.mutateFileName = mutateFileName
    }

    var calls: Int { callCount }
    var wasCancelled: Bool { observedCancellation }

    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        callCount += 1
        let current = callCount
        await withCheckedContinuation { continuation in
            if released {
                continuation.resume()
            }
            else {
                releaseContinuation = continuation
            }
        }
        observedCancellation = Task.isCancelled
        if let mutateFileName {
            try Data("mutated-by-producer".utf8).write(to: directory.appendingPathComponent(mutateFileName))
        }
        finished = true
        return StubTokenizer(identifier: "load-\(current)")
    }

    func waitUntilStarted() async {
        while callCount == 0 {
            await Task.yield()
        }
    }

    func release() {
        released = true
        let continuation = releaseContinuation
        releaseContinuation = nil
        continuation?.resume()
    }

    func waitUntilFinished() async {
        while !finished {
            await Task.yield()
        }
    }

    private let mutateFileName: String?
    private var callCount = 0
    private var released = false
    private var observedCancellation = false
    private var finished = false
    private var releaseContinuation: CheckedContinuation<Void, Never>?
}

// MARK: - TokenizerLoadOutcome

private actor TokenizerLoadOutcome {
    enum Value: Equatable {
        case cancelled
        case otherError
        case value
    }

    var value: Value? { storedValue }

    func record(_ value: Value) {
        storedValue = value
    }

    private var storedValue: Value?
}

// MARK: - TokenizerLifetime

private final class TokenizerLifetime: @unchecked Sendable {}

// MARK: - TokenizerCleanupObservation

private final class TokenizerCleanupObservation: @unchecked Sendable {
    init(lifetime: TokenizerLifetime) {
        self.lifetime = lifetime
    }

    var isAlive: Bool { lock.withLock { lifetime != nil } }
    var wasReleasedBeforeCleanup: Bool { lock.withLock { releasedBeforeCleanup } }

    func recordCleanup() {
        lock.withLock {
            releasedBeforeCleanup = lifetime == nil
        }
    }

    private let lock: NSLock = .init()
    private weak var lifetime: TokenizerLifetime?
    private var releasedBeforeCleanup = false
}

// MARK: - LifetimeTokenizer

private struct LifetimeTokenizer: MLXLMCommon.Tokenizer {
    let lifetime: TokenizerLifetime

    func encode(text _: String, addSpecialTokens _: Bool) -> [Int] { [] }
    func decode(tokenIds _: [Int], skipSpecialTokens _: Bool) -> String { "" }
    func convertTokenToId(_: String) -> Int? { nil }
    func convertIdToToken(_: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

// MARK: - LifetimeTokenizerLoader

private actor LifetimeTokenizerLoader: TokenizerLoader {
    init(lifetime: TokenizerLifetime) {
        self.lifetime = lifetime
    }

    func load(from _: URL) async throws -> any MLXLMCommon.Tokenizer {
        let value = try #require(lifetime)
        lifetime = nil
        return LifetimeTokenizer(lifetime: value)
    }

    private var lifetime: TokenizerLifetime?
}

// MARK: - TokenizerDiagnosticSink

private final class TokenizerDiagnosticSink: @unchecked Sendable {
    var data: Data { lock.withLock { storage } }
    var text: String { String(decoding: data, as: UTF8.self) }

    func append(_ data: Data) {
        lock.withLock { storage.append(data) }
    }

    private let lock: NSLock = .init()
    private var storage: Data = .init()
}

// MARK: - TokenizerFixture

private struct TokenizerFixture {
    let root: URL

    init() throws {
        root = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-tokenizer-cache-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    func directory(named name: String, seed: String = "shared") throws -> URL {
        let directory = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for fileName in expectedTokenizerInputFileNames {
            let data = fileName == "config.json"
                ? Data(#"{"model_type":"\#(seed)-model"}"#.utf8)
                : Data("\(seed):\(fileName)".utf8)
            try data.write(to: directory.appendingPathComponent(fileName))
        }
        return directory
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }
}

private let expectedTokenizerInputFileNames = [
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

private func waitForOutcome(
    _ outcome: TokenizerLoadOutcome,
    timeout: Duration
) async throws {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: timeout)
    while await outcome.value == nil {
        guard clock.now < deadline else {
            throw TokenizerCacheTestError.timeout
        }

        try await Task.sleep(for: .milliseconds(2))
    }
}

// MARK: - TokenizerCacheTestError

private enum TokenizerCacheTestError: Error {
    case timeout
}

private func diagnosticEvents(from data: Data) throws -> [SwamaDiagnosticEvent] {
    switch SwamaDiagnosticTimeline.parse(data) {
    case let .valid(events):
        events
    case .degraded:
        throw StubLoaderError.injected
    case .unknown:
        throw StubLoaderError.injected
    }
}

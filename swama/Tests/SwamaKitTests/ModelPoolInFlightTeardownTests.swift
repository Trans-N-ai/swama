import Foundation
import MLX
import MLXEmbedders
@preconcurrency import MLXLMCommon
import MLXNN
@testable import SwamaKit
import Testing

// MARK: - ModelPoolInFlightTeardownTests

@Suite("ModelPool in-flight load teardown", .serialized)
struct ModelPoolInFlightTeardownTests {
    @Test
    func concurrentEmbeddingWaitersShareOneLoadAndBothSucceed() async throws {
        let modelName = "coalesced-embedding-waiters"
        let loader = ControlledLoader(
            first: makeLifetimeEmbeddingRunner(),
            next: makeLifetimeEmbeddingRunner()
        )
        let witness = CleanupWitness(nil)
        let pool = makePool(witness: witness, embeddingLoader: loader)

        let firstRequest = Task<Result<String, Error>, Never> {
            do {
                return try await .success(pool.runEmbeddingWithConcurrencyControl(modelName: modelName) { _ in
                    "first"
                })
            }
            catch {
                return .failure(error)
            }
        }
        await loader.waitUntilFirstLoadEntered()

        let secondRequest = Task<Result<String, Error>, Never> {
            do {
                return try await .success(pool.runEmbeddingWithConcurrencyControl(modelName: modelName) { _ in
                    "second"
                })
            }
            catch {
                return .failure(error)
            }
        }
        await waitUntilRunningInferenceCount(2, pool: pool)
        #expect(await loader.numberOfLoads == 1)

        await loader.releaseFirstLoad()
        for (result, expected) in await [(firstRequest.value, "first"), (secondRequest.value, "second")] {
            switch result {
            case let .success(value):
                #expect(value == expected)
            case let .failure(error):
                Issue.record("coalesced embedding waiter unexpectedly failed: \(error)")
            }
        }
        #expect(await pool.isCachedForTesting(.embedding, modelName: modelName))
        #expect(witness.cleanupReleaseStates.isEmpty)
    }

    @Test
    func currentLoadedEmbeddingDiscardedBehindExistingCacheCleansUpAfterRelease() async throws {
        let modelName = "embedding-cache-won-before-load"
        var loaded: EmbeddingRunner? = makeLifetimeEmbeddingRunner()
        let loader = try ControlledLoader(
            first: #require(loaded),
            next: makeLifetimeEmbeddingRunner()
        )
        let witness = CleanupWitness(loaded)
        loaded = nil
        let pool = makePool(witness: witness, embeddingLoader: loader)
        let cached = makeLifetimeEmbeddingRunner()

        let request = Task<Result<Bool, Error>, Never> {
            do {
                return try await .success(pool.runEmbeddingWithConcurrencyControl(modelName: modelName) { runner in
                    runner === cached
                })
            }
            catch {
                return .failure(error)
            }
        }

        await loader.waitUntilFirstLoadEntered()
        await pool.setEmbeddingRunner(cached, for: modelName)
        await loader.releaseFirstLoad()

        switch await request.value {
        case let .success(receivedExistingRunner):
            #expect(receivedExistingRunner)
        case let .failure(error):
            Issue.record("current load should use the existing cache value: \(error)")
        }
        #expect(witness.cleanupReleaseStates == [true])
        #expect(await pool.isCachedForTesting(.embedding, modelName: modelName))
    }

    @Test
    func newerEmbeddingLoadSurvivesOlderGenerationFinishingLate() async throws {
        let modelName = "overlapping-embedding-generations"
        var first: EmbeddingRunner? = makeLifetimeEmbeddingRunner()
        let loader = try ControlledLoader(
            first: #require(first),
            next: makeLifetimeEmbeddingRunner(),
            blockSecond: true
        )
        let witness = CleanupWitness(first)
        first = nil
        let pool = makePool(witness: witness, embeddingLoader: loader)

        let staleRequest = Task<Result<String, Error>, Never> {
            do {
                return try await .success(pool.runEmbeddingWithConcurrencyControl(modelName: modelName) { _ in
                    "stale"
                })
            }
            catch {
                return .failure(error)
            }
        }
        await loader.waitUntilFirstLoadEntered()
        await pool.clearCache()

        let freshRequest = Task<Result<String, Error>, Never> {
            do {
                return try await .success(pool.runEmbeddingWithConcurrencyControl(modelName: modelName) { _ in
                    "fresh"
                })
            }
            catch {
                return .failure(error)
            }
        }
        await loader.waitUntilSecondLoadEntered()

        await loader.releaseFirstLoad()
        switch await staleRequest.value {
        case .success:
            Issue.record("the older embedding generation unexpectedly reached its operation")
        case let .failure(error):
            #expect(error is CancellationError)
        }
        #expect(await pool.hasPendingLoadForTesting(.embedding, modelName: modelName))

        await loader.releaseSecondLoad()
        switch await freshRequest.value {
        case let .success(value):
            #expect(value == "fresh")
        case let .failure(error):
            Issue.record("newer embedding generation unexpectedly failed: \(error)")
        }
        #expect(await pool.isCachedForTesting(.embedding, modelName: modelName))
        #expect(await !(pool.hasPendingLoadForTesting(.embedding, modelName: modelName)))
        #expect(witness.cleanupReleaseStates == [false, true])
    }

    @Test
    func pendingLoadWithSameNameDoesNotLetEvictionInterruptALiveOperation() async throws {
        let modelName = "shared-modality-key"
        let languageLoader = ControlledLoader(
            first: makeLifetimeTestContainer(),
            next: makeLifetimeTestContainer()
        )
        await languageLoader.releaseFirstLoad()
        let embeddingLoader = ControlledLoader(
            first: makeLifetimeEmbeddingRunner(),
            next: makeLifetimeEmbeddingRunner()
        )
        let witness = CleanupWitness(nil)
        let pool = makePool(
            witness: witness,
            languageLoader: languageLoader,
            embeddingLoader: embeddingLoader
        )
        let operationBarrier = OperationBarrier()

        let languageRequest = Task<Result<String, Error>, Never> {
            do {
                return try await .success(pool.run(modelName: modelName) { _ in
                    await operationBarrier.enterAndWait()
                    return "language"
                })
            }
            catch {
                return .failure(error)
            }
        }
        await operationBarrier.waitUntilEntered()

        let embeddingRequest = Task<Result<String, Error>, Never> {
            do {
                return try await .success(pool.runEmbeddingWithConcurrencyControl(modelName: modelName) { _ in
                    "embedding"
                })
            }
            catch {
                return .failure(error)
            }
        }
        await embeddingLoader.waitUntilFirstLoadEntered()

        await pool.evictModelForTesting(modelName: modelName)
        #expect(await pool.isCachedForTesting(.language, modelName: modelName))
        #expect(await pool.hasPendingLoadForTesting(.embedding, modelName: modelName))
        #expect(witness.cleanupReleaseStates.isEmpty)

        await embeddingLoader.releaseFirstLoad()
        switch await embeddingRequest.value {
        case let .success(value):
            #expect(value == "embedding")
        case let .failure(error):
            Issue.record("pending embedding load unexpectedly failed: \(error)")
        }

        await operationBarrier.release()
        switch await languageRequest.value {
        case let .success(value):
            #expect(value == "language")
        case let .failure(error):
            Issue.record("active language operation unexpectedly failed: \(error)")
        }
    }

    @Test
    func periodicEvictionStillSkipsALiveInferenceOperation() async throws {
        let modelName = "active-language-operation"
        let loader = ControlledLoader(
            first: makeLifetimeTestContainer(),
            next: makeLifetimeTestContainer()
        )
        await loader.releaseFirstLoad()
        let witness = CleanupWitness(nil)
        let pool = makePool(witness: witness, languageLoader: loader)
        let operationBarrier = OperationBarrier()

        let request = Task<Result<String, Error>, Never> {
            do {
                return try await .success(pool.run(modelName: modelName) { _ in
                    await operationBarrier.enterAndWait()
                    return "finished"
                })
            }
            catch {
                return .failure(error)
            }
        }

        await operationBarrier.waitUntilEntered()
        #expect(await pool.isCachedForTesting(.language, modelName: modelName))
        await pool.evictModelForTesting(modelName: modelName)
        #expect(await pool.isCachedForTesting(.language, modelName: modelName))
        #expect(witness.cleanupReleaseStates.isEmpty)

        await operationBarrier.release()
        switch await request.value {
        case let .success(value):
            #expect(value == "finished")
        case let .failure(error):
            Issue.record("active operation unexpectedly failed: \(error)")
        }
    }

    @Test
    func cancelledLoadFailureStillRunsCleanupAfterItsPartialOwnerReleases() async throws {
        let modelName = "in-flight-failing-language"
        var partialOwner: PartialLoadOwner? = PartialLoadOwner()
        let loader = try FailingControlledLoader(partialOwner: #require(partialOwner))
        let witness = CleanupWitness(partialOwner)
        partialOwner = nil
        let pool = ModelPool(
            memoryHooks: .init(
                activeMemory: { 0 },
                clearCache: { witness.recordCleanup() }
            ),
            loadOverrides: .init(
                modelExistsLocally: { _ in true },
                determineIsVLM: { _ in false },
                loadLanguage: { _, _ in try await loader.load() },
                loadSpeechToText: { _ in fatalError("unexpected STT load") },
                loadTTS: { _, _, _ in fatalError("unexpected TTS load") },
                loadEmbedding: { _ in fatalError("unexpected embedding load") }
            )
        )

        let request = Task<Result<String, Error>, Never> {
            do {
                return try await .success(pool.run(modelName: modelName) { _ in "unexpected" })
            }
            catch {
                return .failure(error)
            }
        }

        await loader.waitUntilEntered()
        await pool.clearCache()
        #expect(await loader.isBlocked)
        await loader.releaseAndFail()

        switch await request.value {
        case .success:
            Issue.record("an invalidated failed load must not reach its operation")
        case let .failure(error):
            #expect(error is CancellationError)
        }
        #expect(witness.cleanupReleaseStates == [false, true])
    }

    @Test(arguments: TeardownPath.allCases)
    func languageLoadCannotResurrectAfterTeardown(_ path: TeardownPath) async throws {
        let modelName = "in-flight-language-\(path)"
        var first: MLXLMCommon.ModelContainer? = makeLifetimeTestContainer()
        let loader = try ControlledLoader(
            first: #require(first),
            next: makeLifetimeTestContainer()
        )
        let witness = CleanupWitness(first)
        first = nil
        let pool = makePool(witness: witness, languageLoader: loader)
        let operationCount = LockedCounter()

        let request = Task<Result<String, Error>, Never> {
            do {
                return try await .success(pool.run(modelName: modelName) { _ in
                    operationCount.increment()
                    return "stale"
                })
            }
            catch {
                return .failure(error)
            }
        }

        await loader.waitUntilFirstLoadEntered()
        #expect(await pool.hasPendingLoadForTesting(.language, modelName: modelName))
        #expect(await pool.hasMatchingLoadingOperationForTesting(modelName: modelName))
        await apply(path, to: pool, modelName: modelName)
        #expect(await loader.isFirstLoadBlocked)
        await loader.releaseFirstLoad()

        await assertStaleLoadWasDiscarded(
            request,
            pool: pool,
            kind: .language,
            modelName: modelName,
            operationCount: operationCount,
            witness: witness
        )

        let fresh = try await pool.run(modelName: modelName) { _ in "fresh" }
        #expect(fresh == "fresh")
        #expect(await pool.isCachedForTesting(.language, modelName: modelName))
    }

    @Test(arguments: TeardownPath.allCases)
    func speechToTextLoadCannotResurrectAfterTeardown(_ path: TeardownPath) async throws {
        let modelName = "in-flight-stt-\(path)"
        var first: SpeechToTextRunner? = await MainActor.run { SpeechToTextRunner() }
        let next = await MainActor.run { SpeechToTextRunner() }
        let loader = try ControlledLoader(first: #require(first), next: next)
        let witness = CleanupWitness(first)
        first = nil
        let pool = makePool(witness: witness, speechToTextLoader: loader)
        let operationCount = LockedCounter()

        let request = Task<Result<String, Error>, Never> {
            do {
                return try await .success(pool.runSpeechToText(modelName: modelName) { _ in
                    operationCount.increment()
                    return "stale"
                })
            }
            catch {
                return .failure(error)
            }
        }

        await loader.waitUntilFirstLoadEntered()
        #expect(await pool.hasPendingLoadForTesting(.speechToText, modelName: modelName))
        #expect(await pool.hasMatchingLoadingOperationForTesting(modelName: modelName))
        await apply(path, to: pool, modelName: modelName)
        #expect(await loader.isFirstLoadBlocked)
        await loader.releaseFirstLoad()

        await assertStaleLoadWasDiscarded(
            request,
            pool: pool,
            kind: .speechToText,
            modelName: modelName,
            operationCount: operationCount,
            witness: witness
        )

        let fresh = try await pool.runSpeechToText(modelName: modelName) { _ in "fresh" }
        #expect(fresh == "fresh")
        #expect(await pool.isCachedForTesting(.speechToText, modelName: modelName))
    }

    @Test(arguments: TeardownPath.allCases)
    func textToSpeechLoadCannotResurrectAfterTeardown(_ path: TeardownPath) async throws {
        let modelName = "in-flight-tts-\(path)"
        var first: TTSRunner? = TTSRunner(kind: .orpheus)
        let loader = try ControlledLoader(
            first: #require(first),
            next: TTSRunner(kind: .orpheus)
        )
        let witness = CleanupWitness(first)
        first = nil
        let pool = makePool(witness: witness, textToSpeechLoader: loader)
        let operationCount = LockedCounter()

        let request = Task<Result<String, Error>, Never> {
            do {
                return try await .success(pool.runTTS(modelKey: modelName, kind: .orpheus) { _ in
                    operationCount.increment()
                    return "stale"
                })
            }
            catch {
                return .failure(error)
            }
        }

        await loader.waitUntilFirstLoadEntered()
        #expect(await pool.hasPendingLoadForTesting(.textToSpeech, modelName: modelName))
        #expect(await pool.hasMatchingLoadingOperationForTesting(modelName: modelName))
        await apply(path, to: pool, modelName: modelName)
        #expect(await loader.isFirstLoadBlocked)
        await loader.releaseFirstLoad()

        await assertStaleLoadWasDiscarded(
            request,
            pool: pool,
            kind: .textToSpeech,
            modelName: modelName,
            operationCount: operationCount,
            witness: witness
        )

        let fresh = try await pool.runTTS(modelKey: modelName, kind: .orpheus) { _ in "fresh" }
        #expect(fresh == "fresh")
        #expect(await pool.isCachedForTesting(.textToSpeech, modelName: modelName))
    }

    @Test(arguments: TeardownPath.allCases)
    func embeddingLoadCannotResurrectAfterTeardown(_ path: TeardownPath) async throws {
        let modelName = "in-flight-embedding-\(path)"
        var first: EmbeddingRunner? = makeLifetimeEmbeddingRunner()
        let loader = try ControlledLoader(
            first: #require(first),
            next: makeLifetimeEmbeddingRunner()
        )
        let witness = CleanupWitness(first)
        first = nil
        let pool = makePool(witness: witness, embeddingLoader: loader)
        let operationCount = LockedCounter()

        let request = Task<Result<String, Error>, Never> {
            do {
                return try await .success(pool.runEmbeddingWithConcurrencyControl(modelName: modelName) { _ in
                    operationCount.increment()
                    return "stale"
                })
            }
            catch {
                return .failure(error)
            }
        }

        await loader.waitUntilFirstLoadEntered()
        #expect(await pool.hasPendingLoadForTesting(.embedding, modelName: modelName))
        #expect(await !(pool.hasMatchingLoadingOperationForTesting(modelName: modelName)))
        await apply(path, to: pool, modelName: modelName)
        #expect(await loader.isFirstLoadBlocked)
        await loader.releaseFirstLoad()

        await assertStaleLoadWasDiscarded(
            request,
            pool: pool,
            kind: .embedding,
            modelName: modelName,
            operationCount: operationCount,
            witness: witness
        )

        let fresh = try await pool.runEmbeddingWithConcurrencyControl(modelName: modelName) { _ in "fresh" }
        #expect(fresh == "fresh")
        #expect(await pool.isCachedForTesting(.embedding, modelName: modelName))
    }

    private func makePool(
        witness: CleanupWitness,
        languageLoader: ControlledLoader<MLXLMCommon.ModelContainer>? = nil,
        speechToTextLoader: ControlledLoader<SpeechToTextRunner>? = nil,
        textToSpeechLoader: ControlledLoader<TTSRunner>? = nil,
        embeddingLoader: ControlledLoader<EmbeddingRunner>? = nil
    ) -> ModelPool {
        ModelPool(
            memoryHooks: .init(
                activeMemory: { 0 },
                clearCache: { witness.recordCleanup() }
            ),
            loadOverrides: .init(
                modelExistsLocally: { _ in true },
                determineIsVLM: { _ in false },
                loadLanguage: { _, _ in
                    guard let languageLoader else {
                        fatalError("unexpected language load")
                    }

                    return await languageLoader.load()
                },
                loadSpeechToText: { _ in
                    guard let speechToTextLoader else {
                        fatalError("unexpected STT load")
                    }

                    return await speechToTextLoader.load()
                },
                loadTTS: { _, _, _ in
                    guard let textToSpeechLoader else {
                        fatalError("unexpected TTS load")
                    }

                    return await textToSpeechLoader.load()
                },
                loadEmbedding: { _ in
                    guard let embeddingLoader else {
                        fatalError("unexpected embedding load")
                    }

                    return await embeddingLoader.load()
                }
            )
        )
    }

    private func apply(_ path: TeardownPath, to pool: ModelPool, modelName: String) async {
        switch path {
        case .clear:
            await pool.clearCache()
        case .remove:
            await pool.remove(modelName: modelName)
        case .periodicEviction:
            await pool.evictModelForTesting(modelName: modelName)
        }
    }

    private func assertStaleLoadWasDiscarded(
        _ request: Task<Result<String, Error>, Never>,
        pool: ModelPool,
        kind: ModelPoolModelKind,
        modelName: String,
        operationCount: LockedCounter,
        witness: CleanupWitness
    ) async {
        switch await request.value {
        case .success:
            Issue.record("a load invalidated by teardown must not reach its operation")
        case let .failure(error):
            #expect(error is CancellationError, "expected CancellationError, got \(error)")
        }

        #expect(operationCount.value == 0)
        #expect(await !(pool.isCachedForTesting(kind, modelName: modelName)))
        #expect(await !(pool.hasPendingLoadForTesting(kind, modelName: modelName)))
        #expect(witness.cleanupReleaseStates == [false, true])
    }

    private func waitUntilRunningInferenceCount(_ expected: Int, pool: ModelPool) async {
        for _ in 0 ..< 1000 {
            if await pool.runningInferenceCountForTesting() == expected {
                return
            }
            await Task.yield()
        }
        Issue.record("timed out waiting for \(expected) running inference slots")
    }
}

// MARK: - TeardownPath

enum TeardownPath: String, CaseIterable, CustomStringConvertible, Sendable {
    case clear
    case remove
    case periodicEviction

    var description: String { rawValue }
}

// MARK: - ControlledLoader

private actor ControlledLoader<Value: AnyObject & Sendable> {
    init(first: Value, next: Value, blockSecond: Bool = false) {
        values = [first, next]
        self.blockSecond = blockSecond
    }

    func load() async -> Value {
        loadCount += 1
        precondition(!values.isEmpty, "test loader exhausted")
        let value = values.removeFirst()
        if loadCount == 1 {
            firstLoadEntered = true
            enteredWaiters.forEach { $0.resume() }
            enteredWaiters.removeAll()

            if !firstLoadReleased {
                await withCheckedContinuation { continuation in
                    releaseWaiter = continuation
                }
            }
        }
        else if loadCount == 2, blockSecond {
            secondLoadEntered = true
            secondEnteredWaiters.forEach { $0.resume() }
            secondEnteredWaiters.removeAll()

            if !secondLoadReleased {
                await withCheckedContinuation { continuation in
                    secondReleaseWaiter = continuation
                }
            }
        }

        return value
    }

    func waitUntilFirstLoadEntered() async {
        guard !firstLoadEntered else {
            return
        }

        await withCheckedContinuation { continuation in
            enteredWaiters.append(continuation)
        }
    }

    func releaseFirstLoad() {
        firstLoadReleased = true
        releaseWaiter?.resume()
        releaseWaiter = nil
    }

    func waitUntilSecondLoadEntered() async {
        guard !secondLoadEntered else {
            return
        }

        await withCheckedContinuation { continuation in
            secondEnteredWaiters.append(continuation)
        }
    }

    func releaseSecondLoad() {
        secondLoadReleased = true
        secondReleaseWaiter?.resume()
        secondReleaseWaiter = nil
    }

    var isFirstLoadBlocked: Bool {
        firstLoadEntered && !firstLoadReleased
    }

    var numberOfLoads: Int { loadCount }

    private var values: [Value]
    private let blockSecond: Bool
    private var loadCount = 0
    private var firstLoadEntered = false
    private var firstLoadReleased = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiter: CheckedContinuation<Void, Never>?
    private var secondLoadEntered = false
    private var secondLoadReleased = false
    private var secondEnteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var secondReleaseWaiter: CheckedContinuation<Void, Never>?
}

// MARK: - FailingControlledLoader

private actor FailingControlledLoader {
    init(partialOwner: PartialLoadOwner) {
        self.partialOwner = partialOwner
    }

    func load() async throws -> MLXLMCommon.ModelContainer {
        entered = true
        enteredWaiters.forEach { $0.resume() }
        enteredWaiters.removeAll()

        if !released {
            await withCheckedContinuation { continuation in
                releaseWaiter = continuation
            }
        }

        partialOwner = nil
        throw SyntheticLoadError()
    }

    func waitUntilEntered() async {
        guard !entered else {
            return
        }

        await withCheckedContinuation { continuation in
            enteredWaiters.append(continuation)
        }
    }

    func releaseAndFail() {
        released = true
        releaseWaiter?.resume()
        releaseWaiter = nil
    }

    var isBlocked: Bool { entered && !released }

    private var partialOwner: PartialLoadOwner?
    private var entered = false
    private var released = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiter: CheckedContinuation<Void, Never>?
}

// MARK: - OperationBarrier

private actor OperationBarrier {
    func enterAndWait() async {
        entered = true
        enteredWaiters.forEach { $0.resume() }
        enteredWaiters.removeAll()
        guard !released else {
            return
        }

        await withCheckedContinuation { continuation in
            releaseWaiter = continuation
        }
    }

    func waitUntilEntered() async {
        guard !entered else {
            return
        }

        await withCheckedContinuation { continuation in
            enteredWaiters.append(continuation)
        }
    }

    func release() {
        released = true
        releaseWaiter?.resume()
        releaseWaiter = nil
    }

    private var entered = false
    private var released = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiter: CheckedContinuation<Void, Never>?
}

// MARK: - SyntheticLoadError

private struct SyntheticLoadError: Error {}

// MARK: - PartialLoadOwner

private final class PartialLoadOwner: @unchecked Sendable {}

// MARK: - CleanupWitness

private final class CleanupWitness: @unchecked Sendable {
    init(_ object: AnyObject?) {
        trackedObject = object
    }

    func recordCleanup() {
        lock.withLock {
            cleanupStates.append(trackedObject == nil)
        }
    }

    var cleanupReleaseStates: [Bool] {
        lock.withLock { cleanupStates }
    }

    private let lock: NSLock = .init()
    private weak var trackedObject: AnyObject?
    private var cleanupStates: [Bool] = []
}

// MARK: - LockedCounter

private final class LockedCounter: @unchecked Sendable {
    func increment() {
        lock.withLock { count += 1 }
    }

    var value: Int {
        lock.withLock { count }
    }

    private let lock: NSLock = .init()
    private var count = 0
}

// MARK: - Minimal embedding value

private func makeLifetimeEmbeddingRunner() -> EmbeddingRunner {
    let context = EmbedderModelContext(
        configuration: .init(id: "in-flight-embedding-test"),
        model: LifetimeEmbeddingModel(),
        tokenizer: LifetimeTokenizer(),
        pooling: Pooling(strategy: .none)
    )
    return EmbeddingRunner(container: EmbedderModelContainer(context: context))
}

// MARK: - LifetimeEmbeddingModel

private final class LifetimeEmbeddingModel: Module, EmbeddingModel {
    let vocabularySize = 8
    let maxPositionEmbeddings: Int? = nil

    func callAsFunction(
        _: MLXArray,
        positionIds _: MLXArray?,
        tokenTypeIds _: MLXArray?,
        attentionMask _: MLXArray?
    ) -> EmbeddingModelOutput {
        fatalError("lifetime-only model must not run inference")
    }
}

// MARK: - LifetimeTokenizer

private struct LifetimeTokenizer: MLXLMCommon.Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text _: String, addSpecialTokens _: Bool) -> [Int] { [] }
    func decode(tokenIds _: [Int], skipSpecialTokens _: Bool) -> String { "" }
    func convertTokenToId(_: String) -> Int? { nil }
    func convertIdToToken(_: Int) -> String? { nil }
    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

import Foundation
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

// MARK: - ModelPoolError

/// Errors specific to ModelPool operations
public enum ModelPoolError: Error, LocalizedError {
    case modelNotFoundLocally(String)
    case failedToLoadModel(String, Error)

    public var errorDescription: String? {
        switch self {
        case let .modelNotFoundLocally(modelName):
            "Model '\(modelName)' not found locally. ModelPool only works with locally available models."
        case let .failedToLoadModel(modelName, underlyingError):
            "Failed to load model '\(modelName)': \(underlyingError.localizedDescription)"
        }
    }
}

// MARK: - ModelUsageInfo

/// Statistics for tracking model usage patterns
private struct ModelUsageInfo {
    var lastUsedTime: Date
    var usageCount: Int
    var loadTime: Date

    init() {
        let now = Date()
        self.lastUsedTime = now
        self.usageCount = 1
        self.loadTime = now
    }

    mutating func recordUsage() {
        self.lastUsedTime = Date()
        self.usageCount += 1
    }

    var idleTime: TimeInterval {
        Date().timeIntervalSince(lastUsedTime)
    }

    var ageTime: TimeInterval {
        Date().timeIntervalSince(loadTime)
    }
}

// MARK: - ModelPoolMemoryHooks

struct ModelPoolMemoryHooks: Sendable {
    let activeMemory: @Sendable () -> Int
    let clearCache: @Sendable () -> Void

    static let live: ModelPoolMemoryHooks = .init(
        activeMemory: { MLX.Memory.snapshot().activeMemory },
        clearCache: { MLX.Memory.clearCache() }
    )
}

// MARK: - ModelPoolLoadOverrides

/// Test-only seams for making each asynchronous load path deterministic without touching MLX.
/// Production pools leave this nil and use the real model factories below.
struct ModelPoolLoadOverrides: Sendable {
    let modelExistsLocally: @Sendable (String) -> Bool
    let determineIsVLM: @Sendable (String) -> Bool
    let loadLanguage: @Sendable (String, Bool) async throws -> MLXLMCommon.ModelContainer
    let loadSpeechToText: @Sendable (String) async throws -> SpeechToTextRunner
    let loadTTS: @Sendable (String, TTSModelKind, String?) async throws -> TTSRunner
    let loadEmbedding: @Sendable (String) async throws -> EmbeddingRunner
}

// MARK: - ModelLoadToken

private struct ModelLoadToken: Equatable, Sendable {
    let globalGeneration: UInt64
    let modelGeneration: UInt64
    let sequence: UInt64
}

// MARK: - LoadedValueOwner

/// A completed Task normally retains its success value. Keeping that value behind a releasable
/// owner lets a stale load drop the last Task-owned model reference *before* the follow-up MLX
/// cleanup runs.
private final class LoadedValueOwner<Value: AnyObject>: @unchecked Sendable {
    init(_ value: Value) {
        self.value = value
    }

    func retainedValue() -> Value? {
        lock.withLock { value }
    }

    @discardableResult
    func release() -> Bool {
        lock.withLock {
            guard value != nil else {
                return false
            }

            value = nil
            return true
        }
    }

    private let lock: NSLock = .init()
    private var value: Value?
}

// MARK: - PendingModelLoad

private struct PendingModelLoad<Value: AnyObject>: Sendable {
    init(
        token: ModelLoadToken,
        task: Task<LoadedValueOwner<Value>, Error>,
        diagnosticOperation: SwamaDiagnosticOperation,
        diagnosticPhases: ModelLoadPhaseRecorder
    ) {
        self.token = token
        self.task = task
        self.diagnosticOperation = diagnosticOperation
        self.diagnosticPhases = diagnosticPhases
    }

    let token: ModelLoadToken
    let task: Task<LoadedValueOwner<Value>, Error>
    let diagnosticOperation: SwamaDiagnosticOperation
    let diagnosticPhases: ModelLoadPhaseRecorder
    let postLoadCleanupGate: OneShotGate = .init()
    let diagnosticTerminalGate: OneShotGate = .init()
}

// MARK: - OneShotGate

private final class OneShotGate: @unchecked Sendable {
    func claim() -> Bool {
        lock.withLock {
            guard !claimed else {
                return false
            }

            claimed = true
            return true
        }
    }

    private let lock: NSLock = .init()
    private var claimed = false
}

// MARK: - ModelPoolModelKind

enum ModelPoolModelKind: CaseIterable, Sendable {
    case language
    case speechToText
    case textToSpeech
    case embedding
}

// MARK: - ModelPool

/// A pool to manage and cache `ModelContainer` instances with built-in concurrency control.
/// This helps in reusing already loaded models to save resources and time while preventing
/// MLX heap corruption through controlled concurrent access.
///
/// Features intelligent memory management with automatic model eviction based on idle time
/// to prevent GPU memory exhaustion when loading multiple models.
public actor ModelPool {
    // MARK: Lifecycle

    public init() {
        memoryHooks = .live
        loadOverrides = nil
        // Cache limit and memory management timer are both deferred: the former until MLX is
        // genuinely about to be used (see `ensureCacheLimitConfigured()`), the latter until
        // first model access (see `ensureMemoryManagementStarted()`). Neither may run here --
        // `configureCacheLimit()` touches `MLX.Memory`, which triggers MLX's Metal/device
        // library load, so calling it eagerly from `init()` would force that initialization
        // just to construct a `ModelPool`, which is what made `ModelPool` impossible to
        // construct under `swift test` (no colocated `mlx.metallib` for the test binary).
    }

    init(
        memoryHooks: ModelPoolMemoryHooks,
        loadOverrides: ModelPoolLoadOverrides? = nil
    ) {
        self.memoryHooks = memoryHooks
        self.loadOverrides = loadOverrides
    }

    /// Ensures memory management timer is running (called on first model access)
    private func ensureMemoryManagementStarted() {
        guard memoryManagementTask == nil else {
            return
        }

        startMemoryManagementTimer()
    }

    /// Ensures the MLX cache limit has been configured. Idempotent -- safe to call from every
    /// site that's about to actually load a model.
    ///
    /// Must only be called immediately before code that will genuinely touch MLX (i.e. right
    /// before loading a model container/runner), never from MLX-free paths like
    /// `ensureMemoryManagementStarted()` or `getContainer`'s cache/task lookups -- those must
    /// stay MLX-free so a test can reach `ModelPoolError.modelNotFoundLocally` without
    /// initializing Metal.
    private func ensureCacheLimitConfigured() {
        guard !didConfigureCacheLimit else {
            return
        }

        didConfigureCacheLimit = true
        if loadOverrides == nil {
            Self.configureCacheLimit()
        }
    }

    /// Whether `configureCacheLimit()` has already run for this pool.
    private var didConfigureCacheLimit = false

    private static func configureCacheLimit() {
        let ramBytes = ProcessInfo.processInfo.physicalMemory
        let maxCacheBytes: UInt64 = 8 * 1024 * 1024 * 1024
        let limit = min(maxCacheBytes, ramBytes / 10)
        if limit > 0 {
            MLX.Memory.cacheLimit = Int(limit)
        }
    }

    // MARK: Public

    public static let shared: ModelPool = .init()

    // MARK: - Speech Recognition Support

    /// Safely run a speech-to-text transcription operation with caching and concurrency control
    public func runSpeechToText<T: Sendable>(
        modelName: String,
        operation: @Sendable @escaping (SpeechToTextRunner) async throws -> T
    ) async throws -> T {
        // Wait for available slot AND ensure the specific model is not already running
        while runningInferences >= maxConcurrentInferences || runningModels[modelName] != nil {
            try await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }

        try Task.checkCancellation()

        runningInferences += 1
        let operationToken = beginModelOperation(modelName)
        defer {
            runningInferences = max(0, runningInferences - 1)
            endModelOperation(modelName, token: operationToken)
        }

        // Get or load the speech-to-text runner
        let runner = try await getSpeechToTextRunner(modelName: modelName)
        finishLoadingPhase(modelName, token: operationToken)

        // Execute the operation
        return try await operation(runner)
    }

    /// Safely run a TTS operation with caching and concurrency control
    public func runTTS<T: Sendable>(
        modelKey: String,
        kind: TTSModelKind,
        repository: String? = nil,
        operation: @Sendable @escaping (TTSRunner) async throws -> T
    ) async throws -> T {
        while runningInferences >= maxConcurrentInferences || runningModels[modelKey] != nil {
            try await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }

        try Task.checkCancellation()

        runningInferences += 1
        let operationToken = beginModelOperation(modelKey)
        defer {
            runningInferences = max(0, runningInferences - 1)
            endModelOperation(modelKey, token: operationToken)
        }

        let runner = try await getTTSRunner(
            modelKey: modelKey,
            kind: kind,
            repository: repository
        )
        finishLoadingPhase(modelKey, token: operationToken)
        return try await operation(runner)
    }

    /// Gets or loads a speech-to-text runner for the given model name
    private func getSpeechToTextRunner(modelName: String) async throws -> SpeechToTextRunner {
        // Ensure memory management is started
        ensureMemoryManagementStarted()

        if let runner = sttRunnerCache[modelName] {
            // Record usage for existing cached runner
            modelUsageInfo[modelName]?.recordUsage()
            return runner
        }

        if let pending = sttTasks[modelName] {
            return try await finishSpeechToTextLoad(pending, modelName: modelName)
        }

        // Check if we need to free up memory before loading a new model
        await performMemoryPressureCheck()

        // About to genuinely touch MLX -- configure the cache limit now, idempotently.
        ensureCacheLimitConfigured()

        let token = makeLoadToken(for: modelName)
        let diagnosticPhases = ModelLoadPhaseRecorder()
        let diagnosticOperation = SwamaDiagnostics.startModelLoad(model: modelName)
        let task = Task { [self] in
            let runner = try await loadSpeechToTextRunner(modelName: modelName)
            return LoadedValueOwner(runner)
        }
        let pending = PendingModelLoad(
            token: token,
            task: task,
            diagnosticOperation: diagnosticOperation,
            diagnosticPhases: diagnosticPhases
        )
        sttTasks[modelName] = pending
        return try await finishSpeechToTextLoad(pending, modelName: modelName)
    }

    private func setSpeechToTextRunner(_ runner: SpeechToTextRunner, forKey modelName: String) {
        sttRunnerCache[modelName] = runner
        modelUsageInfo[modelName] = ModelUsageInfo()
    }

    /// Gets or loads a TTS runner for the given model key
    private func getTTSRunner(
        modelKey: String,
        kind: TTSModelKind,
        repository: String?
    ) async throws -> TTSRunner {
        ensureMemoryManagementStarted()

        if let runner = ttsRunnerCache[modelKey] {
            modelUsageInfo[modelKey]?.recordUsage()
            return runner
        }

        if let pending = ttsTasks[modelKey] {
            return try await finishTTSLoad(pending, modelKey: modelKey)
        }

        await performMemoryPressureCheck()

        // About to genuinely touch MLX -- configure the cache limit now, idempotently.
        ensureCacheLimitConfigured()

        let token = makeLoadToken(for: modelKey)
        let diagnosticPhases = ModelLoadPhaseRecorder()
        let diagnosticOperation = SwamaDiagnostics.startModelLoad(model: modelKey)
        let task = Task { [self] in
            let runner = try await loadTTSRunner(
                modelKey: modelKey,
                kind: kind,
                repository: repository
            )
            return LoadedValueOwner(runner)
        }

        let pending = PendingModelLoad(
            token: token,
            task: task,
            diagnosticOperation: diagnosticOperation,
            diagnosticPhases: diagnosticPhases
        )
        ttsTasks[modelKey] = pending
        return try await finishTTSLoad(pending, modelKey: modelKey)
    }

    private func setTTSRunner(_ runner: TTSRunner, forKey modelKey: String) {
        ttsRunnerCache[modelKey] = runner
        modelUsageInfo[modelKey] = ModelUsageInfo()
    }

    // MARK: - Memory Management Configuration

    /// Maximum idle time before a model becomes eligible for eviction (5 minutes for production)
    private let maxIdleTime: TimeInterval = 5 * 60

    /// Interval for checking and evicting idle models (1 minute for production)
    private let memoryCheckInterval: TimeInterval = 60

    /// Maximum number of models to keep in cache before triggering aggressive cleanup
    private let maxCacheSize = 4

    /// Limit idle evictions per cleanup cycle to reduce churn
    private let maxIdleEvictionsPerCycle = 1

    /// Task for periodic memory management
    private var memoryManagementTask: Task<Void, Never>?

    // MARK: - Concurrency Control

    private var runningInferences = 0
    private let maxConcurrentInferences = 3 // Optimal for high-performance machines

    /// Per-model concurrency control: map each reserved model to its owning operation token.
    private var runningModels: [String: UInt64] = [:]

    /// The same operation token remains here only until its model load finishes. Matching tokens
    /// are an explicit credential for eviction to cancel a load without touching a live operation.
    private var loadingModels: [String: UInt64] = [:]
    private var nextOperationSequence: UInt64 = 0

    private func beginModelOperation(_ modelName: String) -> UInt64 {
        nextOperationSequence &+= 1
        let token = nextOperationSequence
        runningModels[modelName] = token
        loadingModels[modelName] = token
        return token
    }

    private func finishLoadingPhase(_ modelName: String, token: UInt64) {
        if loadingModels[modelName] == token {
            loadingModels.removeValue(forKey: modelName)
        }
    }

    private func endModelOperation(_ modelName: String, token: UInt64) {
        finishLoadingPhase(modelName, token: token)
        if runningModels[modelName] == token {
            runningModels.removeValue(forKey: modelName)
        }
    }

    /// Safely run a model operation with concurrency control to prevent MLX heap corruption
    public func run<T: Sendable>(
        modelName: String,
        operation: @Sendable @escaping (ModelRunner) async throws -> T
    ) async throws -> T {
        // Wait for available slot AND ensure the specific model is not already running
        while runningInferences >= maxConcurrentInferences || runningModels[modelName] != nil {
            try await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }

        try Task.checkCancellation()

        runningInferences += 1
        let operationToken = beginModelOperation(modelName)
        defer {
            runningInferences = max(0, runningInferences - 1)
            endModelOperation(modelName, token: operationToken)
        }

        // Get or load the model container
        let container = try await getContainer(modelName: modelName)
        finishLoadingPhase(modelName, token: operationToken)
        let runner = ModelRunner(container: container)

        // Execute the operation
        return try await operation(runner)
    }

    /// Safely run an embedding operation with concurrency control to prevent MLX heap corruption
    public func runEmbeddingWithConcurrencyControl<T: Sendable>(
        modelName: String,
        operation: @Sendable @escaping (EmbeddingRunner) async throws -> T
    ) async throws -> T {
        // Embedding requests share the global pool limit but deliberately do not reserve
        // `runningModels`: EmbeddingRunner is an actor and EmbedderModelContainer serializes
        // access internally, so same-model callers may coalesce one load and then queue there.
        while runningInferences >= maxConcurrentInferences {
            try await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }

        try Task.checkCancellation()

        runningInferences += 1
        defer {
            runningInferences = max(0, runningInferences - 1)
        }

        let runner = try await getOrLoadEmbeddingRunner(modelName: modelName)

        // Execute the operation
        return try await operation(runner)
    }

    /// Gets or loads a ModelContainer
    private func getContainer(modelName: String) async throws -> MLXLMCommon.ModelContainer {
        // Ensure memory management is started
        ensureMemoryManagementStarted()

        if let container = cache[modelName] {
            // Record usage for existing cached model
            modelUsageInfo[modelName]?.recordUsage()
            return container
        }

        if let pending = tasks[modelName] {
            return try await finishLanguageLoad(pending, modelName: modelName)
        }

        // Check if we need to free up memory before loading a new model
        await performMemoryPressureCheck()

        guard modelExistsLocally(modelName) else {
            modelTypeCache.removeValue(forKey: modelName)
            throw ModelPoolError.modelNotFoundLocally(modelName)
        }

        let isVLM = modelTypeCache[modelName] ?? determineModelType(modelName)
        modelTypeCache[modelName] = isVLM

        let token = makeLoadToken(for: modelName)
        let diagnosticPhases = ModelLoadPhaseRecorder()
        let diagnosticOperation = SwamaDiagnostics.startModelLoad(model: modelName)
        let task = Task { [self] in
            let container = try await loadLanguageContainer(
                modelName: modelName,
                isVLM: isVLM,
                diagnosticPhases: diagnosticPhases
            )
            return LoadedValueOwner(container)
        }
        let pending = PendingModelLoad(
            token: token,
            task: task,
            diagnosticOperation: diagnosticOperation,
            diagnosticPhases: diagnosticPhases
        )
        tasks[modelName] = pending
        return try await finishLanguageLoad(pending, modelName: modelName)
    }

    /// Gets or loads an embedding model runner for the given model name.
    public func getEmbeddingRunner(for modelName: String) async -> EmbeddingRunner? {
        embeddingRunnerCache[modelName]
    }

    /// Sets an embedding model runner for the given model name.
    public func setEmbeddingRunner(_ runner: EmbeddingRunner, for modelName: String) async {
        embeddingRunnerCache[modelName] = runner
    }

    /// Clears the entire model cache and cancels any ongoing loading tasks.
    public func clearCache() {
        let memoryBefore = memoryHooks.activeMemory()
        let evictedCount = cache.count
        let modelNames = Set(cache.keys)
            .union(tasks.keys)
            .union(embeddingRunnerCache.keys)
            .union(embeddingTasks.keys)
            .union(sttRunnerCache.keys)
            .union(sttTasks.keys)
            .union(ttsRunnerCache.keys)
            .union(ttsTasks.keys)
        let diagnosticOperations = Dictionary(uniqueKeysWithValues: modelNames.map {
            ($0, SwamaDiagnostics.startEviction(model: $0))
        })

        invalidateAllLoads()

        // Cancel loading work before dropping the task dictionaries. Do not snapshot their
        // values into local arrays: completed Task values can retain loaded model containers.
        for pending in tasks.values {
            pending.task.cancel()
        }
        for pending in embeddingTasks.values {
            pending.task.cancel()
        }
        for pending in sttTasks.values {
            pending.task.cancel()
        }
        for pending in ttsTasks.values {
            pending.task.cancel()
        }

        // Release every owner before asking MLX to return unused allocations. The previous
        // implementation copied `cache.values` into `containersToEvict`, then captured that
        // array in an asynchronous cleanup Task. MLX therefore cleared while all model weights
        // were still strongly retained; after the Task released them, no second clear occurred.
        cache.removeAll()
        tasks.removeAll()
        embeddingTasks.removeAll()
        sttTasks.removeAll()
        ttsTasks.removeAll()
        modelTypeCache.removeAll()
        vlmRegistryCache = nil
        embeddingRunnerCache.removeAll()
        PromptCacheStore.shared.dropAll()
        sttRunnerCache.removeAll()
        ttsRunnerCache.removeAll()
        modelUsageInfo.removeAll()

        // Synchronous completion is part of the contract: when clearCache returns, model owners
        // are gone and MLX cleanup has run. Callers no longer race a detached cleanup Task.
        memoryHooks.clearCache()

        let memoryAfter = memoryHooks.activeMemory()
        let memoryReleased = max(0, memoryBefore - memoryAfter)
        let residentBytesAfter = SwamaDiagnostics.residentBytes()
        for (modelName, operation) in diagnosticOperations {
            SwamaDiagnostics.completeEviction(
                operation,
                model: modelName,
                generation: globalLoadGeneration,
                residentBytesAfter: residentBytesAfter
            )
        }
        NSLog(
            "SwamaKit.ModelPool: Cache cleared (\(evictedCount) models). Released \(memoryReleased / (1024 * 1024))MB active memory. Active: \(memoryAfter / (1024 * 1024))MB"
        )
    }

    /// Removes a specific model from the cache and cancels its loading task if active.
    public func remove(modelName: String) {
        let hadEntry = cache[modelName] != nil
            || tasks[modelName] != nil
            || embeddingRunnerCache[modelName] != nil
            || embeddingTasks[modelName] != nil
            || sttRunnerCache[modelName] != nil
            || sttTasks[modelName] != nil
            || ttsRunnerCache[modelName] != nil
            || ttsTasks[modelName] != nil
        let diagnosticOperation = hadEntry ? SwamaDiagnostics.startEviction(model: modelName) : nil
        invalidateLoads(for: modelName)

        cache.removeValue(forKey: modelName)
        modelTypeCache.removeValue(forKey: modelName) // Clear type cache for this model
        embeddingRunnerCache.removeValue(forKey: modelName) // Clear embedding cache for this model
        PromptCacheStore.shared.drop(modelName: modelName) // Clear prompt/KV-cache slot for this model
        sttRunnerCache.removeValue(forKey: modelName) // Clear speech-to-text cache for this model
        ttsRunnerCache.removeValue(forKey: modelName) // Clear TTS cache for this model
        modelUsageInfo.removeValue(forKey: modelName) // Clear usage tracking for this model

        if let pending = tasks.removeValue(forKey: modelName) {
            pending.task.cancel()
        }

        if let pending = embeddingTasks.removeValue(forKey: modelName) {
            pending.task.cancel()
        }

        if let pending = sttTasks.removeValue(forKey: modelName) {
            pending.task.cancel()
        }

        if let pending = ttsTasks.removeValue(forKey: modelName) {
            pending.task.cancel()
        }

        // All strong owners have been removed before MLX cleanup runs.
        memoryHooks.clearCache()
        if let diagnosticOperation {
            SwamaDiagnostics.completeEviction(
                diagnosticOperation,
                model: modelName,
                generation: modelLoadGenerations[modelName, default: 0],
                residentBytesAfter: SwamaDiagnostics.residentBytes()
            )
        }
    }

    #if DEBUG
        func cacheContainerForTesting(_ container: MLXLMCommon.ModelContainer, modelName: String) {
            cache[modelName] = container
        }

        func evictModelForTesting(modelName: String) async {
            await evictModel(modelName: modelName, reason: "test")
        }

        func isCachedForTesting(_ kind: ModelPoolModelKind, modelName: String) -> Bool {
            switch kind {
            case .language:
                cache[modelName] != nil
            case .speechToText:
                sttRunnerCache[modelName] != nil
            case .textToSpeech:
                ttsRunnerCache[modelName] != nil
            case .embedding:
                embeddingRunnerCache[modelName] != nil
            }
        }

        func hasPendingLoadForTesting(_ kind: ModelPoolModelKind, modelName: String) -> Bool {
            switch kind {
            case .language:
                tasks[modelName] != nil
            case .speechToText:
                sttTasks[modelName] != nil
            case .textToSpeech:
                ttsTasks[modelName] != nil
            case .embedding:
                embeddingTasks[modelName] != nil
            }
        }

        func hasMatchingLoadingOperationForTesting(modelName: String) -> Bool {
            guard let runningOwner = runningModels[modelName] else {
                return false
            }

            return loadingModels[modelName] == runningOwner
        }

        func runningInferenceCountForTesting() -> Int {
            runningInferences
        }
    #endif

    // MARK: Private

    private var cache: [String: MLXLMCommon.ModelContainer] = .init()
    private var tasks: [String: PendingModelLoad<MLXLMCommon.ModelContainer>] = .init()
    private var embeddingRunnerCache: [String: EmbeddingRunner] = .init()
    private var embeddingTasks: [String: PendingModelLoad<EmbeddingRunner>] = .init()
    private var sttRunnerCache: [String: SpeechToTextRunner] = .init()
    private var sttTasks: [String: PendingModelLoad<SpeechToTextRunner>] = .init()
    private var ttsRunnerCache: [String: TTSRunner] = .init()
    private var ttsTasks: [String: PendingModelLoad<TTSRunner>] = .init()

    /// Every teardown changes either the global generation or the named model generation.
    /// A load may ignore cooperative cancellation, but it cannot commit across this boundary.
    private var globalLoadGeneration: UInt64 = 0
    private var modelLoadGenerations: [String: UInt64] = .init()
    private var nextLoadSequence: UInt64 = 0

    /// Memory management tracking
    private var modelUsageInfo: [String: ModelUsageInfo] = .init()

    private var modelTypeCache: [String: Bool] = .init()
    private var vlmRegistryCache: [String: MLXLMCommon.ModelConfiguration]?
    private let memoryHooks: ModelPoolMemoryHooks
    private let loadOverrides: ModelPoolLoadOverrides?

    private func makeLoadToken(for modelName: String) -> ModelLoadToken {
        nextLoadSequence &+= 1
        return ModelLoadToken(
            globalGeneration: globalLoadGeneration,
            modelGeneration: modelLoadGenerations[modelName, default: 0],
            sequence: nextLoadSequence
        )
    }

    private func invalidateAllLoads() {
        globalLoadGeneration &+= 1
        modelLoadGenerations.removeAll()
    }

    private func invalidateLoads(for modelName: String) {
        modelLoadGenerations[modelName, default: 0] &+= 1
    }

    private func isCurrent(_ token: ModelLoadToken, modelName: String) -> Bool {
        token.globalGeneration == globalLoadGeneration
            && token.modelGeneration == modelLoadGenerations[modelName, default: 0]
    }

    private func resolvedLoadedValue<Value: AnyObject>(
        _ owner: LoadedValueOwner<Value>,
        token: ModelLoadToken,
        modelName: String,
        postLoadCleanupGate: OneShotGate
    ) throws -> Value {
        guard isCurrent(token, modelName: modelName) else {
            // clear/remove/evict already ran one cleanup while this owner was still live. Drop
            // the Task-retained value first, then run the bounded follow-up cleanup exactly once.
            owner.release()
            if postLoadCleanupGate.claim() {
                memoryHooks.clearCache()
            }
            throw CancellationError()
        }
        guard let value = owner.retainedValue() else {
            throw CancellationError()
        }

        return value
    }

    private func finishLanguageLoad(
        _ pending: PendingModelLoad<MLXLMCommon.ModelContainer>,
        modelName: String
    ) async throws -> MLXLMCommon.ModelContainer {
        try await finishLoad(
            pending,
            modelName: modelName,
            currentPendingToken: { tasks[modelName]?.token },
            removePending: { tasks.removeValue(forKey: modelName) },
            cachedValue: { cache[modelName] },
            storeLoaded: { loaded in
                cache[modelName] = loaded
                modelUsageInfo[modelName] = ModelUsageInfo()
            },
            recordCachedUse: {
                modelUsageInfo[modelName]?.recordUsage()
            }
        )
    }

    private func finishSpeechToTextLoad(
        _ pending: PendingModelLoad<SpeechToTextRunner>,
        modelName: String
    ) async throws -> SpeechToTextRunner {
        try await finishLoad(
            pending,
            modelName: modelName,
            currentPendingToken: { sttTasks[modelName]?.token },
            removePending: { sttTasks.removeValue(forKey: modelName) },
            cachedValue: { sttRunnerCache[modelName] },
            storeLoaded: { loaded in
                setSpeechToTextRunner(loaded, forKey: modelName)
            },
            recordCachedUse: {
                modelUsageInfo[modelName]?.recordUsage()
            }
        )
    }

    private func finishTTSLoad(
        _ pending: PendingModelLoad<TTSRunner>,
        modelKey: String
    ) async throws -> TTSRunner {
        try await finishLoad(
            pending,
            modelName: modelKey,
            currentPendingToken: { ttsTasks[modelKey]?.token },
            removePending: { ttsTasks.removeValue(forKey: modelKey) },
            cachedValue: { ttsRunnerCache[modelKey] },
            storeLoaded: { loaded in
                setTTSRunner(loaded, forKey: modelKey)
            },
            recordCachedUse: {
                modelUsageInfo[modelKey]?.recordUsage()
            }
        )
    }

    private func getOrLoadEmbeddingRunner(modelName: String) async throws -> EmbeddingRunner {
        if let runner = embeddingRunnerCache[modelName] {
            return runner
        }

        if let pending = embeddingTasks[modelName] {
            return try await finishEmbeddingLoad(pending, modelName: modelName)
        }

        ensureCacheLimitConfigured()
        let token = makeLoadToken(for: modelName)
        let diagnosticPhases = ModelLoadPhaseRecorder()
        let diagnosticOperation = SwamaDiagnostics.startModelLoad(model: modelName)
        let task = Task { [self] in
            let runner = try await loadEmbeddingRunner(
                modelName: modelName,
                diagnosticPhases: diagnosticPhases
            )
            return LoadedValueOwner(runner)
        }
        let pending = PendingModelLoad(
            token: token,
            task: task,
            diagnosticOperation: diagnosticOperation,
            diagnosticPhases: diagnosticPhases
        )
        embeddingTasks[modelName] = pending
        return try await finishEmbeddingLoad(pending, modelName: modelName)
    }

    private func finishEmbeddingLoad(
        _ pending: PendingModelLoad<EmbeddingRunner>,
        modelName: String
    ) async throws -> EmbeddingRunner {
        try await finishLoad(
            pending,
            modelName: modelName,
            currentPendingToken: { embeddingTasks[modelName]?.token },
            removePending: { embeddingTasks.removeValue(forKey: modelName) },
            cachedValue: { embeddingRunnerCache[modelName] },
            storeLoaded: { embeddingRunnerCache[modelName] = $0 },
            recordCachedUse: {}
        )
    }

    private func finishLoad<Value: AnyObject>(
        _ pending: PendingModelLoad<Value>,
        modelName: String,
        currentPendingToken: () -> ModelLoadToken?,
        removePending: () -> Void,
        cachedValue: () -> Value?,
        storeLoaded: (Value) -> Void,
        recordCachedUse: () -> Void
    ) async throws -> Value {
        do {
            let owner = try await pending.task.value

            guard isCurrent(pending.token, modelName: modelName) else {
                return try resolvedLoadedValue(
                    owner,
                    token: pending.token,
                    modelName: modelName,
                    postLoadCleanupGate: pending.postLoadCleanupGate
                )
            }

            if let existing = cachedValue() {
                recordCachedUse()
                // Another waiter or an explicit cache setter won the commit. A completed Task
                // must not retain its unused load; if it owned a distinct value, clean up after
                // releasing it. Multiple waiters share this one-shot cleanup gate.
                if owner.release(), pending.postLoadCleanupGate.claim() {
                    memoryHooks.clearCache()
                }
                if currentPendingToken() == pending.token {
                    removePending()
                }
                completeLoadDiagnostics(pending, modelName: modelName)
                return existing
            }

            let loaded = try resolvedLoadedValue(
                owner,
                token: pending.token,
                modelName: modelName,
                postLoadCleanupGate: pending.postLoadCleanupGate
            )
            storeLoaded(loaded)
            owner.release()

            if currentPendingToken() == pending.token {
                removePending()
            }
            completeLoadDiagnostics(pending, modelName: modelName)
            return loaded
        }
        catch {
            let wasInvalidated = !isCurrent(pending.token, modelName: modelName)
            cleanupAfterStaleFailure(pending, modelName: modelName)
            if currentPendingToken() == pending.token {
                removePending()
            }
            finishLoadDiagnostics(
                pending,
                modelName: modelName,
                error: error,
                wasInvalidated: wasInvalidated
            )
            if wasInvalidated {
                throw CancellationError()
            }
            throw error
        }
    }

    private func modelExistsLocally(_ modelName: String) -> Bool {
        loadOverrides?.modelExistsLocally(modelName)
            ?? ModelPaths.modelExistsLocally(modelName)
    }

    private func cleanupAfterStaleFailure(
        _ pending: PendingModelLoad<some AnyObject>,
        modelName: String
    ) {
        guard !isCurrent(pending.token, modelName: modelName),
              pending.postLoadCleanupGate.claim()
        else {
            return
        }

        memoryHooks.clearCache()
    }

    private func completeLoadDiagnostics(
        _ pending: PendingModelLoad<some AnyObject>,
        modelName: String
    ) {
        guard pending.diagnosticTerminalGate.claim() else {
            return
        }

        SwamaDiagnostics.completeModelLoad(
            pending.diagnosticOperation,
            model: modelName,
            phases: pending.diagnosticPhases.phases
        )
    }

    private func finishLoadDiagnostics(
        _ pending: PendingModelLoad<some AnyObject>,
        modelName: String,
        error: Error,
        wasInvalidated: Bool
    ) {
        guard pending.diagnosticTerminalGate.claim() else {
            return
        }

        if wasInvalidated || error is CancellationError {
            SwamaDiagnostics.cancelModelLoad(
                pending.diagnosticOperation,
                model: modelName,
                phases: pending.diagnosticPhases.phases
            )
            return
        }

        let code: SwamaDiagnosticErrorCode =
            if let poolError = error as? ModelPoolError {
                switch poolError {
                case .modelNotFoundLocally:
                    .modelNotFound
                case .failedToLoadModel:
                    .modelLoadFailed
                }
            }
            else {
                .modelLoadFailed
            }
        SwamaDiagnostics.failModelLoad(
            pending.diagnosticOperation,
            model: modelName,
            code: code,
            phases: pending.diagnosticPhases.phases
        )
    }

    private func determineModelType(_ modelName: String) -> Bool {
        loadOverrides?.determineIsVLM(modelName)
            ?? determineIfVLMModel(modelName: modelName)
    }

    private func loadLanguageContainer(
        modelName: String,
        isVLM: Bool,
        diagnosticPhases: ModelLoadPhaseRecorder
    ) async throws -> MLXLMCommon.ModelContainer {
        if let load = loadOverrides?.loadLanguage {
            return try await load(modelName, isVLM)
        }
        return try await loadModelContainer(
            modelName: modelName,
            isVLM: isVLM,
            diagnosticPhases: diagnosticPhases
        )
    }

    private func loadSpeechToTextRunner(modelName: String) async throws -> SpeechToTextRunner {
        if let load = loadOverrides?.loadSpeechToText {
            return try await load(modelName)
        }
        let runner = await MainActor.run { SpeechToTextRunner() }
        try await runner.loadModel(modelName)
        return runner
    }

    private func loadTTSRunner(
        modelKey: String,
        kind: TTSModelKind,
        repository: String?
    ) async throws -> TTSRunner {
        if let load = loadOverrides?.loadTTS {
            return try await load(modelKey, kind, repository)
        }
        let runner = repository.map { TTSRunner(kind: kind, repository: $0) }
            ?? TTSRunner(kind: kind)
        try await runner.loadModel()
        return runner
    }

    private func loadEmbeddingRunner(
        modelName: String,
        diagnosticPhases: ModelLoadPhaseRecorder
    ) async throws -> EmbeddingRunner {
        if let load = loadOverrides?.loadEmbedding {
            return try await load(modelName)
        }
        let container = try await loadEmbeddingModelContainer(
            modelName: modelName,
            tokenizerLoader: DiagnosticTokenizerLoader(
                upstream: #huggingFaceTokenizerLoader(),
                phases: diagnosticPhases
            )
        )
        return EmbeddingRunner(container: container)
    }

    private func hasPendingLoad(for modelName: String) -> Bool {
        tasks[modelName] != nil
            || embeddingTasks[modelName] != nil
            || sttTasks[modelName] != nil
            || ttsTasks[modelName] != nil
    }

    /// Unified VLM detection logic with registry priority
    private func determineIfVLMModel(modelName: String) -> Bool {
        // Ensure VLM registry is initialized
        ensureVLMRegistryInitialized()

        if vlmRegistryCache!.keys.contains(where: { $0.caseInsensitiveCompare(modelName) == .orderedSame }) {
            return true
        }

        // Prefer local config-driven detection for multimodal models whose names do not
        // include "vl" (for example Qwen3.5 variants).
        if isVLMModelByLocalConfig(modelName) {
            return true
        }

        return isVLMModelByName(modelName)
    }

    private func loadModelContainer(
        modelName: String,
        isVLM: Bool,
        diagnosticPhases: ModelLoadPhaseRecorder
    ) async throws -> MLXLMCommon.ModelContainer {
        // About to genuinely touch MLX (loading a container triggers Metal init) -- configure
        // the cache limit now, idempotently, rather than eagerly in `init()`.
        ensureCacheLimitConfigured()

        ensureChatTemplateIfNeeded(for: modelName)
        // Configure extra EOS tokens for models with known issues
        let extraEOSTokens = getExtraEOSTokens(for: modelName)

        let localConfig = MLXLMCommon.ModelConfiguration(
            directory: ModelPaths.getModelDirectory(for: modelName),
            extraEOSTokens: extraEOSTokens
        )

        do {
            let tokenizerLoader = DiagnosticTokenizerLoader(
                upstream: #huggingFaceTokenizerLoader(),
                phases: diagnosticPhases
            )
            if isVLM {
                return try await VLMModelFactory.shared.loadContainer(
                    from: LocalOnlyModelDownloader(),
                    using: tokenizerLoader,
                    configuration: localConfig
                )
            }
            else {
                return try await LLMModelFactory.shared.loadContainer(
                    from: LocalOnlyModelDownloader(),
                    using: tokenizerLoader,
                    configuration: localConfig
                )
            }
        }
        catch {
            throw ModelPoolError.failedToLoadModel(modelName, error)
        }
    }

    /// Get extra EOS tokens for models with known tokenization issues
    private func getExtraEOSTokens(for modelName: String) -> Set<String> {
        let lowercaseName = modelName.lowercased()
        var tokens = detectEOSTokensFromModelFiles(modelName: modelName)

        if lowercaseName.contains("gemma") {
            tokens.insert("<end_of_turn>")
        }

        if lowercaseName.contains("qwen3-coder") {
            tokens.insert("<endoftext>")
        }

        if !tokens.isEmpty {
            let tokenList = Array(tokens)
            NSLog(
                "SwamaKit.ModelPool: extra EOS tokens for %@ -> %@",
                modelName,
                tokenList.joined(separator: ",")
            )
        }

        return tokens
    }

    private func detectEOSTokensFromModelFiles(modelName: String) -> Set<String> {
        let modelDirectory = ModelPaths.getModelDirectory(for: modelName)
        let configURL = modelDirectory.appendingPathComponent("config.json")
        let generationConfigURL = modelDirectory.appendingPathComponent("generation_config.json")
        let tokenizerURL = modelDirectory.appendingPathComponent("tokenizer.json")

        var eosTokenIDs: Set<Int> = []

        if let ids = parseEOSTokenIDs(from: configURL) {
            eosTokenIDs.formUnion(ids)
        }

        if let ids = parseEOSTokenIDs(from: generationConfigURL) {
            eosTokenIDs.formUnion(ids)
        }

        guard !eosTokenIDs.isEmpty,
              let tokens = mapTokenIDsToStrings(ids: eosTokenIDs, tokenizerURL: tokenizerURL)
        else {
            return []
        }

        return tokens
    }

    private func ensureChatTemplateIfNeeded(for modelName: String) {
        let modelDirectory = ModelPaths.getModelDirectory(for: modelName)
        let tokenizerConfigURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
        let templateURL = modelDirectory.appendingPathComponent("chat_template.jinja")

        guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
            return
        }
        guard let configData = try? Data(contentsOf: tokenizerConfigURL),
              var jsonObject = try? JSONSerialization.jsonObject(with: configData) as? [String: Any]
        else {
            return
        }

        if let existingTemplate = jsonObject["chat_template"] as? String,
           !existingTemplate.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return
        }

        guard FileManager.default.fileExists(atPath: templateURL.path),
              let templateString = try? String(contentsOf: templateURL, encoding: .utf8),
              !templateString.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            return
        }

        jsonObject["chat_template"] = templateString

        guard let updatedData = try? JSONSerialization
            .data(withJSONObject: jsonObject, options: [.prettyPrinted])
        else {
            return
        }

        let mergedURL = tokenizerConfigURL.deletingLastPathComponent()
            .appendingPathComponent("tokenizer_config.merged.json")

        do {
            try updatedData.write(to: mergedURL, options: .atomic)
        }
        catch {
            NSLog(
                "SwamaKit.ModelPool: failed to write merged chat template for %@ - %@",
                modelName,
                error.localizedDescription
            )
            return
        }

        do {
            try FileManager.default.removeItem(at: tokenizerConfigURL)
        }
        catch {
            NSLog(
                "SwamaKit.ModelPool: failed to remove original tokenizer config for %@ - %@",
                modelName,
                error.localizedDescription
            )
        }

        do {
            try FileManager.default.copyItem(at: mergedURL, to: tokenizerConfigURL)
        }
        catch {
            NSLog(
                "SwamaKit.ModelPool: failed to install merged tokenizer config for %@ - %@",
                modelName,
                error.localizedDescription
            )
        }
    }

    private func parseEOSTokenIDs(from url: URL) -> Set<Int>? {
        guard let data = try? Data(contentsOf: url),
              let jsonObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        var ids: Set<Int> = []
        var eosIDs: Set<Int> = []

        if let eosValue = jsonObject["eos_token_id"] {
            eosIDs = extractIDs(from: eosValue)
            ids.formUnion(eosIDs)
        }

        if !eosIDs.isEmpty,
           let padValue = jsonObject["pad_token_id"]
        {
            let padIDs = extractIDs(from: padValue)
            if !padIDs.isDisjoint(with: eosIDs) {
                ids.formUnion(padIDs)
            }
        }

        return ids
    }

    private func extractIDs(from value: Any) -> Set<Int> {
        switch value {
        case let intValue as Int:
            [intValue]

        case let number as NSNumber:
            [number.intValue]

        case let doubleValue as Double:
            [Int(doubleValue)]

        case let array as [Any]:
            array.reduce(into: Set<Int>()) { result, element in
                result.formUnion(extractIDs(from: element))
            }

        default:
            []
        }
    }

    private func mapTokenIDsToStrings(ids: Set<Int>, tokenizerURL: URL) -> Set<String>? {
        guard let data = try? Data(contentsOf: tokenizerURL),
              let jsonObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        var remainingIDs = ids
        var tokens: Set<String> = []

        if let addedTokens = jsonObject["added_tokens"] as? [[String: Any]] {
            for tokenInfo in addedTokens {
                guard let idValue = tokenInfo["id"],
                      let id = extractIDs(from: idValue).first,
                      remainingIDs.contains(id),
                      let content = tokenInfo["content"] as? String
                else {
                    continue
                }

                tokens.insert(content)
                remainingIDs.remove(id)
            }
        }

        if !remainingIDs.isEmpty,
           let modelDict = jsonObject["model"] as? [String: Any],
           let vocab = modelDict["vocab"] as? [String: Any]
        {
            for (token, idValue) in vocab {
                let idSet = extractIDs(from: idValue)
                guard let id = idSet.first,
                      remainingIDs.contains(id)
                else {
                    continue
                }

                tokens.insert(token)
                remainingIDs.remove(id)

                if remainingIDs.isEmpty {
                    break
                }
            }
        }

        if remainingIDs.isEmpty {
            return tokens
        }

        if !tokens.isEmpty {
            NSLog(
                "SwamaKit.ModelPool: Missing tokenizer mappings for EOS token ids: %@",
                remainingIDs.map(String.init).joined(separator: ",")
            )
            return tokens
        }

        return nil
    }

    private func ensureVLMRegistryInitialized() {
        guard vlmRegistryCache == nil else {
            return
        }

        vlmRegistryCache = [:]
        for vlmConfigEntry in VLMRegistry.all() {
            let configIDString: String = vlmConfigEntry.name
            vlmRegistryCache![configIDString] = vlmConfigEntry
        }
    }

    private func isVLMModelByLocalConfig(_ modelName: String) -> Bool {
        let modelDirectory = ModelPaths.getModelDirectory(for: modelName)
        let candidateConfigFiles = [
            "config.json",
            "preprocessor_config.json",
            "processor_config.json"
        ]

        for fileName in candidateConfigFiles {
            let fileURL = modelDirectory.appendingPathComponent(fileName)
            guard let json = loadJSONDictionary(at: fileURL) else {
                continue
            }

            if ModelTypeDetector.isVLMModelConfig(json) {
                return true
            }
        }

        return false
    }

    private func loadJSONDictionary(at url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let jsonObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }

        return jsonObject
    }

    /// Helper method to detect VLM models by name pattern (heuristic for models not in registry)
    private func isVLMModelByName(_ modelName: String) -> Bool {
        ModelTypeDetector.isVLMModelName(modelName)
    }

    // MARK: - Memory Management

    /// Starts the periodic memory management timer
    private func startMemoryManagementTimer() {
        // Cancel existing task if any
        memoryManagementTask?.cancel()

        // Create new task for periodic memory cleanup
        let interval = memoryCheckInterval
        memoryManagementTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))

                guard !Task.isCancelled, let self else {
                    break
                }

                await self.performPeriodicMemoryCleanup()
            }
        }
    }

    /// Performs periodic cleanup of idle models
    private func performPeriodicMemoryCleanup() async {
        let idleModels = getIdleModels()

        if !idleModels.isEmpty {
            NSLog("SwamaKit.ModelPool: Found \(idleModels.count) idle models for cleanup: \(idleModels.map(\.name))")

            for model in idleModels.prefix(maxIdleEvictionsPerCycle) {
                await evictModel(
                    modelName: model.name,
                    reason: "idle timeout (\(String(format: "%.1f", model.idleTime))s)"
                )
            }
        }
    }

    /// Checks for memory pressure and evicts models if needed
    private func performMemoryPressureCheck() async {
        let currentCacheSize = cache.count

        if currentCacheSize >= maxCacheSize {
            NSLog(
                "SwamaKit.ModelPool: Cache size (\(currentCacheSize)) at limit (\(maxCacheSize)). Performing memory pressure cleanup."
            )

            // Get models sorted by eviction priority (best candidates first)
            let candidates = getEvictionCandidates()

            // Evict the least valuable model to make room
            if let candidate = candidates.first {
                await evictModel(modelName: candidate.name, reason: "memory pressure (cache size: \(currentCacheSize))")
            }
        }
    }

    /// Gets models that have been idle for too long
    private func getIdleModels() -> [(name: String, idleTime: TimeInterval)] {
        modelUsageInfo.compactMap { modelName, usage in
            // Skip models that are currently running
            guard runningModels[modelName] == nil else {
                return nil
            }

            let idleTime = usage.idleTime
            if idleTime > maxIdleTime {
                return (name: modelName, idleTime: idleTime)
            }
            return nil
        }
        .sorted { $0.idleTime > $1.idleTime } // Sort by idle time descending
    }

    /// Gets models sorted by eviction priority (best candidates first)
    private func getEvictionCandidates() -> [(name: String, score: Double)] {
        modelUsageInfo.compactMap { modelName, usage in
            // Skip models that are currently running
            guard runningModels[modelName] == nil else {
                return nil
            }

            // Calculate eviction score (higher score = better candidate for eviction)
            let idleTime = usage.idleTime
            let usageFrequency = Double(usage.usageCount) / usage.ageTime
            let ageTime = usage.ageTime

            // Score formula: prioritize older idle models with lower usage frequency
            let score = idleTime / 60.0 + ageTime / 3600.0 - usageFrequency * 100.0

            return (name: modelName, score: score)
        }
        .sorted { $0.score > $1.score } // Sort by score descending
    }

    /// Evicts a specific model from the cache
    private func evictModel(modelName: String, reason: String) async {
        // A pending load has not reached inference yet and is safe to invalidate. Continue to
        // protect a genuinely running cached model from periodic eviction.
        let runningOwner = runningModels[modelName]
        let isPendingLoad = runningOwner != nil
            && loadingModels[modelName] == runningOwner
            && hasPendingLoad(for: modelName)
        guard runningOwner == nil || isPendingLoad else {
            NSLog("SwamaKit.ModelPool: Skipping eviction of \(modelName) - currently running")
            return
        }

        let diagnosticOperation = SwamaDiagnostics.startEviction(model: modelName)
        let memoryBefore = memoryHooks.activeMemory()

        invalidateLoads(for: modelName)

        // Remove from all caches to release strong references
        cache.removeValue(forKey: modelName)
        modelTypeCache.removeValue(forKey: modelName)
        embeddingRunnerCache.removeValue(forKey: modelName)
        PromptCacheStore.shared.drop(modelName: modelName)
        sttRunnerCache.removeValue(forKey: modelName)
        ttsRunnerCache.removeValue(forKey: modelName)
        modelUsageInfo.removeValue(forKey: modelName)

        // Cancel loading task if active
        if let pending = tasks.removeValue(forKey: modelName) {
            pending.task.cancel()
        }

        if let pending = embeddingTasks.removeValue(forKey: modelName) {
            pending.task.cancel()
        }

        // Cancel speech-to-text loading task if active
        if let pending = sttTasks.removeValue(forKey: modelName) {
            pending.task.cancel()
        }

        if let pending = ttsTasks.removeValue(forKey: modelName) {
            pending.task.cancel()
        }

        // All strong owners have been removed before MLX cleanup runs.
        memoryHooks.clearCache()

        // Get memory snapshot after cleanup to measure actual release
        let memoryAfter = memoryHooks.activeMemory()
        let memoryReleased = max(0, memoryBefore - memoryAfter)
        SwamaDiagnostics.completeEviction(
            diagnosticOperation,
            model: modelName,
            generation: modelLoadGenerations[modelName, default: 0],
            residentBytesAfter: SwamaDiagnostics.residentBytes()
        )

        NSLog(
            "SwamaKit.ModelPool: Evicted model \(modelName) - \(reason). Released \(memoryReleased / (1024 * 1024))MB active memory. Active: \(memoryAfter / (1024 * 1024))MB"
        )
    }

    // Aggressive cleanup removed to keep memory management simple and predictable.
}

// MARK: - ModelTypeDetector

enum ModelTypeDetector {
    static func isVLMModelName(_ modelName: String) -> Bool {
        let lowercaseName = modelName.lowercased()

        if lowercaseName.contains("gemma") {
            // Gemma models with DWQ are LLM (not VLM)
            if lowercaseName.contains("dwq") {
                return false
            }

            if lowercaseName.contains("3n"), lowercaseName.contains("lm") {
                return false // Gemma 3n - Text Only (LM) are LLMs
            }
            return true
        }

        let vlmPatterns = [
            "-vl-", // Lowercase variant
            "vl-", // Prefix variant
            "vision", // Vision models
            "visual", // Visual models
            "multimodal", // Multimodal models
            "omni" // Omni models are generally multimodal
        ]

        for pattern in vlmPatterns where lowercaseName.contains(pattern) {
            return true
        }

        return false
    }

    static func isVLMModelConfig(_ json: [String: Any]) -> Bool {
        let vlmKeyHints = [
            "vision_config",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "vision_tower",
            "visual",
            "mm_projector",
            "multi_modal_projector",
            "image_processor"
        ]

        let vlmValueHints = [
            "vl",
            "vision",
            "visual",
            "multimodal",
            "omni",
            "llava",
            "idefics",
            "paligemma",
            "pixtral",
            "minicpmv",
            "qwen3_5forconditionalgeneration"
        ]

        func containsVLMHints(in value: Any, keyContext: String? = nil) -> Bool {
            if let dictionary = value as? [String: Any] {
                for (rawKey, nestedValue) in dictionary {
                    let key = rawKey.lowercased()

                    if vlmKeyHints.contains(where: { key.contains($0) }) {
                        return true
                    }

                    if containsVLMHints(in: nestedValue, keyContext: key) {
                        return true
                    }
                }
                return false
            }

            if let array = value as? [Any] {
                for item in array where containsVLMHints(in: item, keyContext: keyContext) {
                    return true
                }
                return false
            }

            if let stringValue = value as? String {
                let normalized = stringValue.lowercased()
                if keyContext == "architectures" || keyContext == "model_type" {
                    return vlmValueHints.contains(where: { normalized.contains($0) })
                }
            }

            return false
        }

        return containsVLMHints(in: json)
    }
}

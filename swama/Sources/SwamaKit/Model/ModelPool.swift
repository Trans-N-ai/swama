import Foundation
import MLX
import MLXEmbedders
import MLXLLM
import MLXLMCommon
import MLXVLM

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
        // Memory management timer will be started when first accessed
    }

    /// Ensures memory management timer is running (called on first model access)
    private func ensureMemoryManagementStarted() {
        guard memoryManagementTask == nil else {
            return
        }

        startMemoryManagementTimer()
    }

    // MARK: Public

    public static let shared: ModelPool = .init()

    // MARK: - Audio Support

    /// Safely run an audio transcription operation with caching and concurrency control
    public func runAudio<T: Sendable>(
        modelName: String,
        operation: @Sendable @escaping (AudioRunner) async throws -> T
    ) async throws -> T {
        // Wait for available slot AND ensure the specific model is not already running
        while runningInferences >= maxConcurrentInferences || runningModels.contains(modelName) {
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }

        runningInferences += 1
        runningModels.insert(modelName)

        do {
            // Get or load the audio runner
            let runner = try await getAudioRunner(modelName: modelName)

            // Execute the operation
            let result = try await operation(runner)

            runningInferences = max(0, runningInferences - 1)
            runningModels.remove(modelName)

            return result
        }
        catch {
            runningInferences = max(0, runningInferences - 1)
            runningModels.remove(modelName)
            throw error
        }
    }

    /// Safely run a TTS operation with caching and concurrency control
    public func runTTS<T: Sendable>(
        modelKey: String,
        kind: TTSModelKind,
        operation: @Sendable @escaping (TTSRunner) async throws -> T
    ) async throws -> T {
        while runningInferences >= maxConcurrentInferences || runningModels.contains(modelKey) {
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }

        runningInferences += 1
        runningModels.insert(modelKey)

        do {
            let runner = try await getTTSRunner(modelKey: modelKey, kind: kind)
            let result = try await operation(runner)

            runningInferences = max(0, runningInferences - 1)
            runningModels.remove(modelKey)

            return result
        }
        catch {
            runningInferences = max(0, runningInferences - 1)
            runningModels.remove(modelKey)
            throw error
        }
    }

    /// Gets or loads an audio runner for the given model name
    private func getAudioRunner(modelName: String) async throws -> AudioRunner {
        // Ensure memory management is started
        ensureMemoryManagementStarted()

        if let runner = audioRunnerCache[modelName] {
            // Record usage for existing cached runner
            modelUsageInfo[modelName]?.recordUsage()
            return runner
        }

        if let task = audioTasks[modelName] {
            let runner = try await task.value
            // Record usage for newly loaded runner
            modelUsageInfo[modelName]?.recordUsage()
            return runner
        }

        // Check if we need to free up memory before loading a new model
        await performMemoryPressureCheck()

        let task = Task {
            let runner = await MainActor.run { AudioRunner() }
            try await runner.loadModel(modelName)

            // Update caches back on the actor
            self.setAudioRunner(runner, forKey: modelName)
            return runner
        }
        audioTasks[modelName] = task
        return try await task.value
    }

    private func setAudioRunner(_ runner: AudioRunner, forKey modelName: String) {
        audioRunnerCache[modelName] = runner
        modelUsageInfo[modelName] = ModelUsageInfo()
    }

    /// Gets or loads a TTS runner for the given model key
    private func getTTSRunner(modelKey: String, kind: TTSModelKind) async throws -> TTSRunner {
        ensureMemoryManagementStarted()

        if let runner = ttsRunnerCache[modelKey] {
            modelUsageInfo[modelKey]?.recordUsage()
            return runner
        }

        if let task = ttsTasks[modelKey] {
            let runner = try await task.value
            modelUsageInfo[modelKey]?.recordUsage()
            return runner
        }

        await performMemoryPressureCheck()

        let task = Task {
            let runner = await MainActor.run { TTSRunner(kind: kind) }
            try await runner.loadModel()
            self.setTTSRunner(runner, forKey: modelKey)
            return runner
        }

        ttsTasks[modelKey] = task
        return try await task.value
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

    /// Task for periodic memory management
    private var memoryManagementTask: Task<Void, Never>?

    // MARK: - Concurrency Control

    private var runningInferences = 0
    private let maxConcurrentInferences = 3 // Optimal for high-performance machines

    /// Per-model concurrency control: track which models are currently running inference
    private var runningModels: Set<String> = []

    /// Safely run a model operation with concurrency control to prevent MLX heap corruption
    public func run<T: Sendable>(
        modelName: String,
        operation: @Sendable @escaping (ModelRunner) async throws -> T
    ) async throws -> T {
        // Wait for available slot AND ensure the specific model is not already running
        while runningInferences >= maxConcurrentInferences || runningModels.contains(modelName) {
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }

        runningInferences += 1
        runningModels.insert(modelName)

        do {
            // Get or load the model container
            let container = try await getContainer(modelName: modelName)
            let runner = ModelRunner(container: container)

            // Execute the operation
            let result = try await operation(runner)

            runningInferences = max(0, runningInferences - 1)
            runningModels.remove(modelName)

            return result
        }
        catch {
            runningInferences = max(0, runningInferences - 1)
            runningModels.remove(modelName)
            throw error
        }
    }

    /// Safely run an embedding operation with concurrency control to prevent MLX heap corruption
    public func runEmbeddingWithConcurrencyControl<T: Sendable>(
        modelName: String,
        operation: @Sendable @escaping (EmbeddingRunner) async throws -> T
    ) async throws -> T {
        // Wait for available slot
        while runningInferences >= maxConcurrentInferences {
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }

        runningInferences += 1

        do {
            // Get or create embedding runner
            let runner: EmbeddingRunner
            if let existingRunner = embeddingRunnerCache[modelName] {
                runner = existingRunner
            }
            else {
                // Load the embedding model
                let container = try await loadEmbeddingModelContainer(modelName: modelName)
                runner = EmbeddingRunner(container: container)
                embeddingRunnerCache[modelName] = runner
            }

            // Execute the operation
            let result = try await operation(runner)

            runningInferences = max(0, runningInferences - 1)

            return result
        }
        catch {
            runningInferences = max(0, runningInferences - 1)
            throw error
        }
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

        if let task = tasks[modelName] {
            let container = try await task.value
            modelUsageInfo[modelName]?.recordUsage()
            return container
        }

        // Check if we need to free up memory before loading a new model
        await performMemoryPressureCheck()

        let task = Task {
            defer { tasks[modelName] = nil }

            if let isVLM = modelTypeCache[modelName] {
                guard ModelPaths.modelExistsLocally(modelName) else {
                    modelTypeCache.removeValue(forKey: modelName)
                    throw ModelPoolError.modelNotFoundLocally(modelName)
                }

                let container = try await loadModelContainer(
                    modelName: modelName,
                    isVLM: isVLM
                )

                cache[modelName] = container
                modelUsageInfo[modelName] = ModelUsageInfo()
                return container
            }

            guard ModelPaths.modelExistsLocally(modelName) else {
                throw ModelPoolError.modelNotFoundLocally(modelName)
            }

            // Determine if this is a VLM model using unified detection logic (registry priority)
            let isVLMModel = determineIfVLMModel(modelName: modelName)

            // Cache the model type for future fast path
            modelTypeCache[modelName] = isVLMModel

            // Load container using unified logic
            let container = try await loadModelContainer(
                modelName: modelName,
                isVLM: isVLMModel
            )

            cache[modelName] = container
            modelUsageInfo[modelName] = ModelUsageInfo()
            return container
        }
        tasks[modelName] = task
        return try await task.value
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
        // Get memory snapshot before clearing
        let memoryBefore = MLX.Memory.snapshot()

        // Store references to help with cleanup
        let containersToEvict = Array(cache.values)
        let tasksToCancel = Array(tasks.values)
        let audioTasksToCancel = Array(audioTasks.values)
        let ttsTasksToCancel = Array(ttsTasks.values)

        // Clear all caches to remove strong references
        cache.removeAll()
        modelTypeCache.removeAll() // Clear type cache too
        vlmRegistryCache = nil // Reset VLM registry cache
        embeddingRunnerCache.removeAll() // Clear embedding cache
        audioRunnerCache.removeAll() // Clear audio cache
        ttsRunnerCache.removeAll() // Clear TTS cache
        modelUsageInfo.removeAll() // Clear usage tracking

        // Cancel all loading tasks
        for task in tasksToCancel {
            task.cancel()
        }
        tasks.removeAll()

        // Cancel all audio loading tasks
        for task in audioTasksToCancel {
            task.cancel()
        }
        audioTasks.removeAll()

        // Cancel all TTS loading tasks
        for task in ttsTasksToCancel {
            task.cancel()
        }
        ttsTasks.removeAll()

        // Explicitly release container references
        _ = containersToEvict

        // Perform aggressive memory cleanup
        Task {
            await performAggressiveMemoryCleanup()

            let memoryAfter = MLX.Memory.snapshot()
            let memoryReleased = memoryBefore.activeMemory - memoryAfter.activeMemory
            NSLog(
                "SwamaKit.ModelPool: Cache cleared (\(containersToEvict.count) models). Released \(memoryReleased / (1024 * 1024))MB active memory. Active: \(memoryAfter.activeMemory / (1024 * 1024))MB"
            )
        }

        NSLog("SwamaKit.ModelPool: Cache clearing initiated for \(containersToEvict.count) models")
    }

    /// Removes a specific model from the cache and cancels its loading task if active.
    public func remove(modelName: String) {
        // Get reference before removing
        let containerToRemove = cache[modelName]

        cache.removeValue(forKey: modelName)
        modelTypeCache.removeValue(forKey: modelName) // Clear type cache for this model
        embeddingRunnerCache.removeValue(forKey: modelName) // Clear embedding cache for this model
        audioRunnerCache.removeValue(forKey: modelName) // Clear audio cache for this model
        ttsRunnerCache.removeValue(forKey: modelName) // Clear TTS cache for this model
        modelUsageInfo.removeValue(forKey: modelName) // Clear usage tracking for this model

        if let task = tasks.removeValue(forKey: modelName) {
            task.cancel()
        }

        if let audioTask = audioTasks.removeValue(forKey: modelName) {
            audioTask.cancel()
        }

        if let ttsTask = ttsTasks.removeValue(forKey: modelName) {
            ttsTask.cancel()
        }

        // Release container reference
        _ = containerToRemove

        // Clear MLX GPU cache after removing model
        MLX.Memory.clearCache()
    }

    // MARK: Private

    private var cache: [String: MLXLMCommon.ModelContainer] = .init()
    private var tasks: [String: Task<MLXLMCommon.ModelContainer, Error>] = .init()
    private var embeddingRunnerCache: [String: EmbeddingRunner] = .init()
    private var audioRunnerCache: [String: AudioRunner] = .init()
    private var audioTasks: [String: Task<AudioRunner, Error>] = .init()
    private var ttsRunnerCache: [String: TTSRunner] = .init()
    private var ttsTasks: [String: Task<TTSRunner, Error>] = .init()

    /// Memory management tracking
    private var modelUsageInfo: [String: ModelUsageInfo] = .init()

    private var modelTypeCache: [String: Bool] = .init()
    private var vlmRegistryCache: [String: MLXLMCommon.ModelConfiguration]?

    /// Unified VLM detection logic with registry priority
    private func determineIfVLMModel(modelName: String) -> Bool {
        // Ensure VLM registry is initialized
        ensureVLMRegistryInitialized()

        if vlmRegistryCache![modelName] != nil {
            return true
        }

        return isVLMModelByName(modelName)
    }

    private func loadModelContainer(
        modelName: String,
        isVLM: Bool
    ) async throws -> MLXLMCommon.ModelContainer {
        ensureChatTemplateIfNeeded(for: modelName)
        // Configure extra EOS tokens for models with known issues
        let extraEOSTokens = getExtraEOSTokens(for: modelName)

        let localConfig = MLXLMCommon.ModelConfiguration(
            directory: ModelPaths.getModelDirectory(for: modelName),
            extraEOSTokens: extraEOSTokens
        )

        do {
            if isVLM {
                return try await VLMModelFactory.shared.loadContainer(configuration: localConfig)
            }
            else {
                return try await LLMModelFactory.shared.loadContainer(configuration: localConfig)
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

    /// Helper method to detect VLM models by name pattern (heuristic for models not in registry)
    private func isVLMModelByName(_ modelName: String) -> Bool {
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
            "multimodal" // Multimodal models
        ]

        for pattern in vlmPatterns {
            if lowercaseName.contains(pattern) {
                return true
            }
        }

        return false
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

            for model in idleModels {
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
            guard !runningModels.contains(modelName) else {
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
            guard !runningModels.contains(modelName) else {
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
        // Double-check the model is not currently running
        guard !runningModels.contains(modelName) else {
            NSLog("SwamaKit.ModelPool: Skipping eviction of \(modelName) - currently running")
            return
        }

        // Get memory snapshot before eviction
        let memoryBefore = MLX.Memory.snapshot()

        // Get reference to the model container before removing it
        let containerToEvict = cache[modelName]

        // Remove from all caches to release strong references
        cache.removeValue(forKey: modelName)
        modelTypeCache.removeValue(forKey: modelName)
        embeddingRunnerCache.removeValue(forKey: modelName)
        audioRunnerCache.removeValue(forKey: modelName)
        ttsRunnerCache.removeValue(forKey: modelName)
        modelUsageInfo.removeValue(forKey: modelName)

        // Cancel loading task if active
        if let task = tasks.removeValue(forKey: modelName) {
            task.cancel()
        }

        // Cancel audio loading task if active
        if let audioTask = audioTasks.removeValue(forKey: modelName) {
            audioTask.cancel()
        }

        if let ttsTask = ttsTasks.removeValue(forKey: modelName) {
            ttsTask.cancel()
        }

        // Explicitly nil out the container reference to help ARC
        _ = containerToEvict

        // Aggressive memory cleanup sequence - force immediate GPU memory release
        await performAggressiveMemoryCleanup()

        // Get memory snapshot after cleanup to measure actual release
        let memoryAfter = MLX.Memory.snapshot()
        let memoryReleased = memoryBefore.activeMemory - memoryAfter.activeMemory

        NSLog(
            "SwamaKit.ModelPool: Evicted model \(modelName) - \(reason). Released \(memoryReleased / (1024 * 1024))MB active memory. Active: \(memoryAfter.activeMemory / (1024 * 1024))MB"
        )
    }

    /// Performs aggressive memory cleanup to actually release GPU memory
    private func performAggressiveMemoryCleanup() async {
        // Step 1: Force Swift ARC to run garbage collection
        // Create and release a temporary array to trigger GC
        autoreleasepool {
            _ = Array(0 ..< 1000)
        }

        // Step 2: Clear MLX computational cache
        MLX.Memory.clearCache()

        // Step 3: Temporarily disable cache to force immediate memory release
        let originalCacheLimit = MLX.Memory.cacheLimit
        MLX.Memory.cacheLimit = 0

        // Step 4: Clear cache again with disabled limit
        MLX.Memory.clearCache()

        // Step 5: Brief pause to allow memory cleanup to propagate
        try? await Task.sleep(nanoseconds: 100_000_000) // 100ms

        // Step 6: Force another garbage collection cycle
        autoreleasepool {
            _ = Array(0 ..< 1000)
        }

        // Step 7: Clear cache one more time to ensure cleanup
        MLX.Memory.clearCache()

        // Step 8: Restore original cache limit
        MLX.Memory.cacheLimit = originalCacheLimit
    }
}

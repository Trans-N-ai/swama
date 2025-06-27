import Foundation
import MLX
import mlx_embeddings
import MLXLLM
import MLXLMCommon
import MLXVLM

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

    // MARK: - WhisperKit Support

    /// Safely run a WhisperKit operation with caching and concurrency control
    public func runWhisperKit<T: Sendable>(
        modelName: String,
        operation: @Sendable @escaping (WhisperKitRunner) async throws -> T
    ) async throws -> T {
        // Wait for available slot AND ensure the specific model is not already running
        while runningInferences >= maxConcurrentInferences || runningModels.contains(modelName) {
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms
        }

        runningInferences += 1
        runningModels.insert(modelName)

        do {
            // Get or load the WhisperKit runner
            let runner = try await getWhisperKitRunner(modelName: modelName)

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

    /// Gets or loads a WhisperKit runner for the given model name
    private func getWhisperKitRunner(modelName: String) async throws -> WhisperKitRunner {
        // Ensure memory management is started
        ensureMemoryManagementStarted()

        if let runner = whisperKitRunnerCache[modelName] {
            // Record usage for existing cached runner
            modelUsageInfo[modelName]?.recordUsage()
            return runner
        }

        if let task = whisperKitTasks[modelName] {
            let runner = try await task.value
            // Record usage for newly loaded runner
            modelUsageInfo[modelName]?.recordUsage()
            return runner
        }

        // Check if we need to free up memory before loading a new model
        await performMemoryPressureCheck()

        let task = Task {
            defer { whisperKitTasks[modelName] = nil }

            let runner = WhisperKitRunner()
            try await runner.loadModel(modelName)

            whisperKitRunnerCache[modelName] = runner
            // Initialize usage tracking for new runner
            modelUsageInfo[modelName] = ModelUsageInfo()
            return runner
        }
        whisperKitTasks[modelName] = task
        return try await task.value
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

            // Create a fresh ModelRunner instance for this request to avoid sharing conflicts
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

    /// Gets or loads a ModelContainer (internal method without concurrency control)
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
            // Record usage for newly loaded model
            modelUsageInfo[modelName]?.recordUsage()
            return container
        }

        // Check if we need to free up memory before loading a new model
        await performMemoryPressureCheck()

        let task = Task {
            defer { tasks[modelName] = nil }

            // Fast path: Check if we already know the model type
            if let isVLM = modelTypeCache[modelName] {
                let container: MLXLMCommon.ModelContainer
                if isVLM {
                    // We know it's a VLM, load directly
                    let vlmConfig = try getVLMConfiguration(modelName: modelName)
                    container = try await VLMModelFactory.shared.loadContainer(configuration: vlmConfig)
                }
                else {
                    // We know it's an LLM, load directly
                    let llmConfig = MLXLMCommon.ModelConfiguration(id: modelName)
                    container = try await LLMModelFactory.shared.loadContainer(configuration: llmConfig)
                }
                cache[modelName] = container
                // Initialize usage tracking for new model
                modelUsageInfo[modelName] = ModelUsageInfo()
                return container
            }

            // Slow path: First time loading this model - determine type
            var configToLoad: MLXLMCommon.ModelConfiguration?

            // Lazy load and cache VLM registry for better performance
            if vlmRegistryCache == nil {
                vlmRegistryCache = [:]
                for vlmConfigEntry in VLMRegistry.all() {
                    // Extract the model ID string from the identifier
                    let configIDString: String = vlmConfigEntry.name
                    vlmRegistryCache![configIDString] = vlmConfigEntry
                }
            }

            // Fast lookup in cached registry
            if let vlmConfig = vlmRegistryCache![modelName] {
                // Check if VLM model exists locally first using shared function
                let localConfig = createModelConfiguration(modelName: modelName)

                // If it's a directory-based config, use it; otherwise use registry config
                if case .directory = localConfig.id {
                    configToLoad = localConfig
                }
                else {
                    configToLoad = vlmConfig
                }
                modelTypeCache[modelName] = true // Cache for future fast path
            }
            else if isVLMModelByName(modelName) {
                // Heuristic detection: model name suggests it's a VLM but not in registry
                // Try to load it as VLM with directory-based configuration
                let localConfig = createModelConfiguration(modelName: modelName)

                if case .directory = localConfig.id {
                    configToLoad = localConfig
                    modelTypeCache[modelName] = true // Cache for future fast path
                }
                else {
                    // VLM model not available locally and not in registry - this will likely fail
                    // Create a basic VLM configuration and let VLMModelFactory handle it
                    configToLoad = MLXLMCommon.ModelConfiguration(id: modelName)
                    modelTypeCache[modelName] = true // Cache for future fast path
                }
            }
            else {
                // It's an LLM - use the shared loadModelContainer function
                let container = try await loadModelContainer(modelName: modelName)

                cache[modelName] = container
                modelUsageInfo[modelName] = ModelUsageInfo()
                modelTypeCache[modelName] = false // Cache for future fast path
                return container
            }

            guard let finalConfig = configToLoad else {
                throw NSError(
                    domain: "ModelPoolError",
                    code: 1,
                    userInfo: [
                        NSLocalizedDescriptionKey: "Failed to determine or create a ModelConfiguration for \(modelName)."
                    ]
                )
            }

            // Only VLM models reach this point
            let container: MLXLMCommon.ModelContainer
            container = try await VLMModelFactory.shared.loadContainer(configuration: finalConfig)

            cache[modelName] = container
            // Initialize usage tracking for new model
            modelUsageInfo[modelName] = ModelUsageInfo()
            NSLog("SwamaKit.ModelPool: Successfully loaded \(modelName)")
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
        let memoryBefore = MLX.GPU.snapshot()

        // Store references to help with cleanup
        let containersToEvict = Array(cache.values)
        let tasksToCancel = Array(tasks.values)
        let whisperKitTasksToCancel = Array(whisperKitTasks.values)

        // Clear all caches to remove strong references
        cache.removeAll()
        modelTypeCache.removeAll() // Clear type cache too
        vlmRegistryCache = nil // Reset VLM registry cache
        embeddingRunnerCache.removeAll() // Clear embedding cache
        whisperKitRunnerCache.removeAll() // Clear WhisperKit cache
        modelUsageInfo.removeAll() // Clear usage tracking

        // Cancel all loading tasks
        for task in tasksToCancel {
            task.cancel()
        }
        tasks.removeAll()

        // Cancel all WhisperKit loading tasks
        for task in whisperKitTasksToCancel {
            task.cancel()
        }
        whisperKitTasks.removeAll()

        // Explicitly release container references
        _ = containersToEvict

        // Perform aggressive memory cleanup
        Task {
            await performAggressiveMemoryCleanup()

            let memoryAfter = MLX.GPU.snapshot()
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
        whisperKitRunnerCache.removeValue(forKey: modelName) // Clear WhisperKit cache for this model
        modelUsageInfo.removeValue(forKey: modelName) // Clear usage tracking for this model

        if let task = tasks.removeValue(forKey: modelName) {
            task.cancel()
        }

        if let whisperKitTask = whisperKitTasks.removeValue(forKey: modelName) {
            whisperKitTask.cancel()
        }

        // Release container reference
        _ = containerToRemove

        // Clear MLX GPU cache after removing model
        MLX.GPU.clearCache()
    }

    // MARK: Private

    private var cache: [String: MLXLMCommon.ModelContainer] = .init()
    private var tasks: [String: Task<MLXLMCommon.ModelContainer, Error>] = .init()
    private var embeddingRunnerCache: [String: EmbeddingRunner] = .init()
    private var whisperKitRunnerCache: [String: WhisperKitRunner] = .init()
    private var whisperKitTasks: [String: Task<WhisperKitRunner, Error>] = .init()

    /// Memory management tracking
    private var modelUsageInfo: [String: ModelUsageInfo] = .init()

    // Performance optimization: Cache model types to avoid repeated Registry lookups
    private var modelTypeCache: [String: Bool] = .init() // true = VLM, false = LLM
    private var vlmRegistryCache: [String: MLXLMCommon.ModelConfiguration]?

    /// Helper method to get VLM configuration (used in fast path)
    private func getVLMConfiguration(modelName: String) throws -> MLXLMCommon.ModelConfiguration {
        // Ensure VLM registry cache is available
        if vlmRegistryCache == nil {
            vlmRegistryCache = [:]
            for vlmConfigEntry in VLMRegistry.all() {
                // Extract the model ID string from the identifier
                let configIDString: String = vlmConfigEntry.name
                vlmRegistryCache![configIDString] = vlmConfigEntry
            }
        }

        guard let vlmConfig = vlmRegistryCache![modelName] else {
            throw NSError(
                domain: "ModelPoolError",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "VLM configuration not found for \(modelName)"]
            )
        }

        return vlmConfig
    }

    /// Helper method to detect VLM models by name pattern (heuristic for models not in registry)
    private func isVLMModelByName(_ modelName: String) -> Bool {
        let vlmPatterns = [
            "gemma", // Gemma models
            "-VL-", // Common VLM naming pattern (e.g., Qwen2.5-VL-32B)
            "-vl-", // Lowercase variant
            "VL-", // Prefix variant
            "vision", // Vision models
            "Visual", // Visual models
            "multimodal" // Multimodal models
        ]

        for pattern in vlmPatterns {
            if modelName.contains(pattern) {
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

        NSLog("SwamaKit.ModelPool: Memory management task started (interval: \(memoryCheckInterval)s)")
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
        let memoryBefore = MLX.GPU.snapshot()

        // Get reference to the model container before removing it
        let containerToEvict = cache[modelName]

        // Remove from all caches to release strong references
        cache.removeValue(forKey: modelName)
        modelTypeCache.removeValue(forKey: modelName)
        embeddingRunnerCache.removeValue(forKey: modelName)
        whisperKitRunnerCache.removeValue(forKey: modelName)
        modelUsageInfo.removeValue(forKey: modelName)

        // Cancel loading task if active
        if let task = tasks.removeValue(forKey: modelName) {
            task.cancel()
        }

        // Cancel WhisperKit loading task if active
        if let whisperKitTask = whisperKitTasks.removeValue(forKey: modelName) {
            whisperKitTask.cancel()
        }

        // Explicitly nil out the container reference to help ARC
        _ = containerToEvict

        // Aggressive memory cleanup sequence - force immediate GPU memory release
        await performAggressiveMemoryCleanup()

        // Get memory snapshot after cleanup to measure actual release
        let memoryAfter = MLX.GPU.snapshot()
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
        MLX.GPU.clearCache()

        // Step 3: Temporarily disable cache to force immediate memory release
        let originalCacheLimit = MLX.GPU.cacheLimit
        MLX.GPU.set(cacheLimit: 0)

        // Step 4: Clear cache again with disabled limit
        MLX.GPU.clearCache()

        // Step 5: Brief pause to allow memory cleanup to propagate
        try? await Task.sleep(nanoseconds: 100_000_000) // 100ms

        // Step 6: Force another garbage collection cycle
        autoreleasepool {
            _ = Array(0 ..< 1000)
        }

        // Step 7: Clear cache one more time to ensure cleanup
        MLX.GPU.clearCache()

        // Step 8: Restore original cache limit
        MLX.GPU.set(cacheLimit: originalCacheLimit)

        NSLog("SwamaKit.ModelPool: Aggressive memory cleanup completed - forced GC, cache disabled/cleared/restored")
    }
}

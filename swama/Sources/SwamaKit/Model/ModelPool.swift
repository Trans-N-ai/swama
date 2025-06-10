import Foundation
import MLX
import mlx_embeddings
import MLXLLM
import MLXLMCommon
import MLXVLM

/// A pool to manage and cache `ModelContainer` instances with built-in concurrency control.
/// This helps in reusing already loaded models to save resources and time while preventing
/// MLX heap corruption through controlled concurrent access.
public actor ModelPool {
    // MARK: Lifecycle

    public init() {}

    // MARK: Public

    public static let shared: ModelPool = .init()
    
    // MARK: - Concurrency Control
    
    private var runningInferences = 0
    private let maxConcurrentInferences = 3 // Optimal for high-performance machines
    
    // Per-model concurrency control: track which models are currently running inference
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
        NSLog("SwamaKit.ModelPool: Acquired inference slot for \(modelName). Running: \(runningInferences)/\(maxConcurrentInferences), Active models: \(runningModels)")
        
        do {
            // Get or load the model container
            let container = try await getContainer(modelName: modelName)
            
            // Create a fresh ModelRunner instance for this request to avoid sharing conflicts
            let runner = ModelRunner(container: container)
            
            // Execute the operation
            let result = try await operation(runner)
            
            runningInferences = max(0, runningInferences - 1)
            runningModels.remove(modelName)
            NSLog("SwamaKit.ModelPool: Released inference slot for \(modelName). Running: \(runningInferences)/\(maxConcurrentInferences), Active models: \(runningModels)")
            
            return result
        } catch {
            runningInferences = max(0, runningInferences - 1)
            runningModels.remove(modelName)
            NSLog("SwamaKit.ModelPool: Released inference slot for \(modelName) (error). Running: \(runningInferences)/\(maxConcurrentInferences), Active models: \(runningModels)")
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
        NSLog("SwamaKit.ModelPool: Acquired inference slot (embedding). Running: \(runningInferences)/\(maxConcurrentInferences)")
        
        do {
            // Get or create embedding runner
            let runner: EmbeddingRunner
            if let existingRunner = embeddingRunnerCache[modelName] {
                runner = existingRunner
            } else {
                // Load the embedding model
                let container = try await loadEmbeddingModelContainer(modelName: modelName)
                runner = EmbeddingRunner(container: container)
                embeddingRunnerCache[modelName] = runner
                NSLog("SwamaKit.ModelPool: Cached embedding runner for \(modelName).")
            }
            
            // Execute the operation
            let result = try await operation(runner)
            
            runningInferences = max(0, runningInferences - 1)
            NSLog("SwamaKit.ModelPool: Released inference slot (embedding). Running: \(runningInferences)/\(maxConcurrentInferences)")
            
            return result
        } catch {
            runningInferences = max(0, runningInferences - 1)
            NSLog("SwamaKit.ModelPool: Released inference slot (embedding error). Running: \(runningInferences)/\(maxConcurrentInferences)")
            throw error
        }
    }

    /// Gets or loads a ModelContainer (internal method without concurrency control)
    private func getContainer(modelName: String) async throws -> MLXLMCommon.ModelContainer {
        if let container = cache[modelName] {
            NSLog("SwamaKit.ModelPool: Cache hit for \(modelName).")
            return container
        }

        if let task = tasks[modelName] {
            NSLog("SwamaKit.ModelPool: Joining existing loading task for \(modelName).")
            return try await task.value
        }

        NSLog("SwamaKit.ModelPool: Cache miss for \(modelName). Starting new loading task.")
        let task = Task {
            defer { tasks[modelName] = nil }

            // Fast path: Check if we already know the model type
            if let isVLM = modelTypeCache[modelName] {
                let container: MLXLMCommon.ModelContainer
                if isVLM {
                    // We know it's a VLM, load directly
                    let vlmConfig = try getVLMConfiguration(modelName: modelName)
                    NSLog("SwamaKit.ModelPool: Fast path - Loading VLM model \(modelName) using VLMModelFactory.")
                    container = try await VLMModelFactory.shared.loadContainer(configuration: vlmConfig)
                }
                else {
                    // We know it's an LLM, load directly
                    NSLog("SwamaKit.ModelPool: Fast path - Loading LLM model \(modelName) using LLMModelFactory.")
                    let llmConfig = MLXLMCommon.ModelConfiguration(id: modelName)
                    container = try await LLMModelFactory.shared.loadContainer(configuration: llmConfig)
                }
                cache[modelName] = container
                NSLog("SwamaKit.ModelPool: Successfully loaded and cached \(modelName) via fast path.")
                return container
            }

            // Slow path: First time loading this model - determine type
            var configToLoad: MLXLMCommon.ModelConfiguration?
            var useVLMFactory = false

            // Lazy load and cache VLM registry for better performance
            if vlmRegistryCache == nil {
                vlmRegistryCache = [:]
                for vlmConfigEntry in VLMRegistry.all() {
                    if case let .id(configIDString) = vlmConfigEntry.id {
                        vlmRegistryCache![configIDString] = vlmConfigEntry
                    }
                }
                NSLog("SwamaKit.ModelPool: Built VLM registry cache with \(vlmRegistryCache!.count) entries.")
            }

            // Fast lookup in cached registry
            if let vlmConfig = vlmRegistryCache![modelName] {
                configToLoad = vlmConfig
                useVLMFactory = true
                modelTypeCache[modelName] = true // Cache for future fast path
                NSLog("SwamaKit.ModelPool: Found \(modelName) in VLMRegistry.")
            }
            else {
                // It's an LLM
                configToLoad = MLXLMCommon.ModelConfiguration(id: modelName)
                useVLMFactory = false
                modelTypeCache[modelName] = false // Cache for future fast path
                NSLog("SwamaKit.ModelPool: \(modelName) not found in VLMRegistry. Treating as LLM.")
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

            let container: MLXLMCommon.ModelContainer
            if useVLMFactory {
                NSLog("SwamaKit.ModelPool: Loading VLM model \(finalConfig.name) using VLMModelFactory.")
                container = try await VLMModelFactory.shared.loadContainer(configuration: finalConfig)
            }
            else {
                NSLog("SwamaKit.ModelPool: Loading LLM model \(finalConfig.name) using LLMModelFactory.")
                container = try await LLMModelFactory.shared.loadContainer(configuration: finalConfig)
            }

            cache[modelName] = container
            NSLog("SwamaKit.ModelPool: Successfully loaded and cached \(modelName) (resolved as \(finalConfig.name)).")
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
        NSLog("SwamaKit.ModelPool: Cached embedding runner for \(modelName).")
    }

    /// Clears the entire model cache and cancels any ongoing loading tasks.
    public func clearCache() {
        cache.removeAll()
        modelTypeCache.removeAll() // Clear type cache too
        vlmRegistryCache = nil // Reset VLM registry cache
        embeddingRunnerCache.removeAll() // Clear embedding cache
        for (_, task) in tasks {
            task.cancel()
        }
        tasks.removeAll()
        NSLog("SwamaKit.ModelPool: Cache cleared and all loading tasks cancelled.")
    }

    /// Removes a specific model from the cache and cancels its loading task if active.
    public func remove(modelName: String) {
        cache.removeValue(forKey: modelName)
        modelTypeCache.removeValue(forKey: modelName) // Clear type cache for this model
        embeddingRunnerCache.removeValue(forKey: modelName) // Clear embedding cache for this model
        if let task = tasks.removeValue(forKey: modelName) {
            task.cancel()
            NSLog("SwamaKit.ModelPool: Removed \(modelName) from cache and cancelled its loading task.")
        }
        else {
            NSLog("SwamaKit.ModelPool: Removed \(modelName) from cache (no active loading task).")
        }
    }

    // MARK: Private

    private var cache: [String: MLXLMCommon.ModelContainer] = .init()
    private var tasks: [String: Task<MLXLMCommon.ModelContainer, Error>] = .init()
    private var embeddingRunnerCache: [String: EmbeddingRunner] = .init()

    // Performance optimization: Cache model types to avoid repeated Registry lookups
    private var modelTypeCache: [String: Bool] = .init() // true = VLM, false = LLM
    private var vlmRegistryCache: [String: MLXLMCommon.ModelConfiguration]?

    /// Helper method to get VLM configuration (used in fast path)
    private func getVLMConfiguration(modelName: String) throws -> MLXLMCommon.ModelConfiguration {
        // Ensure VLM registry cache is available
        if vlmRegistryCache == nil {
            vlmRegistryCache = [:]
            for vlmConfigEntry in VLMRegistry.all() {
                if case let .id(configIDString) = vlmConfigEntry.id {
                    vlmRegistryCache![configIDString] = vlmConfigEntry
                }
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
}

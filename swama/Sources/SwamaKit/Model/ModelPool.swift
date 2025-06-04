import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

/// A pool to manage and cache `ModelContainer` instances.
/// This helps in reusing already loaded models to save resources and time.
public actor ModelPool {
    // MARK: Lifecycle

    public init() {}

    // MARK: Public

    public static let shared: ModelPool = .init()

    public func get(modelName: String) async throws -> ModelContainer {
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
                let container: ModelContainer
                if isVLM {
                    // We know it's a VLM, load directly
                    let vlmConfig = try getVLMConfiguration(modelName: modelName)
                    NSLog("SwamaKit.ModelPool: Fast path - Loading VLM model \(modelName) using VLMModelFactory.")
                    container = try await VLMModelFactory.shared.loadContainer(configuration: vlmConfig)
                }
                else {
                    // We know it's an LLM, load directly
                    NSLog("SwamaKit.ModelPool: Fast path - Loading LLM model \(modelName) using LLMModelFactory.")
                    let llmConfig = ModelConfiguration(id: modelName)
                    container = try await LLMModelFactory.shared.loadContainer(configuration: llmConfig)
                }
                cache[modelName] = container
                NSLog("SwamaKit.ModelPool: Successfully loaded and cached \(modelName) via fast path.")
                return container
            }

            // Slow path: First time loading this model - determine type
            var configToLoad: ModelConfiguration?
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
                configToLoad = ModelConfiguration(id: modelName)
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

            let container: ModelContainer
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

    /// Clears the entire model cache and cancels any ongoing loading tasks.
    public func clearCache() {
        cache.removeAll()
        modelTypeCache.removeAll() // Clear type cache too
        vlmRegistryCache = nil // Reset VLM registry cache
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
        if let task = tasks.removeValue(forKey: modelName) {
            task.cancel()
            NSLog("SwamaKit.ModelPool: Removed \(modelName) from cache and cancelled its loading task.")
        }
        else {
            NSLog("SwamaKit.ModelPool: Removed \(modelName) from cache (no active loading task).")
        }
    }

    // MARK: Private

    private var cache: [String: ModelContainer] = .init()
    private var tasks: [String: Task<ModelContainer, Error>] = .init()

    // Performance optimization: Cache model types to avoid repeated Registry lookups
    private var modelTypeCache: [String: Bool] = .init() // true = VLM, false = LLM
    private var vlmRegistryCache: [String: ModelConfiguration]?

    /// Helper method to get VLM configuration (used in fast path)
    private func getVLMConfiguration(modelName: String) throws -> ModelConfiguration {
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

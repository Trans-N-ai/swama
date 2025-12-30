import Foundation

// MARK: - ModelPaths

/// Centralized configuration for model storage paths
public enum ModelPaths {
    /// The custom path for storing models (dynamically read from environment)
    public static var customModelsDirectory: URL? {
        if let customPath = ProcessInfo.processInfo.environment["SWAMA_MODELS"],
           !customPath.isEmpty
        {
            return URL(fileURLWithPath: customPath)
        }
        return nil
    }

    /// The preferred path for storing models (new installations)
    public static let preferredModelsDirectory: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".swama/models")
    }()

    /// The legacy path for models (for compatibility)
    public static let legacyModelsDirectory: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent("Documents/huggingface/models")
    }()

    /// The audio/TTS models cache directory (used by MLXAudio/Hub)
    public static let audioModelsDirectory: URL = {
        let cachesURL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return cachesURL.appendingPathComponent("huggingface/models")
    }()

    /// The actual directory used for storing new models (respects SWAMA_MODELS environment variable)
    public static var activeModelsDirectory: URL {
        customModelsDirectory ?? preferredModelsDirectory
    }

    /// Get the local directory path for a specific model, checking both preferred and legacy locations
    /// Returns the first location where the model exists, or the preferred location if neither exists
    public static func getModelDirectory(for modelName: String) -> URL {
        // Check if it's an audio or TTS model - use audio cache directory
        if isAudioModelName(modelName) || isTTSModelRepo(modelName) {
            // Audio models use standard directory structure: {org}/{model}
            // e.g., mlx-community/whisper-large-v3-turbo-4bit -> mlx-community/whisper-large-v3-turbo-4bit
            return audioModelsDirectory.appendingPathComponent(modelName)
        }

        let customPath = customModelsDirectory?.appendingPathComponent(modelName)
        let preferredPath = preferredModelsDirectory.appendingPathComponent(modelName)
        let legacyPath = legacyModelsDirectory.appendingPathComponent(modelName)

        // Check if model exists in custom location first
        if let customPath,
           FileManager.default.fileExists(atPath: customPath.appendingPathComponent(".swama-meta.json").path)
        {
            return parseModelMetadataPath(from: customPath.appendingPathComponent(".swama-meta.json"))
        }

        // Check if model exists in preferred location
        if FileManager.default.fileExists(atPath: preferredPath.appendingPathComponent(".swama-meta.json").path) {
            return parseModelMetadataPath(from: preferredPath.appendingPathComponent(".swama-meta.json"))
        }

        // Check if model exists in legacy location
        if FileManager.default.fileExists(atPath: legacyPath.appendingPathComponent(".swama-meta.json").path) {
            return legacyPath
        }

        // If model doesn't exist in either location, and if custom path is set, return custom path for new downloads
        if let customPath {
            return customPath
        }
        // Else return preferred location for new downloads
        return preferredPath
    }

    private static func parseModelMetadataPath(from metaURL: URL) -> URL {
        guard let data = try? Data(contentsOf: metaURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let path = json["path"] as? String
        else {
            return metaURL.deletingLastPathComponent()
        }

        return URL(fileURLWithPath: path)
    }

    /// Check if a model exists locally by checking for .swama-meta.json file
    public static func modelExistsLocally(_ modelName: String) -> Bool {
        if let ttsModel = TTSModelResolver.resolve(modelName) {
            return ttsModelExistsLocally(kind: ttsModel.kind)
        }

        if isTTSModelRepo(modelName) {
            return modelMetadataExists(for: modelName) ||
                FileManager.default.fileExists(atPath: hubCacheDirectory(for: modelName).path)
        }

        if ModelAliasResolver.isAudioModel(modelName) {
            return audioModelExistsLocally(modelName)
        }

        let modelDir = getModelDirectory(for: modelName)
        let metaPath = modelDir.appendingPathComponent(".swama-meta.json").path
        return FileManager.default.fileExists(atPath: metaPath)
    }

    /// Get all directories that should be scanned for models
    public static var allModelsDirectories: [URL] {
        var directories = [preferredModelsDirectory, legacyModelsDirectory, audioModelsDirectory]
        if let customDirectory = customModelsDirectory {
            directories.insert(customDirectory, at: 0)
        }
        return directories
    }

    /// Remove a model from disk
    /// Returns true if model was found and deleted, false if model wasn't found
    public static func removeModel(_ modelName: String) throws -> Bool {
        if let ttsModel = TTSModelResolver.resolve(modelName) {
            return removeTTSModel(kind: ttsModel.kind)
        }

        if isTTSModelRepo(modelName) {
            var removedAny = false
            if removeModelDirectoryIfPresent(for: modelName) {
                removedAny = true
            }

            let hubDir = hubCacheDirectory(for: modelName)
            if FileManager.default.fileExists(atPath: hubDir.path) {
                try FileManager.default.removeItem(at: hubDir)
                removedAny = true
            }

            return removedAny
        }

        // Check if it's an audio model
        if isAudioModelName(modelName) {
            // Audio models use HuggingFace Hub format
            return try removeAudioModel(modelName)
        }

        // Check all possible model locations in priority order for LLM models
        let locations = [
            customModelsDirectory?.appendingPathComponent(modelName),
            preferredModelsDirectory.appendingPathComponent(modelName),
            legacyModelsDirectory.appendingPathComponent(modelName)
        ].compactMap(\.self)

        for location in locations {
            let metadataFile = location.appendingPathComponent(".swama-meta.json")
            if FileManager.default.fileExists(atPath: metadataFile.path) {
                try FileManager.default.removeItem(at: location)
                return true
            }
        }

        return false // Model not found
    }

    private static func ttsModelExistsLocally(kind: TTSModelKind) -> Bool {
        let repoIds = TTSModelResolver.repoIDs(for: kind)
        for repoId in repoIds {
            if modelMetadataExists(for: repoId) || FileManager.default.fileExists(atPath: hubCacheDirectory(for: repoId).path) {
                return true
            }
        }
        return false
    }

    private static func audioModelExistsLocally(_ modelName: String) -> Bool {
        for dir in audioModelCacheDirectories(modelName: modelName) {
            let metaPath = dir.appendingPathComponent(".swama-meta.json").path
            if FileManager.default.fileExists(atPath: metaPath) || FileManager.default.fileExists(atPath: dir.path) {
                return true
            }
        }
        return false
    }

    private static func removeTTSModel(kind: TTSModelKind) -> Bool {
        let repoIds = TTSModelResolver.repoIDs(for: kind)
        var removedAny = false

        for repoId in repoIds {
            if removeModelDirectoryIfPresent(for: repoId) {
                removedAny = true
            }

            let hubDir = hubCacheDirectory(for: repoId)
            if FileManager.default.fileExists(atPath: hubDir.path) {
                try? FileManager.default.removeItem(at: hubDir)
                removedAny = true
            }
        }

        return removedAny
    }

    private static func removeAudioModel(_ modelName: String) throws -> Bool {
        let dirs = audioModelCacheDirectories(modelName: modelName)
        guard !dirs.isEmpty else {
            return false
        }

        var removedAny = false
        for dir in dirs {
            if FileManager.default.fileExists(atPath: dir.path) {
                try FileManager.default.removeItem(at: dir)
                removedAny = true
            }
        }

        return removedAny
    }

    private static func audioModelCacheDirectories(modelName: String) -> [URL] {
        guard isAudioModelName(modelName) else {
            return []
        }

        return [
            audioModelsDirectory.appendingPathComponent(modelName),
            hubCacheDirectory(for: modelName),
        ]
    }

    private static func hubCacheDirectory(for repoId: String) -> URL {
        let hubFormattedName = repoId.replacingOccurrences(of: "/", with: "--")
        return audioModelsDirectory.appendingPathComponent("models--\(hubFormattedName)")
    }

    private static func isAudioModelName(_ modelName: String) -> Bool {
        let normalized = modelName.lowercased()
        return normalized.hasPrefix("whisper-") || normalized.hasPrefix("funasr-") ||
            normalized.hasPrefix("mlx-community/whisper") || normalized.hasPrefix("mlx-community/fun-asr")
    }

    private static func isTTSModelRepo(_ modelName: String) -> Bool {
        let normalized = modelName.lowercased()
        let repoIds = TTSModelKind.allCases.flatMap { TTSModelResolver.repoIDs(for: $0) }
        return repoIds.contains(where: { $0.lowercased() == normalized })
    }

    private static func modelMetadataExists(for modelName: String) -> Bool {
        let modelDir = getModelDirectory(for: modelName)
        let metaPath = modelDir.appendingPathComponent(".swama-meta.json").path
        return FileManager.default.fileExists(atPath: metaPath)
    }

    private static func removeModelDirectoryIfPresent(for modelName: String) -> Bool {
        let modelDir = getModelDirectory(for: modelName)
        let metaPath = modelDir.appendingPathComponent(".swama-meta.json").path
        guard FileManager.default.fileExists(atPath: metaPath) else {
            return false
        }

        try? FileManager.default.removeItem(at: modelDir)
        return true
    }
}

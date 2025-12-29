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

    /// The audio models cache directory (used by MLXAudio/Hub)
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
        // Check if it's an audio model - use audio cache directory
        if modelName.hasPrefix("whisper-") || modelName.hasPrefix("funasr-") || modelName
            .hasPrefix("mlx-community/whisper") || modelName.hasPrefix("mlx-community/SenseVoice")
        {
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
        // Check if it's an audio model
        if modelName.hasPrefix("whisper-") || modelName.hasPrefix("funasr-") || modelName
            .hasPrefix("mlx-community/whisper") || modelName.hasPrefix("mlx-community/SenseVoice")
        {
            // Audio models use HuggingFace Hub format
            let hubFormattedName = modelName.replacingOccurrences(of: "/", with: "--")
            let audioModelDir = audioModelsDirectory.appendingPathComponent("models--\(hubFormattedName)")

            if FileManager.default.fileExists(atPath: audioModelDir.path) {
                try FileManager.default.removeItem(at: audioModelDir)
                return true
            }
            return false
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
}

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

    /// Get the local directory path for a specific model, checking both preferred and legacy locations
    /// Returns the first location where the model exists, or the preferred location if neither exists
    public static func getModelDirectory(for modelName: String) -> URL {
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

    /// Check if a model exists locally (in either preferred or legacy location)
    public static func modelExistsLocally(_ modelName: String) -> Bool {
        let customPath = customModelsDirectory?.appendingPathComponent(modelName)
        let preferredPath = preferredModelsDirectory.appendingPathComponent(modelName)
        let legacyPath = legacyModelsDirectory.appendingPathComponent(modelName)

        return FileManager.default
            .fileExists(atPath: customPath?.appendingPathComponent(".swama-meta.json").path ?? "") ||
            FileManager.default.fileExists(atPath: preferredPath.appendingPathComponent(".swama-meta.json").path) ||
            FileManager.default.fileExists(atPath: legacyPath.appendingPathComponent(".swama-meta.json").path)
    }

    /// Get all directories that should be scanned for models
    public static var allModelsDirectories: [URL] {
        var directories = [preferredModelsDirectory, legacyModelsDirectory]
        if let customDirectory = customModelsDirectory {
            directories.insert(customDirectory, at: 0)
        }
        return directories
    }

    /// Remove a model from disk
    /// Returns true if model was found and deleted, false if model wasn't found
    public static func removeModel(_ modelName: String) throws -> Bool {
        // Check all possible model locations in priority order
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

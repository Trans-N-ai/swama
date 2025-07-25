import Foundation

// MARK: - ModelPaths

/// Centralized configuration for model storage paths
public enum ModelPaths {
    /// The custom path for storing models
    public static let customModelsDirectory: URL? = {
        if let customPath = ProcessInfo.processInfo.environment["SWAMA_MODELS"],
           !customPath.isEmpty
        {
            return URL(fileURLWithPath: customPath)
        }
        return nil
    }()

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

    /// Local directory for models
    public static let localModelsDirectory: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".swama/models/local")
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
            return metaURL
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

    private enum ModelMetadataError: Error, CustomStringConvertible {
        case fileNotFound(URL)
        case invalidFormat(URL)
        case jsonDecodingError(URL, Error)

        var description: String {
            switch self {
            case let .fileNotFound(url):
                "Metadata file not found at \(url.path)"
            case let .invalidFormat(url):
                "Invalid JSON format at \(url.path)"
            case let .jsonDecodingError(url, error):
                "Failed to parse JSON at \(url.path): \(error)"
            }
        }
    }
}

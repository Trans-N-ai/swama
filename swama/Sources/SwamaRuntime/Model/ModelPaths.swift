import Foundation

// MARK: - ModelPaths

/// Centralized configuration for model storage paths.
///
/// Every language, vision, or embedding model is identified on disk by a
/// `.swama-meta.json` file inside its directory.
package enum ModelPaths {
    // MARK: - Root directories

    /// The custom path for storing models (dynamically read from environment).
    package static var customModelsDirectory: URL? {
        if let customPath = ProcessInfo.processInfo.environment["SWAMA_MODELS"],
           !customPath.isEmpty
        {
            return URL(fileURLWithPath: customPath)
        }
        return nil
    }

    /// The preferred path for storing models (new installations).
    package static let preferredModelsDirectory: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".swama/models")
    }()

    /// The legacy path for models (for compatibility).
    package static let legacyModelsDirectory: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent("Documents/huggingface/models")
    }()

    /// The actual directory used for storing new models (respects SWAMA_MODELS).
    package static var activeModelsDirectory: URL {
        customModelsDirectory ?? preferredModelsDirectory
    }

    // MARK: - Model directory resolution

    /// Get the local directory for a model, checking custom → preferred → legacy locations.
    package static func getModelDirectory(for modelName: String) -> URL {
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

        // Not found anywhere — return the location new downloads should use.
        if let customPath {
            return customPath
        }
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

    // MARK: - Existence & listing

    /// A model exists locally iff its directory contains a `.swama-meta.json`.
    package static func modelExistsLocally(_ modelName: String) -> Bool {
        let metaPath = getModelDirectory(for: modelName).appendingPathComponent(".swama-meta.json").path
        return FileManager.default.fileExists(atPath: metaPath)
    }

    /// All directories that should be scanned for models.
    package static var allModelsDirectories: [URL] {
        var directories = [preferredModelsDirectory, legacyModelsDirectory]
        if let customDirectory = customModelsDirectory {
            directories.insert(customDirectory, at: 0)
        }
        return directories
    }

    // MARK: - Removal

    /// Remove a model from disk. Returns true if found and deleted, false otherwise.
    package static func removeModel(_ modelName: String) throws -> Bool {
        // Check all candidate locations in priority order for a metadata file.
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

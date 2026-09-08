import Foundation

// MARK: - ModelPathError

enum ModelPathError: Error {
    case invalidRelativePath
}

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
           customPath.isEmpty == false
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

    /// Model IDs accepted by the Core/runtime boundary. They are either a local alias or a
    /// Hugging Face-style `owner/repository` identifier. Keeping this grammar here makes every
    /// filesystem-backed operation share the same traversal boundary.
    package static func isValidModelIdentifier(_ modelName: String) -> Bool {
        guard modelName == modelName.trimmingCharacters(in: .whitespacesAndNewlines),
              modelName.isEmpty == false,
              modelName.utf8.count <= 192,
              modelName.hasPrefix("/") == false,
              modelName.contains("\\") == false,
              modelName.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains) == false
        else {
            return false
        }

        let components = modelName.split(separator: "/", omittingEmptySubsequences: false)
        guard (1 ... 2).contains(components.count) else {
            return false
        }

        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        return components.allSatisfy { component in
            component != "." &&
                component != ".." &&
                component.isEmpty == false &&
                component.unicodeScalars.allSatisfy(allowed.contains)
        }
    }

    /// Resolve a relative path below `root`, rejecting traversal before any filesystem access.
    static func containedURL(in root: URL, relativePath: String) throws -> URL {
        guard relativePath.isEmpty == false,
              relativePath.hasPrefix("/") == false,
              relativePath.contains("\\") == false,
              relativePath.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains) == false
        else {
            throw ModelPathError.invalidRelativePath
        }

        let components = relativePath.split(separator: "/", omittingEmptySubsequences: false)
        guard components.allSatisfy({ $0.isEmpty == false && $0 != "." && $0 != ".." }) else {
            throw ModelPathError.invalidRelativePath
        }

        let standardizedRoot = root.standardizedFileURL
        let candidate = standardizedRoot.appendingPathComponent(relativePath).standardizedFileURL
        guard isDescendant(candidate, of: standardizedRoot),
              existingPrefixesStayContained(
                  components: components,
                  standardizedRoot: standardizedRoot
              )
        else {
            throw ModelPathError.invalidRelativePath
        }

        return candidate
    }

    static func containedModelDirectory(in root: URL, modelName: String) throws -> URL {
        guard isValidModelIdentifier(modelName) else {
            throw ModelPathError.invalidRelativePath
        }

        return try containedURL(in: root, relativePath: modelName)
    }

    /// Get the local directory for a model, checking custom → preferred → legacy locations.
    package static func getModelDirectory(for modelName: String) -> URL {
        let customRoot = customModelsDirectory
        let customPath = customRoot.flatMap {
            try? containedModelDirectory(in: $0, modelName: modelName)
        }
        let preferredPath = try? containedModelDirectory(
            in: preferredModelsDirectory,
            modelName: modelName
        )
        let legacyPath = try? containedModelDirectory(
            in: legacyModelsDirectory,
            modelName: modelName
        )

        // Check if model exists in custom location first
        if let customPath,
           let customRoot,
           FileManager.default.fileExists(atPath: customPath.appendingPathComponent(".swama-meta.json").path)
        {
            return parseModelMetadataPath(
                from: customPath.appendingPathComponent(".swama-meta.json"),
                within: customRoot
            )
        }

        // Check if model exists in preferred location
        if let preferredPath,
           FileManager.default.fileExists(atPath: preferredPath.appendingPathComponent(".swama-meta.json").path)
        {
            return parseModelMetadataPath(
                from: preferredPath.appendingPathComponent(".swama-meta.json"),
                within: preferredModelsDirectory
            )
        }

        // Check if model exists in legacy location
        if let legacyPath,
           FileManager.default.fileExists(atPath: legacyPath.appendingPathComponent(".swama-meta.json").path)
        {
            return legacyPath
        }

        // Not found anywhere — return the location new downloads should use.
        if let customPath {
            return customPath
        }
        return preferredPath ?? preferredModelsDirectory.appendingPathComponent(".invalid-model-identifier")
    }

    private static func parseModelMetadataPath(from metaURL: URL, within root: URL) -> URL {
        guard let data = try? Data(contentsOf: metaURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let path = json["path"] as? String
        else {
            return metaURL.deletingLastPathComponent()
        }

        let standardizedRoot = root.standardizedFileURL
        let candidate = URL(fileURLWithPath: path).standardizedFileURL
        guard isDescendant(candidate, of: standardizedRoot),
              isDescendant(candidate.resolvingSymlinksInPath(), of: standardizedRoot.resolvingSymlinksInPath())
        else {
            return metaURL.deletingLastPathComponent()
        }

        return candidate
    }

    private static func isDescendant(_ candidate: URL, of root: URL) -> Bool {
        let rootPrefix = root.path.hasSuffix("/") ? root.path : root.path + "/"
        return candidate.path.hasPrefix(rootPrefix)
    }

    private static func existingPrefixesStayContained(
        components: [Substring],
        standardizedRoot: URL
    ) -> Bool {
        let resolvedRoot = standardizedRoot.resolvingSymlinksInPath()
        var prefix = standardizedRoot
        for component in components {
            prefix.appendPathComponent(String(component))
            guard FileManager.default.fileExists(atPath: prefix.path) else {
                break
            }
            guard isDescendant(prefix.resolvingSymlinksInPath(), of: resolvedRoot) else {
                return false
            }
        }
        return true
    }

    // MARK: - Existence & listing

    /// A model exists locally iff its directory contains a `.swama-meta.json`.
    package static func modelExistsLocally(_ modelName: String) -> Bool {
        guard isValidModelIdentifier(modelName) else {
            return false
        }

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
        guard isValidModelIdentifier(modelName) else {
            throw ModelPathError.invalidRelativePath
        }

        let roots = [customModelsDirectory, preferredModelsDirectory, legacyModelsDirectory]
            .compactMap(\.self)
        let locations = try roots.map {
            try containedModelDirectory(in: $0, modelName: modelName)
        }

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

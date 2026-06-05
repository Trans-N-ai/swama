import Foundation

// MARK: - ModelPaths

/// Centralized configuration for model storage paths.
///
/// Design: every model (LLM, STT, TTS) is identified on disk by a `.swama-meta.json`
/// file inside its directory. The only thing that differs between model families is
/// *where* that directory lives — and that single decision is made by
/// `getModelDirectory(for:)`. Audio (STT/TTS) models must follow the layout that
/// mlx-audio-swift's `ModelUtils` hard-codes, which is expressed once in
/// `audioModelDirectory(for:)`. Everything else (existence, listing, removal) is
/// uniform across families.
public enum ModelPaths {
    // MARK: - Root directories

    /// The custom path for storing models (dynamically read from environment).
    public static var customModelsDirectory: URL? {
        if let customPath = ProcessInfo.processInfo.environment["SWAMA_MODELS"],
           !customPath.isEmpty
        {
            return URL(fileURLWithPath: customPath)
        }
        return nil
    }

    /// The preferred path for storing models (new installations).
    public static let preferredModelsDirectory: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".swama/models")
    }()

    /// The legacy path for models (for compatibility).
    public static let legacyModelsDirectory: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent("Documents/huggingface/models")
    }()

    /// The actual directory used for storing new models (respects SWAMA_MODELS).
    public static var activeModelsDirectory: URL {
        customModelsDirectory ?? preferredModelsDirectory
    }

    // MARK: - Audio (STT/TTS) layout

    /// On-disk directory for an audio (STT/TTS) model.
    ///
    /// This MUST match the layout hard-coded in mlx-audio-swift's `ModelUtils`:
    /// `{cacheRoot}/mlx-audio/{repo-with-slashes-as-underscores}`, where the cache
    /// root is `activeModelsDirectory` (handed to the library as `HubCache.cacheDirectory`
    /// in `SpeechToTextRunner` / `TTSRunner`). This is the single source of truth shared
    /// by download, load, list and remove — change the layout here and nowhere else.
    public static func audioModelDirectory(for repo: String) -> URL {
        activeModelsDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(repo.replacingOccurrences(of: "/", with: "_"))
    }

    /// If `modelName` is an audio (STT) or TTS model, return the canonical repo id used
    /// for its on-disk directory; otherwise `nil` (i.e. it's a regular LLM/VLM model).
    static func audioRepo(for modelName: String) -> String? {
        if let tts = TTSModelResolver.resolve(modelName) {
            return TTSModelResolver.repository(for: tts.kind)
        }
        if ModelAliasResolver.isAudioModel(modelName) {
            return ModelAliasResolver.resolve(name: modelName)
        }
        return nil
    }

    // MARK: - Model directory resolution

    /// Get the local directory for a model. Audio/TTS models route to the mlx-audio
    /// cache layout; LLM/VLM models use the standard `{org}/{model}` layout, checking
    /// custom → preferred → legacy locations.
    public static func getModelDirectory(for modelName: String) -> URL {
        // Audio (STT) and TTS models share the mlx-audio cache layout.
        if let repo = audioRepo(for: modelName) {
            return audioModelDirectory(for: repo)
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
    /// Uniform across LLM, STT and TTS — the family-specific directory choice is
    /// fully encapsulated by `getModelDirectory(for:)`.
    public static func modelExistsLocally(_ modelName: String) -> Bool {
        let metaPath = getModelDirectory(for: modelName).appendingPathComponent(".swama-meta.json").path
        return FileManager.default.fileExists(atPath: metaPath)
    }

    /// All directories that should be scanned for models. The mlx-audio subdirectory
    /// lives under `activeModelsDirectory`, so it is covered by scanning these roots.
    public static var allModelsDirectories: [URL] {
        var directories = [preferredModelsDirectory, legacyModelsDirectory]
        if let customDirectory = customModelsDirectory {
            directories.insert(customDirectory, at: 0)
        }
        return directories
    }

    // MARK: - Removal

    /// Remove a model from disk. Returns true if found and deleted, false otherwise.
    public static func removeModel(_ modelName: String) throws -> Bool {
        // Audio / TTS: single canonical directory.
        if let repo = audioRepo(for: modelName) {
            let dir = audioModelDirectory(for: repo)
            guard FileManager.default.fileExists(atPath: dir.path) else {
                return false
            }

            try FileManager.default.removeItem(at: dir)
            return true
        }

        // LLM/VLM: check all candidate locations in priority order for a metadata file.
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

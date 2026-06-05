import Foundation

// MARK: - ModelManager

/// Manages available MLX models, providing functionalities to list and load them.
public enum ModelManager {
    /// Lists all locally available models.
    /// It scans both preferred and legacy directories for models and their metadata.
    public static func models() -> [ModelInfo] {
        var modelInfos: [ModelInfo] = []

        // Scan all model directories (both preferred and legacy)
        for modelsRootDirectory in ModelPaths.allModelsDirectories {
            let directoryModels = scanModelsDirectory(at: modelsRootDirectory)
            modelInfos.append(contentsOf: directoryModels)
        }

        return modelInfos
    }

    /// Scans a models directory and returns ModelInfo array.
    ///
    /// A model is recognised solely by a `.swama-meta.json` file in its directory —
    /// uniform across LLM, STT and TTS. Two on-disk shapes are supported:
    ///   • flat:   `{root}/{model}/.swama-meta.json`
    ///   • nested: `{root}/{org}/{model}/.swama-meta.json`
    /// The nested case also covers the audio layout `{root}/mlx-audio/{repo_underscore}/`.
    private static func scanModelsDirectory(at modelsRootDirectory: URL) -> [ModelInfo] {
        var modelInfos: [ModelInfo] = []

        let topLevel: [URL]
        do {
            topLevel = try FileManager.default.contentsOfDirectory(
                at: modelsRootDirectory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
        }
        catch {
            // Only log error if the directory should exist (not for optional paths)
            if FileManager.default.fileExists(atPath: modelsRootDirectory.path) {
                fputs(
                    "SwamaKit.ModelManager: Error reading models root directory \(modelsRootDirectory.path): \(error.localizedDescription)\n",
                    stderr
                )
            }
            return [] // Return empty if the root directory is inaccessible
        }

        for entry in topLevel {
            guard entry.hasDirectoryPath else {
                continue
            }

            // Flat layout: {root}/{model}/.swama-meta.json
            let flatMeta = entry.appendingPathComponent(".swama-meta.json")
            if FileManager.default.fileExists(atPath: flatMeta.path) {
                if let info = parseModelMetadata(metaURL: flatMeta, fallbackID: entry.lastPathComponent) {
                    modelInfos.append(info)
                }
                continue
            }

            // Nested layout: {root}/{org}/{model}/.swama-meta.json
            let children = (try? FileManager.default.contentsOfDirectory(
                at: entry,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )) ?? []

            for child in children {
                guard child.hasDirectoryPath else {
                    continue
                }

                let metaURL = child.appendingPathComponent(".swama-meta.json")
                guard FileManager.default.fileExists(atPath: metaURL.path) else {
                    continue
                }

                let fallbackID = "\(entry.lastPathComponent)/\(child.lastPathComponent)"
                if let info = parseModelMetadata(metaURL: metaURL, fallbackID: fallbackID) {
                    modelInfos.append(info)
                }
            }
        }

        return modelInfos
    }

    /// Parse a `.swama-meta.json`. The displayed model id comes from the file's `id`
    /// field (the canonical repo, e.g. `mlx-community/Qwen3-ASR-1.7B-bf16`); the
    /// directory-derived `fallbackID` is used only when the file omits it. This matters
    /// for audio models whose directory name is the underscore-joined repo, which cannot
    /// be reliably reversed back into `{org}/{model}`.
    private static func parseModelMetadata(metaURL: URL, fallbackID: String) -> ModelInfo? {
        do {
            let data = try Data(contentsOf: metaURL)
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let created = json["created"] as? Int,
                  let size = (json["size_in_bytes"] as? NSNumber)?.int64Value
            else {
                fputs("SwamaKit.ModelManager: Invalid .swama-meta.json for \(fallbackID)\n", stderr)
                return nil
            }

            let id = (json["id"] as? String) ?? fallbackID
            return ModelInfo(
                id: id,
                created: created,
                sizeInBytes: size,
                source: .metaFile,
                rawMetadata: json
            )
        }
        catch {
            fputs(
                "SwamaKit.ModelManager: Error reading .swama-meta.json for \(fallbackID): \(error.localizedDescription)\n",
                stderr
            )
            return nil
        }
    }
}

// MARK: - MetadataSource

/// Describes the source of the model's metadata.
public enum MetadataSource {
    case metaFile
    case directoryScan
}

// MARK: - ModelInfo

/// Basic information about a model.
public struct ModelInfo: Identifiable { // Added Identifiable for potential UI usage
    public let id: String
    public let created: Int // Unix timestamp of creation
    public let sizeInBytes: Int64
    public let source: MetadataSource
    public let rawMetadata: [String: Any]? // Full metadata from .swama-meta.json, if available

    public init(
        id: String,
        created: Int,
        sizeInBytes: Int64,
        source: MetadataSource,
        rawMetadata: [String: Any]? = nil
    ) {
        self.id = id
        self.created = created
        self.sizeInBytes = sizeInBytes
        self.source = source
        self.rawMetadata = rawMetadata
    }
}

/// Utility extension to check if a URL points to a directory.
private extension URL {
    var hasDirectoryPath: Bool {
        (try? resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
    }
}

/// Extension for URLSession to provide modern async data fetching.
@available(macOS 10.15, iOS 13.0, watchOS 6.0, tvOS 13.0, *)
extension URLSession {
    /// Asynchronously fetches data from a URL.
    /// This is a convenience wrapper around the instance method.
    func data(from url: URL) async throws -> (Data, URLResponse) {
        try await withCheckedThrowingContinuation { continuation in
            let task = self.dataTask(with: url) { data, response, error in
                if let error {
                    continuation.resume(throwing: error)
                }
                else if let data, let response {
                    continuation.resume(returning: (data, response))
                }
                else {
                    continuation.resume(throwing: URLError(.unknown))
                }
            }
            task.resume()
        }
    }
}

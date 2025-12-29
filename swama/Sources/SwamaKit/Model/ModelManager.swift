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

    /// Scans a specific models directory and returns ModelInfo array
    private static func scanModelsDirectory(at modelsRootDirectory: URL) -> [ModelInfo] {
        var modelInfos: [ModelInfo] = []

        // Determine if this is the audio models directory (no .swama-meta.json files)
        let isAudioDirectory = modelsRootDirectory.path.contains("huggingface/models")

        let orgDirs: [URL]
        do {
            orgDirs = try FileManager.default.contentsOfDirectory(
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

        // Iterate through organization directories (e.g., mlx-community)
        for orgDir in orgDirs {
            guard orgDir.hasDirectoryPath else {
                continue
            }

            let modelNameDirs: [URL]
            do {
                modelNameDirs = try FileManager.default.contentsOfDirectory(
                    at: orgDir,
                    includingPropertiesForKeys: [.isDirectoryKey, .creationDateKey],
                    options: [.skipsHiddenFiles]
                )
            }
            catch {
                fputs(
                    "SwamaKit.ModelManager: Error reading contents of org directory \(orgDir.path): \(error.localizedDescription)\n",
                    stderr
                )
                continue
            }

            for modelDir in modelNameDirs {
                guard modelDir.hasDirectoryPath else {
                    continue
                }

                let modelID = "\(orgDir.lastPathComponent)/\(modelDir.lastPathComponent)"
                let metaURL = modelDir.appendingPathComponent(".swama-meta.json")

                if FileManager.default.fileExists(atPath: metaURL.path) {
                    // LLM model with metadata file
                    if let info = parseModelMetadata(metaURL: metaURL, modelID: modelID) {
                        modelInfos.append(info)
                    }
                }
                else if isAudioDirectory {
                    // Audio model without metadata file - create info from directory scan
                    if let attributes = try? FileManager.default.attributesOfItem(atPath: modelDir.path),
                       let creationDate = attributes[.creationDate] as? Date
                    {
                        let size = directorySize(at: modelDir)
                        let created = Int(creationDate.timeIntervalSince1970)

                        modelInfos.append(ModelInfo(
                            id: modelID,
                            created: created,
                            sizeInBytes: size,
                            source: .directoryScan,
                            rawMetadata: nil
                        ))
                    }
                }
            }
        }

        // Additional: support flat model directories (e.g., ~/.swama/models/llama2/.swama-meta.json)
        // Only for non-audio directories
        if !isAudioDirectory {
            for modelDir in orgDirs {
                let metaURL = modelDir.appendingPathComponent(".swama-meta.json")
                if FileManager.default.fileExists(atPath: metaURL.path) {
                    let modelID = modelDir.lastPathComponent
                    if let info = parseModelMetadata(metaURL: metaURL, modelID: modelID) {
                        modelInfos.append(info)
                    }
                }
            }
        }

        return modelInfos
    }

    /// Calculate total size of a directory recursively
    private static func directorySize(at url: URL) -> Int64 {
        var totalSize: Int64 = 0

        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]
        )
        else {
            return 0
        }

        for case let fileURL as URL in enumerator {
            guard let resourceValues = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
                  let isRegularFile = resourceValues.isRegularFile,
                  isRegularFile,
                  let fileSize = resourceValues.fileSize
            else {
                continue
            }

            totalSize += Int64(fileSize)
        }

        return totalSize
    }

    private static func parseModelMetadata(metaURL: URL, modelID: String) -> ModelInfo? {
        do {
            let data = try Data(contentsOf: metaURL)
            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let created = json["created"] as? Int,
                  let size = (json["size_in_bytes"] as? NSNumber)?.int64Value
            else {
                fputs("SwamaKit.ModelManager: Invalid .swama-meta.json for \(modelID)\n", stderr)
                return nil
            }

            return ModelInfo(
                id: modelID,
                created: created,
                sizeInBytes: size,
                source: .metaFile,
                rawMetadata: json
            )
        }
        catch {
            fputs(
                "SwamaKit.ModelManager: Error reading .swama-meta.json for \(modelID): \(error.localizedDescription)\n",
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

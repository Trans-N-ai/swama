import Foundation

// MARK: - ModelManager

/// Manages available MLX models, providing functionalities to list and load them.
public enum ModelManager {
    /// Lists all locally available models.
    /// It scans a predefined directory structure for models and their metadata.
    public static func models() -> [ModelInfo] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        // Standard directory for storing Hugging Face models.
        let modelsRootDirectory = home.appendingPathComponent("Documents")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("models")

        var modelInfos: [ModelInfo] = []
        let orgDirs: [URL]
        do {
            orgDirs = try FileManager.default.contentsOfDirectory(
                at: modelsRootDirectory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )
        }
        catch {
            fputs(
                "SwamaKit.ModelManager: Error reading models root directory \(modelsRootDirectory.path): \(error.localizedDescription)\n",
                stderr
            )
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
                    do {
                        let data = try Data(contentsOf: metaURL)
                        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                        if let created = json?["created"] as? Int,
                           let sizeInBytes = (json?["size_in_bytes"] as? NSNumber)?.int64Value
                        {
                            modelInfos.append(ModelInfo(
                                id: modelID,
                                created: created,
                                sizeInBytes: sizeInBytes,
                                source: .metaFile,
                                rawMetadata: json
                            ))
                        }
                        else {
                            // Meta file exists but content is invalid or incomplete. Do not list.
                            fputs(
                                "SwamaKit.ModelManager: Invalid or incomplete .swama-meta.json for \(modelID). Model will not be listed.\n",
                                stderr
                            )
                        }
                    }
                    catch {
                        // Error reading or parsing .swama-meta.json. Do not list.
                        fputs(
                            "SwamaKit.ModelManager: Error reading or parsing .swama-meta.json for \(modelID): \(error.localizedDescription). Model will not be listed.\n",
                            stderr
                        )
                    }
                }
            }
        }
        return modelInfos
    }

    /// Loads detailed information for a specific model by its ID.
    public static func loadModel(id: String) throws -> LoadedModel {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let modelsBaseDirectory = home.appendingPathComponent("Documents")
            .appendingPathComponent("huggingface")
            .appendingPathComponent("models")
        let modelDir = modelsBaseDirectory.appendingPathComponent(id)
        let configURL = modelDir.appendingPathComponent("config.json")
        let tokenizerURL = modelDir.appendingPathComponent("tokenizer.json")

        // Find weight files with common extensions
        let weightFiles = try? FileManager.default
            .contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            .filter { ["safetensors", "bin", "pth", "npz", "mlx"].contains($0.pathExtension) }

        let configData = try? Data(contentsOf: configURL)
        let configJSON = configData.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }

        var tokenizerJSON: [String: Any]? = nil
        if FileManager.default.fileExists(atPath: tokenizerURL.path) {
            let tokenizerData = try? Data(contentsOf: tokenizerURL)
            tokenizerJSON = tokenizerData.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
        }
        return LoadedModel(
            id: id,
            config: configJSON,
            tokenizer: tokenizerJSON,
            weightFiles: weightFiles ?? []
        )
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

// MARK: - LoadedModel

/// Detailed information about a loaded model, including configuration and tokenizer data.
public struct LoadedModel {
    public let id: String
    public let config: [String: Any]? // Model configuration JSON
    public let tokenizer: [String: Any]? // Tokenizer configuration JSON
    public let weightFiles: [URL] // URLs to model weight files

    public init(id: String, config: [String: Any]?, tokenizer: [String: Any]?, weightFiles: [URL]) {
        self.id = id
        self.config = config
        self.tokenizer = tokenizer
        self.weightFiles = weightFiles
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

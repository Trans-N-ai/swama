import Foundation

// MARK: - ModelDownloader

let Downloaders: [String: AnyClass] = [
    "MODEL_SCOPE": ModelScopeDownloader.self,
    "HUGGING_FACE": HuggingFaceDownloader.self
]

// MARK: - ModelDownloader

public enum ModelDownloader {
    // MARK: Public

    public static func downloadModel(resolvedModelName: String) async throws {
        printMessage("Pulling model: \(resolvedModelName)")

        let modelDir = ModelPaths.getModelDirectory(for: resolvedModelName)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        let swamaRegistry = ProcessInfo.processInfo.environment["SWAMA_REGISTRY"] ?? "HUGGING_FACE"

        guard let downloaderClass: IDownloader.Type = Downloaders[swamaRegistry] as? IDownloader.Type else {
            throw NSError(
                domain: "ModelDownloader",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey: "Invalid registry: \(swamaRegistry). Supported: MODEL_SCOPE, HUGGING_FACE"
                ]
            )
        }

        printMessage("Using downloader: \(swamaRegistry)")
        let downloader = downloaderClass.init()
        let fileInfos = try await downloader.listModelFilesWithSize(repo: resolvedModelName)
        let allowedExtensions = [
            "safetensors",
            "bin",
            "json",
            "model",
            "txt",
            "pt",
            "params",
            "tiktoken",
            "vocab",
            "jinja"
        ]
        let filteredFileInfos = fileInfos.filter { info in
            allowedExtensions.contains(where: { info.path.hasSuffix(".\($0)") })
        }

        if filteredFileInfos.isEmpty, !fileInfos.isEmpty {
            printMessage(
                "Warning: No files with allowed extensions found for model \(resolvedModelName). Allowed: \(allowedExtensions.joined(separator: ", "))"
            )
        }

        if filteredFileInfos.isEmpty, fileInfos.isEmpty {
            printMessage("Warning: No files found on Hugging Face repo for model \(resolvedModelName).")
            throw NSError(
                domain: "ModelDownloader",
                code: 2,
                userInfo: [
                    NSLocalizedDescriptionKey: "No files found on Hugging Face repository for model \(resolvedModelName)."
                ]
            )
        }

        for (idx, info) in filteredFileInfos.enumerated() {
            let file = info.path
            let remoteSize = info.size
            let dest = modelDir.appendingPathComponent(file)

            try FileManager.default.createDirectory(
                at: dest.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            var localSize: Int64 = 0
            if FileManager.default.fileExists(atPath: dest.path) {
                localSize = (try? FileManager.default.attributesOfItem(atPath: dest.path)[.size] as? NSNumber)?
                    .int64Value ?? 0
            }

            if localSize == remoteSize, remoteSize > 0 {
                printMessage("[\(idx + 1)/\(filteredFileInfos.count)] Exists: \(file), skip")
                continue
            }

            printMessage("[\(idx + 1)/\(filteredFileInfos.count)] Downloading: \(file)")
            try await downloader.downloadWithResume(
                repo: resolvedModelName,
                file: file,
                dest: dest,
                totalSize: remoteSize,
                localSize: localSize
            )
        }

        printMessage("✅ Model pull complete: \(resolvedModelName)")
        try writeModelMetadata(modelName: resolvedModelName, modelDir: modelDir)
    }

    public static func fetchModel(modelName: String) async throws -> String {
        let resolved = ModelAliasResolver.resolve(name: modelName)
        let repoIds = resolveRepoIDs(resolvedName: resolved)
        let allExists = repoIds.allSatisfy { ModelPaths.modelExistsLocally($0) }
        
        // Check if model already exists locally
        if allExists {
            let label = TTSModelResolver.resolve(resolved) != nil
                ? "TTS"
                : (ModelAliasResolver.isAudioModel(resolved) ? "Audio" : "Model")
            print("✅ \(label) already exists: \(resolved)")
            return resolved
        }

        // Handle legacy WhisperKit models (if still needed for compatibility)
        if resolved.hasPrefix("openai_whisper") {
            throw ModelPoolError.failedToLoadModel(
                modelName,
                NSError(
                    domain: "SwamaKit.Audio",
                    code: 0,
                    userInfo: [
                        NSLocalizedDescriptionKey: "WhisperKit models are no longer supported. Use MLXAudio whisper models instead (e.g., whisper-large-turbo)"
                    ]
                )
            )
        }

        // Download the model (works for LLM, Audio, and TTS models)
        for repoId in repoIds {
            try await ModelDownloader.downloadModel(resolvedModelName: repoId)
        }

        return resolved
    }

    private static func resolveRepoIDs(resolvedName: String) -> [String] {
        if let ttsModel = TTSModelResolver.resolve(resolvedName) {
            return TTSModelResolver.repoIDs(for: ttsModel.kind)
        }
        return [resolvedName]
    }

    // MARK: Internal

    public static func printMessage(_ message: String) {
        // Use fputs to stdout to behave like print
        fputs(message + "\n", stdout)
        fflush(stdout)
    }

    static func writeModelMetadata(modelName: String, modelDir: URL) throws {
        let size = try calculateFolderSize(at: modelDir)
        let created = Int(Date().timeIntervalSince1970)

        // Write metadata file directly to the model directory
        let metaURL = modelDir.appendingPathComponent(".swama-meta.json")

        let metadata: [String: Any] = [
            "id": modelName,
            "object": "model",
            "created": created,
            "owned_by": "swama",
            "size_in_bytes": size,
            "path": modelDir.path
        ]

        let data = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted])
        try data.write(to: metaURL)
        printMessage("📝 Metadata written to .swama-meta.json")
    }

    public static func calculateFolderSize(at url: URL) throws -> Int64 {
        var total: Int64 = 0
        let resourceKeys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey]
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: resourceKeys,
            options: [.skipsHiddenFiles]
        )
        else {
            return 0
        }

        for case let fileURL as URL in enumerator {
            do {
                let values = try fileURL.resourceValues(forKeys: Set(resourceKeys))
                if values.isRegularFile == true {
                    total += Int64(values.fileSize ?? 0)
                }
            }
            catch {
                fputs(
                    "SwamaKit.ModelDownloader: Error getting resource values for \(fileURL.path): \(error.localizedDescription)\n",
                    stderr
                )
            }
        }
        return total
    }

    // MARK: - Main Download Functions
}

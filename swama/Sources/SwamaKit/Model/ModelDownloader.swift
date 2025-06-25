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
        let allowedExtensions = ["safetensors", "bin", "json", "model", "txt", "pt", "params", "tiktoken", "vocab"]
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

    public static func downloadWhisperKitModel(alias: String) async throws {
        guard let modelFolderName = ModelAliasResolver.whisperKitAliases[alias.lowercased()] else {
            throw NSError(
                domain: "ModelDownloader",
                code: 13,
                userInfo: [NSLocalizedDescriptionKey: "Unknown WhisperKit model: \(alias)"]
            )
        }

        let swamaRegistry = ProcessInfo.processInfo.environment["SWAMA_REGISTRY"] ?? "HUGGING_FACE"

        let modelDir = ModelPaths.getModelDirectory(for: "whisperkit/\(modelFolderName)")
        // Create the directory if it doesn't exist
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        printMessage("Pulling model: \(alias)")
        let openaiModelName = getOpenAIModelNameFromFolderName(modelFolderName)
        guard let downloaderClass: IDownloader.Type = Downloaders[swamaRegistry] as? IDownloader.Type else {
            throw NSError(
                domain: "ModelDownloader",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey: "Invalid registry: \(swamaRegistry). Supported: MODEL_SCOPE, HUGGING_FACE"
                ]
            )
        }

        let downloader = downloaderClass.init()
        let allFiles = try await downloader.listWhisperKitModelFile(
            modelDir: modelDir,
            modelFolderName: modelFolderName,
            openaiModelName: openaiModelName
        )

        // Download all files with unified progress display and resume support
        let totalFiles = allFiles.count
        for (index, fileInfo) in allFiles.enumerated() {
            let fileIndex = index + 1
            do {
                // Get remote file size first
                let remoteSize = try await downloader.getWhisperKitFileSize(url: fileInfo.url)

                // Check local file size
                var localSize: Int64 = 0
                if FileManager.default.fileExists(atPath: fileInfo.localPath.path) {
                    localSize = (try? FileManager.default
                        .attributesOfItem(atPath: fileInfo.localPath.path)[.size] as? NSNumber
                    )?
                        .int64Value ?? 0
                }

                // Skip if file is complete
                if localSize == remoteSize, remoteSize > 0 {
                    printMessage("[\(fileIndex)/\(totalFiles)] Exists: \(fileInfo.fileName), skip")
                    continue
                }

                // Download with resume support
                printMessage("[\(fileIndex)/\(totalFiles)] Downloading: \(fileInfo.fileName)")
                try await downloader.downloadWhisperKitFileWithResume(
                    from: fileInfo.url,
                    to: fileInfo.localPath,
                    totalSize: remoteSize,
                    localSize: localSize
                )
            }
            catch {
                // Check if it's a 404 error
                let errorDescription = error.localizedDescription
                if errorDescription.contains("HTTP 404") {
                    printMessage("[\(fileIndex)/\(totalFiles)] Exists: \(fileInfo.fileName), skip")
                    continue
                }
                throw error
            }
        }

        printMessage("✅ Model pull complete: \(alias)")

        // Generate metadata file so the model appears in 'swama list'
        try writeModelMetadata(modelName: alias, modelDir: modelDir)
    }

    // MARK: Internal

    static func printMessage(_ message: String) {
        // Use fputs to stdout to behave like print
        fputs(message + "\n", stdout)
        fflush(stdout)
    }

    static func writeModelMetadata(modelName: String, modelDir: URL) throws {
        let size = try calculateFolderSize(at: modelDir)
        let created = Int(Date().timeIntervalSince1970)
        let metadata: [String: Any] = [
            "id": modelName,
            "object": "model",
            "created": created,
            "owned_by": "swama",
            "size_in_bytes": size
        ]
        let metaURL = modelDir.appendingPathComponent(".swama-meta.json")
        let data = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted])
        try data.write(to: metaURL)
        printMessage("📝 Metadata written to .swama-meta.json")
    }

    static func calculateFolderSize(at url: URL) throws -> Int64 {
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

    // MARK: - WhisperKit Helper Functions

    static func getOpenAIModelName(for alias: String) -> String {
        // Map WhisperKit alias to OpenAI model name
        let lowercasedAlias = alias.lowercased()
        switch lowercasedAlias {
        case "whisper-tiny":
            return "openai/whisper-tiny"
        case "whisper-tiny.en":
            return "openai/whisper-tiny.en"
        case "whisper-base":
            return "openai/whisper-base"
        case "whisper-base.en":
            return "openai/whisper-base.en"
        case "whisper-small":
            return "openai/whisper-small"
        case "whisper-small.en":
            return "openai/whisper-small.en"
        case "whisper-medium":
            return "openai/whisper-medium"
        case "whisper-medium.en":
            return "openai/whisper-medium.en"
        case "whisper-large":
            return "openai/whisper-large-v3" // Default to latest
        case "whisper-large-v2":
            return "openai/whisper-large-v2"
        case "whisper-large-v3":
            return "openai/whisper-large-v3"
        default:
            return "openai/whisper-base" // Fallback
        }
    }

    static func getOpenAIModelNameFromFolderName(_ folderName: String) -> String {
        // Map WhisperKit folder name to OpenAI model name
        switch folderName {
        case "openai_whisper-tiny":
            "openai/whisper-tiny"
        case "openai_whisper-tiny.en":
            "openai/whisper-tiny.en"
        case "openai_whisper-base":
            "openai/whisper-base"
        case "openai_whisper-base.en":
            "openai/whisper-base.en"
        case "openai_whisper-small":
            "openai/whisper-small"
        case "openai_whisper-small.en":
            "openai/whisper-small.en"
        case "openai_whisper-medium":
            "openai/whisper-medium"
        case "openai_whisper-medium.en":
            "openai/whisper-medium.en"
        case "openai_whisper-large-v2":
            "openai/whisper-large-v2"
        case "openai_whisper-large-v3":
            "openai/whisper-large-v3"
        default:
            "openai/whisper-base" // Fallback
        }
    }

    // MARK: - Main Download Functions
}

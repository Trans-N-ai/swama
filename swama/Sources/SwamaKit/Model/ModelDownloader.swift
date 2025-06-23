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
                try await downloadWhisperKitFileWithResume(
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

    static func downloadWhisperKitFileWithResume(
        from urlString: String,
        to localFile: URL,
        totalSize: Int64,
        localSize: Int64
    ) async throws {
        guard let url = URL(string: urlString) else {
            throw NSError(
                domain: "ModelDownloader",
                code: 14,
                userInfo: [NSLocalizedDescriptionKey: "Invalid URL: \(urlString)"]
            )
        }

        var request = URLRequest(url: url)
        var isResuming = false
        if localSize > 0, localSize < totalSize {
            request.setValue("bytes=\(localSize)-", forHTTPHeaderField: "Range")
            isResuming = true
        }

        let (byteStream, response) = try await URLSession.shared.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NSError(
                domain: "ModelDownloader",
                code: 9,
                userInfo: [NSLocalizedDescriptionKey: "Invalid response from server."]
            )
        }

        let statusCode = httpResponse.statusCode
        if !(200 ... 299).contains(statusCode) {
            if statusCode == 404 {
                throw NSError(
                    domain: "ModelDownloader",
                    code: 404,
                    userInfo: [NSLocalizedDescriptionKey: "HTTP 404"]
                )
            }
            throw NSError(
                domain: "ModelDownloader",
                code: 15,
                userInfo: [NSLocalizedDescriptionKey: "HTTP error \(statusCode)"]
            )
        }

        let shouldResume = isResuming && statusCode == 206 // 206 Partial Content

        if !FileManager.default.fileExists(atPath: localFile.path) {
            FileManager.default.createFile(atPath: localFile.path, contents: nil)
        }

        let handle = try FileHandle(forWritingTo: localFile)
        defer { try? handle.close() }

        if shouldResume {
            try handle.seekToEnd()
        }
        else {
            try handle.truncate(atOffset: 0)
        }

        let progressBar = ProgressBar(total: totalSize, initial: shouldResume ? localSize : 0)
        var buffer: [UInt8] = []

        for try await byte in byteStream {
            buffer.append(byte)
            if buffer.count >= 8192 {
                try handle.write(contentsOf: Data(buffer))
                progressBar.update(bytes: Int64(buffer.count))
                buffer.removeAll(keepingCapacity: true)
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: Data(buffer))
            progressBar.update(bytes: Int64(buffer.count))
        }

        progressBar.finish()
    }

    static func downloadWhisperKitFile(
        from urlString: String,
        to localFile: URL,
        description: String,
        fileIndex: Int,
        totalFiles: Int
    ) async throws {
        // Check if file already exists and is not empty
        if FileManager.default.fileExists(atPath: localFile.path) {
            do {
                let attributes = try FileManager.default.attributesOfItem(atPath: localFile.path)
                if let fileSize = attributes[.size] as? Int64, fileSize > 0 {
                    printMessage("[\(fileIndex)/\(totalFiles)] Exists: \(description), skip")
                    return
                }
            }
            catch {
                // If we can't get attributes, assume file is corrupt and re-download
            }
        }

        guard let downloadURL = URL(string: urlString) else {
            throw NSError(
                domain: "ModelDownloader",
                code: 14,
                userInfo: [NSLocalizedDescriptionKey: "Invalid URL: \(urlString)"]
            )
        }

        printMessage("[\(fileIndex)/\(totalFiles)] Downloading: \(description)")

        // Use the same download logic as regular models
        let (byteStream, response) = try await URLSession.shared.bytes(from: downloadURL)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NSError(
                domain: "ModelDownloader",
                code: 9,
                userInfo: [NSLocalizedDescriptionKey: "Invalid response from server for file \(description)."]
            )
        }

        if httpResponse.statusCode == 404 {
            throw NSError(
                domain: "ModelDownloader",
                code: 404,
                userInfo: [NSLocalizedDescriptionKey: "HTTP 404 for \(description)"]
            )
        }
        else if !(200 ... 299).contains(httpResponse.statusCode) {
            throw NSError(
                domain: "ModelDownloader",
                code: 15,
                userInfo: [NSLocalizedDescriptionKey: "HTTP \(httpResponse.statusCode) for \(description)"]
            )
        }

        // Get expected content length for progress bar
        let expectedLength = httpResponse.expectedContentLength > 0 ? httpResponse.expectedContentLength : 1024 * 1024
        let progressBar = ProgressBar(total: expectedLength)

        // Create file for writing
        if !FileManager.default.fileExists(atPath: localFile.path) {
            FileManager.default.createFile(atPath: localFile.path, contents: nil)
        }

        let handle = try FileHandle(forWritingTo: localFile)
        defer { try? handle.close() }

        try handle.truncate(atOffset: 0)

        var buffer: [UInt8] = []
        for try await byte in byteStream {
            buffer.append(byte)
            if buffer.count >= 8192 {
                try handle.write(contentsOf: Data(buffer))
                progressBar.update(bytes: Int64(buffer.count))
                buffer.removeAll(keepingCapacity: true)
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: Data(buffer))
            progressBar.update(bytes: Int64(buffer.count))
        }

        progressBar.finish()
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

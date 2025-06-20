import Foundation

// MARK: - ProgressBar

final class ProgressBar {
    // MARK: Lifecycle

    init(total: Int64, barWidth: Int = 30, initial: Int64 = 0) {
        self.total = total
        self.barWidth = barWidth
        self.downloaded = initial
        self.lastDownloaded = initial
    }

    // MARK: Internal

    func update(bytes: Int64) {
        downloaded += bytes
        let now = Date()
        let interval = now.timeIntervalSince(lastSpeedCheck)
        if interval >= 0.5 {
            let bytesDelta = downloaded - lastDownloaded
            speed = interval > 0 ? Double(bytesDelta) / interval : 0
            lastDownloaded = downloaded
            lastSpeedCheck = now
        }
        if now.timeIntervalSince(lastPrint) >= 0.1 || downloaded == total {
            lastPrint = now
            let percent = Double(downloaded) / Double(total)
            let filled = Int(percent * Double(barWidth))
            let bar =
                if filled < barWidth {
                    String(repeating: "=", count: filled) + ">" +
                        String(repeating: " ", count: max(0, barWidth - filled - 1))
                }
                else {
                    String(repeating: "=", count: barWidth)
                }
            let percentStr = String(format: "%3d%%", Int(percent * 100))
            let speedStr =
                if speed > 1024 * 1024 {
                    String(format: "%.2f MB/s", speed / 1024 / 1024)
                }
                else if speed > 1024 {
                    String(format: "%.2f KB/s", speed / 1024)
                }
                else {
                    String(format: "%.0f B/s", speed)
                }
            // Use fputs to stdout for progress bar to behave like print
            fputs("\r[\(bar)] \(percentStr) \(speedStr)", stdout)
            fflush(stdout)
        }
    }

    func finish() {
        fputs("\n", stdout)
        fflush(stdout)
    }

    // MARK: Private

    private let total: Int64
    private let barWidth: Int
    private var downloaded: Int64
    private var lastPrint: Date = .init(timeIntervalSince1970: 0)
    private var lastDownloaded: Int64
    private var lastSpeedCheck: Date = .init(timeIntervalSince1970: 0)
    private var speed: Double = 0 // bytes/sec
}

// MARK: - ModelDownloader

public enum ModelDownloader {
    // MARK: Public

    public static func downloadModel(resolvedModelName: String) async throws {
        printMessage("Pulling model: \(resolvedModelName)")

        let modelDir = ModelPaths.getModelDirectory(for: resolvedModelName)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let fileInfos = try await listHuggingFaceFilesWithSize(repo: resolvedModelName)
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
            try await downloadWithResume(
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
        guard let huggingFaceFolderName = ModelAliasResolver.whisperKitAliases[alias.lowercased()] else {
            throw NSError(
                domain: "ModelDownloader",
                code: 13,
                userInfo: [NSLocalizedDescriptionKey: "Unknown WhisperKit model: \(alias)"]
            )
        }

        let whisperKitModelsDirectory = FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent(".swama/models/whisperkit")
        let modelDir = whisperKitModelsDirectory.appendingPathComponent(huggingFaceFolderName)

        // Create the directory if it doesn't exist
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        printMessage("Pulling model: \(alias)")

        // Base URLs for different repositories
        let whisperKitBaseURL = "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/main"

        // Collect all files to download with their info
        var allFiles: [(url: String, localPath: URL, fileName: String)] = []

        // Config files
        let configFiles = ["config.json", "generation_config.json"]
        for fileName in configFiles {
            let url = "\(whisperKitBaseURL)/\(huggingFaceFolderName)/\(fileName)"
            let localPath = modelDir.appendingPathComponent(fileName)
            allFiles.append((url: url, localPath: localPath, fileName: fileName))
        }

        // Tokenizer files from OpenAI's original model (required by WhisperKit)
        let openaiModelName = getOpenAIModelNameFromFolderName(huggingFaceFolderName)
        let tokenizerFiles = [
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer_config.json"
        ]
        for fileName in tokenizerFiles {
            let url = "https://huggingface.co/\(openaiModelName)/resolve/main/\(fileName)"
            let localPath = modelDir.appendingPathComponent(fileName)
            allFiles.append((url: url, localPath: localPath, fileName: fileName))
        }

        // MLModelC directories and their files
        let mlmodelcDirs = ["AudioEncoder.mlmodelc", "TextDecoder.mlmodelc", "MelSpectrogram.mlmodelc"]
        for dirName in mlmodelcDirs {
            let modelcDir = modelDir.appendingPathComponent(dirName)
            try FileManager.default.createDirectory(at: modelcDir, withIntermediateDirectories: true)

            // Main files in .mlmodelc directory
            let files = ["coremldata.bin", "metadata.json", "model.mil", "model.mlmodel"]
            for fileName in files {
                let url = "\(whisperKitBaseURL)/\(huggingFaceFolderName)/\(dirName)/\(fileName)"
                let localPath = modelcDir.appendingPathComponent(fileName)
                allFiles.append((url: url, localPath: localPath, fileName: "\(dirName)/\(fileName)"))
            }

            // Subdirectories
            let subdirs = ["analytics", "weights"]
            for subdir in subdirs {
                let subdirLocal = modelcDir.appendingPathComponent(subdir)
                try FileManager.default.createDirectory(at: subdirLocal, withIntermediateDirectories: true)

                let potentialFiles: [String] =
                    if subdir == "analytics" {
                        ["coremldata.bin", "metadata.json"]
                    }
                    else if subdir == "weights" {
                        ["weight.bin", "data.bin", "metadata.json"]
                    }
                    else {
                        ["data.bin", "metadata.json"]
                    }

                for fileName in potentialFiles {
                    let url = "\(whisperKitBaseURL)/\(huggingFaceFolderName)/\(dirName)/\(subdir)/\(fileName)"
                    let localPath = subdirLocal.appendingPathComponent(fileName)
                    allFiles.append((url: url, localPath: localPath, fileName: "\(dirName)/\(subdir)/\(fileName)"))
                }
            }
        }

        // Download all files with unified progress display and resume support
        let totalFiles = allFiles.count
        for (index, fileInfo) in allFiles.enumerated() {
            let fileIndex = index + 1
            do {
                // Get remote file size first
                let remoteSize = try await getWhisperKitFileSize(from: fileInfo.url)

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

    static func listHuggingFaceFilesWithSize(repo: String) async throws -> [(path: String, size: Int64)] {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repo)/tree/main") else {
            throw NSError(
                domain: "ModelDownloader",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Invalid URL for Hugging Face API."]
            )
        }

        var req = URLRequest(url: url)
        req.setValue("application/json", forHTTPHeaderField: "Accept")

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: req)
        }
        catch {
            throw NSError(
                domain: "ModelDownloader",
                code: 4,
                userInfo: [
                    NSLocalizedDescriptionKey: "Failed to fetch model file list from Hugging Face: \(error.localizedDescription)"
                ]
            )
        }

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            var errorMessage = "Failed to fetch model file list from Hugging Face. Status code: \(statusCode)."
            if statusCode == 404 {
                errorMessage += " Model repository not found or private."
            }
            throw NSError(domain: "ModelDownloader", code: 5, userInfo: [NSLocalizedDescriptionKey: errorMessage])
        }

        do {
            guard let json = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                throw NSError(
                    domain: "ModelDownloader",
                    code: 6,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to parse JSON response from Hugging Face."]
                )
            }

            return json.compactMap { dict -> (String, Int64)? in
                guard let path = dict["path"] as? String else { return nil }

                let size = (dict["size"] as? NSNumber)?.int64Value ?? 0
                return (path, size)
            }
        }
        catch {
            throw NSError(
                domain: "ModelDownloader",
                code: 7,
                userInfo: [NSLocalizedDescriptionKey: "JSON parsing error: \(error.localizedDescription)"]
            )
        }
    }

    static func downloadWithResume(
        repo: String,
        file: String,
        dest: URL,
        totalSize: Int64,
        localSize: Int64
    ) async throws {
        guard let url =
            URL(
                string: "https://huggingface.co/\(repo)/resolve/main/\(file.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file)"
            )
        else {
            throw NSError(
                domain: "ModelDownloader",
                code: 8,
                userInfo: [NSLocalizedDescriptionKey: "Invalid URL for downloading file \(file)."]
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
                userInfo: [NSLocalizedDescriptionKey: "Invalid response from server for file \(file)."]
            )
        }

        let statusCode = httpResponse.statusCode

        // Handle common HTTP errors
        if !(200 ... 299).contains(statusCode) {
            if statusCode == 404 {
                throw NSError(
                    domain: "ModelDownloader",
                    code: 10,
                    userInfo: [NSLocalizedDescriptionKey: "File not found on server (404): \(file)"]
                )
            }
            throw NSError(
                domain: "ModelDownloader",
                code: 11,
                userInfo: [NSLocalizedDescriptionKey: "HTTP error \(statusCode) while downloading \(file)."]
            )
        }

        let shouldResume = isResuming && statusCode == 206 // 206 Partial Content

        if !FileManager.default.fileExists(atPath: dest.path) {
            FileManager.default.createFile(atPath: dest.path, contents: nil)
        }

        let handle: FileHandle
        do {
            handle = try FileHandle(forWritingTo: dest)
        }
        catch {
            throw NSError(
                domain: "ModelDownloader",
                code: 12,
                userInfo: [
                    NSLocalizedDescriptionKey: "Cannot open file for writing at \(dest.path): \(error.localizedDescription)"
                ]
            )
        }
        defer { try? handle.close() }

        if shouldResume {
            try handle.seekToEnd()
        }
        else {
            // If not resuming or if server doesn't support range, truncate and start over.
            // This handles cases where localSize > 0 but server sends 200 OK instead of 206.
            try handle.truncate(atOffset: 0)
        }

        let progressBar = ProgressBar(total: totalSize, initial: shouldResume ? localSize : 0)
        var buffer: [UInt8] = []
        // No need for downloadedBytes, progressBar handles its own downloaded count

        for try await byte in byteStream {
            buffer.append(byte)
            if buffer.count >= 8192 { // Increased buffer size
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

    static func getWhisperKitFileSize(from urlString: String) async throws -> Int64 {
        guard let url = URL(string: urlString) else {
            throw NSError(
                domain: "ModelDownloader",
                code: 14,
                userInfo: [NSLocalizedDescriptionKey: "Invalid URL: \(urlString)"]
            )
        }

        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"

        let (_, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            return 0
        }

        if httpResponse.statusCode == 404 {
            throw NSError(
                domain: "ModelDownloader",
                code: 404,
                userInfo: [NSLocalizedDescriptionKey: "HTTP 404 for file"]
            )
        }

        return httpResponse.expectedContentLength > 0 ? httpResponse.expectedContentLength : 0
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

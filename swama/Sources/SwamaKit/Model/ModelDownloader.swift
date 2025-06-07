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

        let modelDir = ModelManager.modelsBaseDirectory.appendingPathComponent(resolvedModelName)
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
}

import Foundation

// MARK: - HuggingFaceDownloader

/// HuggingFace model downloader, implements IDownloader protocol
public class HuggingFaceDownloader: IDownloader {
    // MARK: - Public

    public required init() {}

    /// list model files with size
    /// - Parameter repo: model repo
    /// - Returns: [(path: String, size: Int64)]
    public func listModelFilesWithSize(repo: String) async throws -> [(path: String, size: Int64)] {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repo)/tree/main") else {
            throw NSError(
                domain: "HuggingFaceDownloader",
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
                domain: "HuggingFaceDownloader",
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
            throw NSError(domain: "HuggingFaceDownloader", code: 5, userInfo: [NSLocalizedDescriptionKey: errorMessage])
        }

        do {
            guard let json = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
                throw NSError(
                    domain: "HuggingFaceDownloader",
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
                domain: "HuggingFaceDownloader",
                code: 7,
                userInfo: [NSLocalizedDescriptionKey: "JSON parsing error: \(error.localizedDescription)"]
            )
        }
    }

    /// download with resume
    /// - Parameters:
    ///   - repo: model repo
    ///   - file: file name
    ///   - dest: destination url
    ///   - totalSize: total size
    ///   - localSize: local size
    public func downloadWithResume(
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

    public func listWhisperKitModelFile(modelDir: URL, modelFolderName: String, openaiModelName: String) async throws -> [(
        url: String,
        localPath: URL,
        fileName: String
    )] {
        // Base URLs for different repositories
        let whisperKitBaseURL = "https://huggingface.co/argmaxinc/whisperkit-coreml/resolve/main"

        // Collect all files to download with their info
        var allFiles: [(url: String, localPath: URL, fileName: String)] = []

        // Config files
        let configFiles = ["config.json", "generation_config.json"]
        for fileName in configFiles {
            let url = "\(whisperKitBaseURL)/\(modelFolderName)/\(fileName)"
            let localPath = modelDir.appendingPathComponent(fileName)
            allFiles.append((url: url, localPath: localPath, fileName: fileName))
        }

        // Tokenizer files from OpenAI's original model (required by WhisperKit)
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
                let url = "\(whisperKitBaseURL)/\(modelFolderName)/\(dirName)/\(fileName)"
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
                    let url = "\(whisperKitBaseURL)/\(modelFolderName)/\(dirName)/\(subdir)/\(fileName)"
                    let localPath = subdirLocal.appendingPathComponent(fileName)
                    allFiles.append((url: url, localPath: localPath, fileName: "\(dirName)/\(subdir)/\(fileName)"))
                }
            }
        }

        return allFiles
    }
    
    public func getWhisperKitFileSize(url: String) async throws -> Int64 {
        guard let url = URL(string: url) else {
            throw NSError(
                domain: "ModelDownloader",
                code: 14,
                userInfo: [NSLocalizedDescriptionKey: "Invalid URL: \(url)"]
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
}

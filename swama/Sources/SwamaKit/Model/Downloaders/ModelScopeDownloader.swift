import Foundation

// MARK: - ModelScopeDownloader

/// ModelScope model downloader, implements IDownloader protocol
public class ModelScopeDownloader: IDownloader {
    // MARK: - Public

    public required init() {}

    /// list model files with size
    /// - Parameter repo: model repo
    /// - Returns: [(path: String, size: Int64)]
    public func listModelFilesWithSize(repo: String) async throws -> [(path: String, size: Int64)] {
        guard let url = URL(string: "https://www.modelscope.cn/api/v1/models/\(repo)/repo/files?Revision=master&Root=")
        else {
            throw NSError(
                domain: "ModelScopeDownloader",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Invalid URL for ModelScope API."]
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
                domain: "ModelScopeDownloader",
                code: 4,
                userInfo: [
                    NSLocalizedDescriptionKey: "Failed to fetch model file list from ModelScop: \(error.localizedDescription)"
                ]
            )
        }
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            var errorMessage = "Failed to fetch model file list from ModelScop. Status code: \(statusCode)."
            if statusCode == 404 {
                errorMessage += " Model repository not found or private."
            }
            throw NSError(domain: "ModelScopeDownloader", code: 5, userInfo: [NSLocalizedDescriptionKey: errorMessage])
        }

        do {
            guard let json =
                try ((JSONSerialization
                        .jsonObject(with: data) as? [String: Any]
                )?["Data"] as? [String: Any])?["Files"] as? [[String: Any]]
            else {
                throw NSError(
                    domain: "ModelScopeDownloader",
                    code: 6,
                    userInfo: [NSLocalizedDescriptionKey: "Failed to parse JSON response from ModelScop."]
                )
            }

            return json.compactMap { dict -> (String, Int64)? in
                guard let path = dict["Path"] as? String else { return nil }

                let size = (dict["Size"] as? NSNumber)?.int64Value ?? 0
                return (path, size)
            }
        }
        catch {
            throw NSError(
                domain: "ModelScopeDownloader",
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
        // Example: https://www.modelscope.cn/api/v1/models/mlx-community/Qwen3-8B-4bit/repo?Revision=master&FilePath=added_tokens.json
        guard let url =
            URL(
                string: "https://www.modelscope.cn/api/v1/models/\(repo)/repo?Revision=master&FilePath=\(file.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file)"
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
}

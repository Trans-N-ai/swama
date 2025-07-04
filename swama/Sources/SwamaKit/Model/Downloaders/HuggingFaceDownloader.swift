import Foundation

// MARK: - HuggingFaceDownloader

/// HuggingFace model downloader, implements IDownloader protocol
public class HuggingFaceDownloader: BaseDownloader {
    // MARK: - Public

    public required init() {}

    override public func getListModelFilesWithSizeApi(repo: String, subDir _: String) -> String {
        "https://huggingface.co/api/models/\(repo)/tree/main"
    }

    override public func modelFilesResponseAdapter(data: Data) throws -> [(path: String, size: Int64)] {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw NSError(
                domain: "BaseDownloader",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey: "Failed to parse JSON response from API."]
            )
        }

        return json.compactMap { dict -> (String, Int64)? in
            guard let path = dict["path"] as? String else { return nil }

            let size = (dict["size"] as? NSNumber)?.int64Value ?? 0
            return (path, size)
        }
    }

    override public func getDownloadUrl(repo: String, file: String) -> String {
        "https://huggingface.co/\(repo)/resolve/main/\(file.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file)"
    }

    override public func listWhisperKitModelFile(
        modelDir: URL,
        modelFolderName: String,
        openaiModelName: String
    ) async throws -> [(
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

    override public func getWhisperKitFileSize(url: String) async throws -> Int64 {
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

        let contentLength = httpResponse.expectedContentLength
        
        // If HEAD request doesn't provide content length (common with compression),
        // try a range request to get the actual file size
        if contentLength <= 0 {
            return try await getFileSizeWithRangeRequest(url: url)
        }

        return contentLength
    }
    
    private func getFileSizeWithRangeRequest(url: URL) async throws -> Int64 {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("bytes=0-0", forHTTPHeaderField: "Range") // Request only the first byte

        let (_, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            return 0
        }
        
        // Check Content-Range header for total file size
        if let contentRange = httpResponse.allHeaderFields["Content-Range"] as? String {
            // Content-Range format: "bytes 0-0/1234" where 1234 is the total size
            let components = contentRange.components(separatedBy: "/")
            if components.count == 2, let totalSize = Int64(components[1]) {
                return totalSize
            }
        }
        
        // If range request doesn't work, check if we got a Content-Length in the response
        let contentLength = httpResponse.expectedContentLength
        if contentLength > 0 {
            return contentLength
        }
        
        // Could not determine file size
        return 0
    }
}

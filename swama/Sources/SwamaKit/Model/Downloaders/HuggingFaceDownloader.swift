import Foundation

// MARK: - HuggingFaceDownloader

/// HuggingFace model downloader, implements IDownloader protocol
public class HuggingFaceDownloader: BaseDownloader {
    // MARK: - Public

    public required init() {}
    
    public override func getListModelFilesWithSizeApi(repo: String, subDir: String) -> String {
        return "https://huggingface.co/api/models/\(repo)/tree/main"
    }
    
    public override func modelFilesResponseAdapter(data: Data) throws -> [(path: String, size: Int64)] {
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
    
    public override func getDownloadUrl(repo: String, file: String) -> String {
        return "https://huggingface.co/\(repo)/resolve/main/\(file.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file)"
    }

    public override func listWhisperKitModelFile(modelDir: URL, modelFolderName: String, openaiModelName: String) async throws -> [(
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
    
    public override func getWhisperKitFileSize(url: String) async throws -> Int64 {
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

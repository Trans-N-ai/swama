import Foundation

// MARK: - ModelScopeDownloader

/// ModelScope model downloader, implements IDownloader protocol
public class ModelScopeDownloader: BaseDownloader {
    // MARK: - Public

    public required init() {}

    override public func getListModelFilesWithSizeApi(repo: String, subDir: String) -> String {
        "https://www.modelscope.cn/api/v1/models/\(repo)/repo/files?Revision=master&Root=\(subDir.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? subDir)"
    }

    override public func modelFilesResponseAdapter(data: Data) throws -> [(path: String, size: Int64)] {
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

    override public func getDownloadUrl(repo: String, file: String) -> String {
        "https://www.modelscope.cn/api/v1/models/\(repo)/repo?Revision=master&FilePath=\(file.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file)"
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
        // https://www.modelscope.cn/models/AI-ModelScope/whisperkit-coreml/resolve/master/openai_whisper-base/MelSpectrogram.mlmodelc/coremldata.bin
        let whisperKitBaseURL = "https://www.modelscope.cn/models/AI-ModelScope/whisperkit-coreml/resolve/master"
        let openaiModelName = openaiModelName.replacingOccurrences(of: "openai/", with: "openai-mirror/")

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
            let url = "https://www.modelscope.cn/models/\(openaiModelName)/resolve/master/\(fileName)"
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
                        ["coremldata.bin"]
                    }
                    else if subdir == "weights" {
                        ["weight.bin", "data.bin"]
                    }
                    else {
                        ["data.bin"]
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
        // https://www.modelscope.cn/models/AI-ModelScope/whisperkit-coreml/resolve/master/openai_whisper-base/config.json
        // https:/www.modelscope.cn/models/openai-mirror/whisper-base/resolve/master/merges.txt
        // regex match the repo name, folder name, file name from the url
        let regex = try! NSRegularExpression(pattern: "^https://www.modelscope.cn/models/(.*)/resolve/master/(.*)$")
        let matches = regex.matches(in: url, range: NSRange(location: 0, length: url.utf16.count))
        guard let match = matches.first else {
            throw NSError(
                domain: "ModelScopeDownloader",
                code: 8,
                userInfo: [NSLocalizedDescriptionKey: "Invalid URL for downloading file \(url)."]
            )
        }

        let repo = (url as NSString).substring(with: match.range(at: 1))
        let fileName = (url as NSString).substring(with: match.range(at: 2))
        // the last part after the last / is the file name, the part before the last / is the folder name
        let fileFolderName = fileName.split(separator: "/").dropLast().joined(separator: "/")

        do {
            let files: [(path: String, size: Int64)] = try await listModelFilesWithSize(
                repo: repo,
                subDir: fileFolderName
            )

            // match the file name is [fileName], return the file size
            let file = files.first { $0.path == fileName }

            return file?.size ?? 0
        }
        catch {
            throw NSError(
                domain: "ModelScopeDownloader",
                code: 7,
                userInfo: [NSLocalizedDescriptionKey: "JSON parsing error: \(error.localizedDescription)"]
            )
        }
    }
}

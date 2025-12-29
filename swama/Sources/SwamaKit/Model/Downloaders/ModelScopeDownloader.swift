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
}

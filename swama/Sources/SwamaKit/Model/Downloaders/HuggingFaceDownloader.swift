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
}

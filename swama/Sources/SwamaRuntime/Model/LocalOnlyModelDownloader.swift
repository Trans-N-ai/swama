import Foundation
import MLXLMCommon

struct LocalOnlyModelDownloader: Downloader {
    func download(
        id: String,
        revision _: String?,
        matching _: [String],
        useLatest _: Bool,
        progressHandler _: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        throw ModelPoolError.modelNotFoundLocally(id)
    }
}

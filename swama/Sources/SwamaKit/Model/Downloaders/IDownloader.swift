import Foundation

/// downloader protocol
public protocol IDownloader {
    /// Required initializer
    init()

    /// list model files with size
    /// - Parameter repo: model repo
    func listModelFilesWithSize(repo: String) async throws -> [(path: String, size: Int64)]

    /// download with resume
    /// - Parameters:
    ///   - repo: model repo
    ///   - file: file name
    ///   - dest: destination url
    ///   - totalSize: total size
    ///   - localSize: local size
    func downloadWithResume(repo: String, file: String, dest: URL, totalSize: Int64, localSize: Int64) async throws
}

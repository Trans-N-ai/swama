import Foundation

/// downloader protocol
public protocol IDownloader {
    /// Required initializer
    init()

    /// list model files with size
    /// - Parameter repo: model repo
    func listModelFilesWithSize(repo: String) async throws -> [(path: String, size: Int64)]

    /// list model files with size
    /// - Parameter repo: model repo
    /// - Parameter subDir: sub directory
    func listModelFilesWithSize(repo: String, subDir: String) async throws -> [(path: String, size: Int64)]

    /// download with resume
    /// - Parameters:
    ///   - repo: model repo
    ///   - file: file name
    ///   - dest: destination url
    ///   - totalSize: total size
    ///   - localSize: local size
    func downloadWithResume(repo: String, file: String, dest: URL, totalSize: Int64, localSize: Int64) async throws

    /// list whisperkit model files
    /// - Parameters:
    ///   - modelFolderName: model folder name
    ///   - openaiModelName: openai model name
    func listWhisperKitModelFile(modelDir: URL, modelFolderName: String, openaiModelName: String) async throws -> [(
        url: String,
        localPath: URL,
        fileName: String
    )]

    /// get whisperkit file size
    /// - Parameters:
    ///   - url: url
    func getWhisperKitFileSize(url: String) async throws -> Int64

    /// download whisperkit file with resume
    /// - Parameters:
    ///   - urlString: url string
    ///   - localFile: local file url
    ///   - totalSize: total size
    ///   - localSize: local size
    func downloadWhisperKitFileWithResume(
        from urlString: String,
        to localFile: URL,
        totalSize: Int64,
        localSize: Int64
    ) async throws
}

//
//  BaseDownloader.swift
//  swama
//
//  Created by BBBOND on 2025/6/24.
//
import Foundation

public class BaseDownloader: IDownloader {
    public required init() {}

    /// get list model files with size api
    /// - Parameters:
    ///   - repo: model repo
    ///   - subDir: sub directory
    /// - Returns: api url
    public func getListModelFilesWithSizeApi(repo _: String, subDir _: String) -> String {
        fatalError("Must be implemented by subclass")
    }

    /// model files response adapter
    /// - Parameters:
    ///   - data: data
    /// - Returns: [(path: String, size: Int64)]
    public func modelFilesResponseAdapter(data _: Data) throws -> [(path: String, size: Int64)] {
        fatalError("Must be implemented by subclass")
    }

    /// get download url
    /// - Parameters:
    ///   - repo: model repo
    ///   - file: file name
    /// - Returns: download url
    public func getDownloadUrl(repo _: String, file _: String) -> String {
        fatalError("Must be implemented by subclass")
    }

    /// list whisperkit model file
    /// - Parameters:
    ///   - modelDir: model directory
    ///   - modelFolderName: model folder name
    ///   - openaiModelName: openai model name
    /// - Returns: [(url: String, localPath: URL, fileName: String)]
    public func listWhisperKitModelFile(
        modelDir _: URL,
        modelFolderName _: String,
        openaiModelName _: String
    ) async throws -> [(
        url: String,
        localPath: URL,
        fileName: String
    )] {
        fatalError("Must be implemented by subclass")
    }

    /// get whisperkit file size
    /// - Parameters:
    ///   - url: url
    /// - Returns: file size
    public func getWhisperKitFileSize(url _: String) async throws -> Int64 {
        fatalError("Must be implemented by subclass")
    }

    /// list model files with size
    /// - Parameters:
    ///   - repo: model repo
    /// - Returns: [(path: String, size: Int64)]
    public func listModelFilesWithSize(repo: String) async throws -> [(path: String, size: Int64)] {
        try await listModelFilesWithSize(repo: repo, subDir: "")
    }

    public func listModelFilesWithSize(repo: String, subDir: String) async throws -> [(path: String, size: Int64)] {
        guard let url = URL(string: getListModelFilesWithSizeApi(repo: repo, subDir: subDir)) else {
            throw NSError(
                domain: "BaseDownloader",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Invalid URL for API."]
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
                domain: "BaseDownloader",
                code: 4,
                userInfo: [
                    NSLocalizedDescriptionKey: "Failed to fetch model file list: \(error.localizedDescription)"
                ]
            )
        }

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            var errorMessage = "Failed to fetch model file list from \(url). Status code: \(statusCode)."
            if statusCode == 404 {
                errorMessage += " Model repository not found or private."
            }
            throw NSError(domain: "BaseDownloader", code: 5, userInfo: [NSLocalizedDescriptionKey: errorMessage])
        }

        do {
            return try modelFilesResponseAdapter(data: data)
        }
        catch {
            throw NSError(
                domain: "BaseDownloader",
                code: 7,
                userInfo: [NSLocalizedDescriptionKey: "JSON parsing error: \(error.localizedDescription)"]
            )
        }
    }

    public func downloadWithResume(
        repo: String,
        file: String,
        dest: URL,
        totalSize: Int64,
        localSize: Int64
    ) async throws {
        guard let url = URL(string: getDownloadUrl(repo: repo, file: file))
        else {
            throw NSError(
                domain: "BaseDownloader",
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
                domain: "BaseDownloader",
                code: 9,
                userInfo: [NSLocalizedDescriptionKey: "Invalid response from server for file \(file)."]
            )
        }

        let statusCode = httpResponse.statusCode

        // Handle common HTTP errors
        if !(200 ... 299).contains(statusCode) {
            if statusCode == 404 {
                throw NSError(
                    domain: "BaseDownloader",
                    code: 10,
                    userInfo: [NSLocalizedDescriptionKey: "File not found on server (404): \(file)"]
                )
            }
            throw NSError(
                domain: "BaseDownloader",
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
                domain: "BaseDownloader",
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

    public func downloadWhisperKitFileWithResume(
        from urlString: String,
        to localFile: URL,
        totalSize: Int64,
        localSize: Int64
    ) async throws {
        guard let url = URL(string: urlString) else {
            throw NSError(
                domain: "BaseDownloader",
                code: 14,
                userInfo: [NSLocalizedDescriptionKey: "Invalid URL: \(urlString)"]
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
                domain: "BaseDownloader",
                code: 9,
                userInfo: [NSLocalizedDescriptionKey: "Invalid response from server."]
            )
        }

        let statusCode = httpResponse.statusCode
        if !(200 ... 299).contains(statusCode) {
            if statusCode == 404 {
                throw NSError(
                    domain: "BaseDownloader",
                    code: 404,
                    userInfo: [NSLocalizedDescriptionKey: "HTTP 404"]
                )
            }
            throw NSError(
                domain: "BaseDownloader",
                code: 15,
                userInfo: [NSLocalizedDescriptionKey: "HTTP error \(statusCode)"]
            )
        }

        let shouldResume = isResuming && statusCode == 206 // 206 Partial Content

        if !FileManager.default.fileExists(atPath: localFile.path) {
            FileManager.default.createFile(atPath: localFile.path, contents: nil)
        }

        let handle = try FileHandle(forWritingTo: localFile)
        defer { try? handle.close() }

        if shouldResume {
            try handle.seekToEnd()
        }
        else {
            try handle.truncate(atOffset: 0)
        }

        let progressBar = ProgressBar(total: totalSize, initial: shouldResume ? localSize : 0)
        var buffer: [UInt8] = []

        for try await byte in byteStream {
            buffer.append(byte)
            if buffer.count >= 8192 {
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

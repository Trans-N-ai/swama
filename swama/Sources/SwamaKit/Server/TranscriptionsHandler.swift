//
//  TranscriptionsHandler.swift
//  SwamaKit
//

import Foundation
import NIOCore
import NIOHTTP1
@preconcurrency import WhisperKit

// MARK: - TranscriptionsHandler

public enum TranscriptionsHandler {
    // MARK: - Request/Response Models

    public struct TranscriptionRequest: Decodable {
        let file: String? // File field will be handled as multipart data
        let model: String
        let language: String? // Optional language code (e.g., "en", "zh", "ja")
        let prompt: String? // Optional prompt text (preserved for future token conversion)
        let response_format: String? // "json", "text", "verbose_json"
        let temperature: Float? // 0.0-1.0, default 0.0

        /// Default values
        var responseFormat: ResponseFormat {
            guard let format = response_format else {
                return .json
            }

            return ResponseFormat(rawValue: format) ?? .json
        }

        var samplingTemperature: Float {
            temperature ?? 0.0
        }
    }

    public enum ResponseFormat: String, CaseIterable {
        case json
        case text
        case verboseJson = "verbose_json"
    }

    public struct TranscriptionResponse: Encodable {
        let text: String
        let language: String? // For verbose_json format
        let duration: Double? // For verbose_json format
        let segments: [Segment]? // For verbose_json format
        let task: String? // For verbose_json format ("transcribe")

        struct Segment: Encodable {
            let id: Int
            let seek: Int
            let start: Double
            let end: Double
            let text: String
            let tokens: [Int]?
            let temperature: Double?
            let avg_logprob: Double?
            let compression_ratio: Double?
            let no_speech_prob: Double?
        }
    }

    // MARK: - Main Handler

    public static func handle(
        requestHead: HTTPRequestHead,
        body: ByteBuffer,
        channel: Channel
    ) async {
        do {
            // Validate content type (should be multipart/form-data)
            guard let contentType = requestHead.headers.first(name: "content-type"),
                  contentType.contains("multipart/form-data")
            else {
                throw TranscriptionError.invalidContentType("Content-Type must be multipart/form-data")
            }

            // Parse multipart form data
            let (audioData, request) = try parseMultipartRequest(body: body, contentType: contentType)

            // Validate file size (100MB limit)
            let maxFileSize = 100 * 1024 * 1024 // 100MB
            guard audioData.count <= maxFileSize else {
                throw TranscriptionError.fileTooLarge("File size exceeds 100MB limit")
            }

            // Validate model
            guard ModelAliasResolver.isWhisperKitModel(request.model) else {
                throw TranscriptionError.invalidModel("Model '\(request.model)' is not a valid WhisperKit model")
            }

            // Perform transcription
            let result = try await performTranscription(
                audioData: audioData,
                request: request
            )

            // Send response
            await sendSuccessResponse(
                channel: channel,
                requestHead: requestHead,
                result: result,
                format: request.responseFormat
            )
        }
        catch {
            await sendErrorResponse(
                channel: channel,
                requestHead: requestHead,
                error: error
            )
        }
    }

    // MARK: - Core Transcription Logic

    private static func performTranscription(
        audioData: Data,
        request: TranscriptionRequest
    ) async throws -> TranscriptionResponse {
        // Create temporary file for audio data
        let tempURL = createTemporaryAudioFile(data: audioData)
        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }

        // Use ModelPool for WhisperKit caching and concurrency control
        let transcribedText: String

        // Always use DecodingOptions for consistent behavior across all formats
        // Capture values to avoid sendable issues
        let language = request.language
        let responseFormat = request.responseFormat
        let modelName = request.model

        if responseFormat == .verboseJson {
            // For verbose format, get detailed results with timestamps
            let transcriptionOutput = try await ModelPool.shared.runWhisperKit(modelName: modelName) { runner in
                try await runner.transcribe(
                    audioFile: tempURL,
                    language: language,
                    responseFormat: .verboseJson
                )
            }

            // Extract detailed results from output
            let detailedResults: [TranscriptionResult]
            switch transcriptionOutput {
            case let .detailed(results):
                detailedResults = results
            case .simple:
                throw ServerError.setupFailed("Expected detailed results but got simple output")
            }

            // Extract text from results
            transcribedText = detailedResults.compactMap(\.text).joined(separator: " ")

            // Convert results to segments and extract metadata
            let segments = extractSegments(from: detailedResults)
            let duration = calculateDuration(from: detailedResults)
            let detectedLanguage = extractLanguage(from: detailedResults) ?? request.language

            return TranscriptionResponse(
                text: transcribedText,
                language: detectedLanguage,
                duration: duration,
                segments: segments,
                task: "transcribe"
            )
        }
        else {
            // Simple transcription
            let transcriptionOutput = try await ModelPool.shared.runWhisperKit(modelName: modelName) { runner in
                try await runner.transcribe(
                    audioFile: tempURL,
                    language: language,
                    responseFormat: .simple
                )
            }

            // Extract text from output
            switch transcriptionOutput {
            case let .simple(text):
                transcribedText = text
            case let .detailed(results):
                transcribedText = results.compactMap(\.text).joined(separator: " ")
            }

            return TranscriptionResponse(
                text: transcribedText,
                language: request.language,
                duration: nil,
                segments: nil,
                task: nil
            )
        }
    }

    private static func createTemporaryAudioFile(data: Data) -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent(UUID().uuidString + ".wav")
        try! data.write(to: tempFile)
        return tempFile
    }

    private static func parseMultipartRequest(
        body: ByteBuffer,
        contentType: String
    ) throws -> (Data, TranscriptionRequest) {
        // Extract boundary from content-type header
        guard let boundary = extractBoundary(from: contentType) else {
            throw TranscriptionError.invalidContentType("Missing boundary in multipart/form-data")
        }

        // Parse multipart data
        let bodyData = Data(buffer: body, byteBufferView: body.readerIndex ..< body.writerIndex)
        let parts = try parseMultipartData(data: bodyData, boundary: boundary)

        // Extract file data and form fields
        var audioData: Data?
        var model: String?
        var language: String?
        var prompt: String?
        var responseFormat: String?
        var temperature: Float?

        for part in parts {
            switch part.name {
            case "file":
                audioData = part.data

            case "model":
                model = String(data: part.data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)

            case "language":
                language = String(data: part.data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)

            case "prompt":
                prompt = String(data: part.data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)

            case "response_format":
                responseFormat = String(data: part.data, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)

            case "temperature":
                if let tempStr = String(data: part.data, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                {
                    temperature = Float(tempStr)
                }

            default:
                continue
            }
        }

        // Validate required fields
        guard let audioData else {
            throw TranscriptionError.missingFile("Audio file is required")
        }
        guard let model, !model.isEmpty else {
            throw TranscriptionError.missingModel("Model field is required")
        }

        let request = TranscriptionRequest(
            file: nil, // Not used in parsed request
            model: model,
            language: language,
            prompt: prompt,
            response_format: responseFormat,
            temperature: temperature
        )

        return (audioData, request)
    }

    // MARK: - Response Helpers

    private static func sendSuccessResponse(
        channel: Channel,
        requestHead: HTTPRequestHead,
        result: TranscriptionResponse,
        format: ResponseFormat
    ) async {
        do {
            let responseData: Data
            let contentType: String

            switch format {
            case .json:
                let jsonResponse = ["text": result.text]
                responseData = try JSONSerialization.data(withJSONObject: jsonResponse)
                contentType = "application/json"

            case .text:
                responseData = result.text.data(using: .utf8) ?? Data()
                contentType = "text/plain"

            case .verboseJson:
                responseData = try JSONEncoder().encode(result)
                contentType = "application/json"
            }

            await sendResponse(
                channel: channel,
                requestHead: requestHead,
                data: responseData,
                contentType: contentType,
                status: .ok
            )
        }
        catch {
            await sendErrorResponse(channel: channel, requestHead: requestHead, error: error)
        }
    }

    private static func sendErrorResponse(
        channel: Channel,
        requestHead: HTTPRequestHead,
        error: Error
    ) async {
        let errorMessage: String
        let statusCode: HTTPResponseStatus

        if let transcriptionError = error as? TranscriptionError {
            errorMessage = transcriptionError.localizedDescription
            statusCode = transcriptionError.httpStatus
        }
        else {
            errorMessage = "Internal server error: \(error.localizedDescription)"
            statusCode = .internalServerError
        }

        let errorResponse = [
            "error": [
                "message": errorMessage,
                "type": "transcription_error"
            ]
        ]

        do {
            let errorData = try JSONSerialization.data(withJSONObject: errorResponse)
            await sendResponse(
                channel: channel,
                requestHead: requestHead,
                data: errorData,
                contentType: "application/json",
                status: statusCode
            )
        }
        catch {
            // Fallback to simple error message
            let fallbackData = "Internal Server Error".data(using: .utf8) ?? Data()
            await sendResponse(
                channel: channel,
                requestHead: requestHead,
                data: fallbackData,
                contentType: "text/plain",
                status: .internalServerError
            )
        }
    }

    private static func sendResponse(
        channel: Channel,
        requestHead: HTTPRequestHead,
        data: Data,
        contentType: String,
        status: HTTPResponseStatus
    ) async {
        var buffer = channel.allocator.buffer(capacity: data.count)
        buffer.writeBytes(data)

        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: contentType)
        headers.add(name: "Content-Length", value: "\(buffer.readableBytes)")
        headers.add(name: "Connection", value: "close")

        channel.write(
            HTTPServerResponsePart.head(HTTPResponseHead(
                version: requestHead.version,
                status: status,
                headers: headers
            )),
            promise: nil
        )
        channel.write(HTTPServerResponsePart.body(.byteBuffer(buffer)), promise: nil)
        channel.writeAndFlush(HTTPServerResponsePart.end(nil), promise: nil)
    }
}

// MARK: - Multipart Parsing

extension TranscriptionsHandler {
    struct MultipartPart {
        let name: String
        let filename: String?
        let contentType: String?
        let data: Data
    }

    private static func extractBoundary(from contentType: String) -> String? {
        let components = contentType.components(separatedBy: ";")
        for component in components {
            let trimmed = component.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("boundary=") {
                return String(trimmed.dropFirst(9))
            }
        }
        return nil
    }

    private static func parseMultipartData(data: Data, boundary: String) throws -> [MultipartPart] {
        let boundaryData = ("--" + boundary).data(using: .utf8)!
        let endBoundaryData = ("--" + boundary + "--").data(using: .utf8)!

        var parts: [MultipartPart] = []
        var currentIndex = 0

        // Find first boundary
        guard let firstBoundaryRange = data.range(of: boundaryData, in: currentIndex ..< data.count) else {
            throw TranscriptionError.invalidMultipart("No boundary found")
        }

        currentIndex = firstBoundaryRange.upperBound

        while currentIndex < data.count {
            // Skip CRLF after boundary
            if currentIndex + 1 < data.count, data[currentIndex] == 13, data[currentIndex + 1] == 10 {
                currentIndex += 2
            }

            // Find next boundary or end boundary
            let searchRange = currentIndex ..< data.count
            let nextBoundaryRange = data.range(of: boundaryData, in: searchRange)
            let endBoundaryRange = data.range(of: endBoundaryData, in: searchRange)

            let partEndIndex: Int
            if let nextRange = nextBoundaryRange, let endRange = endBoundaryRange {
                partEndIndex = min(nextRange.lowerBound, endRange.lowerBound)
            }
            else if let nextRange = nextBoundaryRange {
                partEndIndex = nextRange.lowerBound
            }
            else if let endRange = endBoundaryRange {
                partEndIndex = endRange.lowerBound
            }
            else {
                break
            }

            // Extract part data
            let partData = data.subdata(in: currentIndex ..< partEndIndex)
            if let part = try? parseMultipartPart(data: partData) {
                parts.append(part)
            }

            // Move to next part
            if let nextRange = nextBoundaryRange, nextRange.lowerBound == partEndIndex {
                currentIndex = nextRange.upperBound
            }
            else {
                break
            }
        }

        return parts
    }

    private static func parseMultipartPart(data: Data) throws -> MultipartPart {
        // Find header/body separator (double CRLF)
        let separator = "\r\n\r\n".data(using: .utf8)!
        guard let separatorRange = data.range(of: separator) else {
            throw TranscriptionError.invalidMultipart("No header/body separator found")
        }

        let headerData = data.subdata(in: 0 ..< separatorRange.lowerBound)
        let bodyData = data.subdata(in: separatorRange.upperBound ..< data.count)

        // Parse headers
        guard let headerString = String(data: headerData, encoding: .utf8) else {
            throw TranscriptionError.invalidMultipart("Invalid header encoding")
        }

        var name: String?
        var filename: String?
        var contentType: String?

        let lines = headerString.components(separatedBy: "\r\n")
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.lowercased().hasPrefix("content-disposition:") {
                // Parse name and filename from Content-Disposition header
                if let nameMatch = extractQuotedValue(from: trimmed, parameter: "name") {
                    name = nameMatch
                }
                if let filenameMatch = extractQuotedValue(from: trimmed, parameter: "filename") {
                    filename = filenameMatch
                }
            }
            else if trimmed.lowercased().hasPrefix("content-type:") {
                contentType = String(trimmed.dropFirst(13).trimmingCharacters(in: .whitespaces))
            }
        }

        guard let partName = name else {
            throw TranscriptionError.invalidMultipart("Missing name parameter")
        }

        return MultipartPart(
            name: partName,
            filename: filename,
            contentType: contentType,
            data: bodyData
        )
    }

    private static func extractQuotedValue(from string: String, parameter: String) -> String? {
        let pattern = "\(parameter)=\"([^\"]*)\""
        let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive)
        let range = NSRange(string.startIndex ..< string.endIndex, in: string)

        if let match = regex?.firstMatch(in: string, options: [], range: range),
           let valueRange = Range(match.range(at: 1), in: string)
        {
            return String(string[valueRange])
        }

        return nil
    }
}

// MARK: - Error Types

public enum TranscriptionError: Error, LocalizedError {
    case invalidContentType(String)
    case missingFile(String)
    case missingModel(String)
    case fileTooLarge(String)
    case invalidModel(String)
    case invalidMultipart(String)
    case transcriptionFailed(String)

    public var errorDescription: String? {
        switch self {
        case let .fileTooLarge(message),
             let .invalidContentType(message),
             let .invalidModel(message),
             let .invalidMultipart(message),
             let .missingFile(message),
             let .missingModel(message),
             let .transcriptionFailed(message):
            message
        }
    }

    var httpStatus: HTTPResponseStatus {
        switch self {
        case .invalidContentType,
             .invalidMultipart,
             .missingFile,
             .missingModel:
            .badRequest
        case .fileTooLarge:
            .payloadTooLarge
        case .invalidModel:
            .badRequest
        case .transcriptionFailed:
            .internalServerError
        }
    }
}

// MARK: - Data Extension

extension Data {
    init(buffer: ByteBuffer, byteBufferView range: Range<Int>) {
        let bytes = buffer.getBytes(at: range.lowerBound, length: range.count) ?? []
        self.init(bytes)
    }
}

// MARK: - Helper Methods for Verbose JSON Support

extension TranscriptionsHandler {
    private static func extractSegments(from results: [TranscriptionResult]) -> [TranscriptionResponse.Segment] {
        // Flatten all segments from all results
        let allSegments = results.flatMap(\.segments)

        return allSegments.map { segment in
            TranscriptionResponse.Segment(
                id: segment.id,
                seek: segment.seek,
                start: Double(segment.start),
                end: Double(segment.end),
                text: segment.text, // Text is already cleaned by WhisperKitRunner
                tokens: segment.tokens,
                temperature: Double(segment.temperature),
                avg_logprob: Double(segment.avgLogprob),
                compression_ratio: Double(segment.compressionRatio),
                no_speech_prob: Double(segment.noSpeechProb)
            )
        }
    }

    private static func calculateDuration(from results: [TranscriptionResult]) -> Double? {
        guard !results.isEmpty else {
            return nil
        }

        // Find the maximum end time across all segments
        let allSegments = results.flatMap(\.segments)
        guard !allSegments.isEmpty else {
            return nil
        }

        let maxEndTime = allSegments.map(\.end).max() ?? 0.0
        return Double(maxEndTime)
    }

    private static func extractLanguage(from results: [TranscriptionResult]) -> String? {
        // Return the language from the first result (they should all be the same)
        results.first?.language
    }
}

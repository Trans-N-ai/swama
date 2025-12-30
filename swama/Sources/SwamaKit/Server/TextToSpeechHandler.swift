import Foundation
@preconcurrency import MLXAudio
import NIOCore
import NIOHTTP1

// MARK: - TextToSpeechHandler

public enum TextToSpeechHandler {
    struct SpeechRequest: Decodable {
        let model: String
        let input: String
        let voice: String?
        let response_format: String?
        let speed: Float?
    }

    enum ResponseFormat: String {
        case wav
        case pcm

        var contentType: String {
            switch self {
            case .wav:
                "audio/wav"
            case .pcm:
                "audio/pcm"
            }
        }
    }

    public static func handle(
        requestHead: HTTPRequestHead,
        body: ByteBuffer,
        channel: Channel
    ) async {
        do {
            var mutableBody = body
            guard let bodyBytes = mutableBody.readBytes(length: body.readableBytes) else {
                throw TextToSpeechError.invalidRequest("Invalid request body")
            }

            let request = try JSONDecoder().decode(SpeechRequest.self, from: Data(bodyBytes))

            let trimmedInput = request.input.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedInput.isEmpty else {
                throw TextToSpeechError.invalidRequest("Input text is required")
            }

            if let speed = request.speed, speed < 0.25 || speed > 4.0 {
                throw TextToSpeechError.invalidRequest("Speed must be between 0.25 and 4.0")
            }

            guard let modelResolution = TTSModelResolver.resolve(request.model) else {
                throw TextToSpeechError.invalidModel("Model '\(request.model)' is not supported for TTS")
            }

            let responseFormat = try resolveResponseFormat(request.response_format)

            let result = try await ModelPool.shared.runTTS(
                modelKey: modelResolution.cacheKey,
                kind: modelResolution.kind
            ) { runner in
                try await runner.generate(
                    text: trimmedInput,
                    voice: request.voice,
                    speed: request.speed
                )
            }

            let responseData = try encodeAudio(result: result, format: responseFormat)

            await sendResponse(
                channel: channel,
                requestHead: requestHead,
                data: responseData,
                contentType: responseFormat.contentType,
                status: .ok
            )
        }
        catch {
            await sendErrorResponse(channel: channel, requestHead: requestHead, error: error)
        }
    }

    private static func resolveResponseFormat(_ value: String?) throws -> ResponseFormat {
        let normalized = (value ?? "wav").lowercased()
        guard let format = ResponseFormat(rawValue: normalized) else {
            throw TextToSpeechError.invalidRequest("Unsupported response_format '\(value ?? "")'")
        }

        return format
    }

    private static func encodeAudio(result: AudioResult, format: ResponseFormat) throws -> Data {
        switch format {
        case .wav:
            try encodeWav(result: result)
        case .pcm:
            try encodePCM(result: result)
        }
    }

    private static func encodeWav(result: AudioResult) throws -> Data {
        switch result {
        case let .samples(samples, sampleRate, _):
            let filename = "swama_tts_\(UUID().uuidString)"
            let url = try AudioFileWriter.save(
                samples: samples,
                sampleRate: sampleRate,
                to: FileManager.default.temporaryDirectory,
                filename: filename,
                format: .wav
            )
            defer { try? FileManager.default.removeItem(at: url) }
            return try Data(contentsOf: url)

        case let .file(url, _):
            return try Data(contentsOf: url)
        }
    }

    private static func encodePCM(result: AudioResult) throws -> Data {
        guard case let .samples(samples, _, _) = result else {
            throw TextToSpeechError.invalidRequest("PCM output requires in-memory samples")
        }

        var data = Data(capacity: samples.count * 2)
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let intSample = Int16(clamped * Float(Int16.max))
            var littleEndian = intSample.littleEndian
            withUnsafeBytes(of: &littleEndian) { bytes in
                data.append(contentsOf: bytes)
            }
        }
        return data
    }

    private static func sendErrorResponse(
        channel: Channel,
        requestHead: HTTPRequestHead,
        error: Error
    ) async {
        let errorMessage: String
        let statusCode: HTTPResponseStatus

        if let ttsError = error as? TextToSpeechError {
            errorMessage = ttsError.localizedDescription
            statusCode = ttsError.httpStatus
        }
        else if let ttsError = error as? TTSError {
            errorMessage = ttsError.localizedDescription
            statusCode = ttsError.httpStatus
        }
        else {
            errorMessage = "Internal server error: \(error.localizedDescription)"
            statusCode = .internalServerError
        }

        let errorResponse = [
            "error": [
                "message": errorMessage,
                "type": "tts_error",
            ],
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
        HTTPHandler.applyCORSHeaders(&headers)

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

// MARK: - TextToSpeechError

private enum TextToSpeechError: Error, LocalizedError {
    case invalidRequest(String)
    case invalidModel(String)

    var errorDescription: String? {
        switch self {
        case let .invalidModel(message),
             let .invalidRequest(message):
            message
        }
    }

    var httpStatus: HTTPResponseStatus {
        switch self {
        case .invalidModel,
             .invalidRequest:
            .badRequest
        }
    }
}

private extension TTSError {
    var httpStatus: HTTPResponseStatus {
        switch self {
        case .invalidArgument,
             .invalidReferenceAudio,
             .invalidVoice,
             .modelNotLoaded,
             .voiceNotFound:
            .badRequest
        case .insufficientMemory:
            .insufficientStorage
        case .cancelled:
            .requestTimeout
        case .audioPlaybackFailed,
             .fileIOError,
             .generationFailed,
             .modelLoadFailed:
            .internalServerError
        }
    }
}

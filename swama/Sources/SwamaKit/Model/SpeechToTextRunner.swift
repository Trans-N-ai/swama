import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTT

// MARK: - TranscriptionResult

public struct TranscriptionResult: Sendable {
    public struct Segment: Sendable {
        public var id: Int
        public var seek: Int
        public var start: Float
        public var end: Float
        public var text: String
        public var tokens: [Int]?
        public var temperature: Float
        public var avgLogprob: Float
        public var compressionRatio: Float
        public var noSpeechProb: Float

        public init(
            id: Int = 0,
            seek: Int = 0,
            start: Float = 0,
            end: Float = 0,
            text: String = "",
            tokens: [Int]? = nil,
            temperature: Float = 0,
            avgLogprob: Float = 0,
            compressionRatio: Float = 0,
            noSpeechProb: Float = 0
        ) {
            self.id = id
            self.seek = seek
            self.start = start
            self.end = end
            self.text = text
            self.tokens = tokens
            self.temperature = temperature
            self.avgLogprob = avgLogprob
            self.compressionRatio = compressionRatio
            self.noSpeechProb = noSpeechProb
        }
    }

    public var text: String
    public var language: String?
    public var segments: [Segment]

    public init(text: String = "", language: String? = nil, segments: [Segment] = []) {
        self.text = text
        self.language = language
        self.segments = segments
    }
}

// MARK: - TranscriptionResponseFormat

public enum TranscriptionResponseFormat {
    case simple
    case verboseJson
}

// MARK: - TranscriptionOutput

public enum TranscriptionOutput: Sendable {
    case simple(String)
    case detailed([TranscriptionResult])
}

// MARK: - SpeechToTextRunner

@MainActor
public class SpeechToTextRunner: @unchecked Sendable {
    private var stt: (any STTGenerationModel)?
    private var isRunning = false
    private var selectedModelName: String?

    public init() {}

    public func loadModel(from modelName: String) async throws {
        let stt = try await createSTT(for: modelName)
        self.stt = stt
        self.selectedModelName = modelName
    }

    public func transcribe(
        audioFile url: URL,
        language: String? = nil,
        temperature: Float = 0.0,
        responseFormat: TranscriptionResponseFormat = .simple
    ) async throws -> TranscriptionOutput {
        while isRunning {
            try await Task.sleep(nanoseconds: 10_000_000)
        }

        isRunning = true
        defer { isRunning = false }

        guard let stt else {
            throw AudioError.modelLoadFailed("MLXAudioSTT model not initialized")
        }
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioError.transcriptionFailed("Audio file not found: \(url.path)")
        }

        do {
            let audio = try loadPreparedAudio(from: url)
            var params = stt.defaultGenerationParameters
            params = STTGenerateParameters(
                maxTokens: params.maxTokens,
                temperature: temperature,
                topP: params.topP,
                topK: params.topK,
                verbose: responseFormat == .verboseJson,
                language: normalizeLanguage(language),
                chunkDuration: params.chunkDuration,
                minChunkDuration: params.minChunkDuration,
                repetitionPenalty: params.repetitionPenalty,
                repetitionContextSize: params.repetitionContextSize
            )

            let output = stt.generate(audio: audio, generationParameters: params)
            switch responseFormat {
            case .simple:
                return .simple(output.text)
            case .verboseJson:
                return .detailed([makeTranscriptionResult(from: output, fallbackLanguage: language)])
            }
        }
        catch {
            throw AudioError.transcriptionFailed("Transcription failed: \(error.localizedDescription)")
        }
    }

    private func loadPreparedAudio(from url: URL) throws -> MLXArray {
        let (inputSampleRate, inputAudio) = try loadAudioArray(from: url)
        let mono = inputAudio.ndim > 1 ? inputAudio.mean(axis: -1) : inputAudio
        guard inputSampleRate != 16000 else {
            return mono
        }

        return try MLXAudioCore.resampleAudio(mono, from: inputSampleRate, to: 16000)
    }
}

public extension SpeechToTextRunner {
    func loadModel(_ modelIdentifier: String) async throws {
        guard ModelAliasResolver.isAudioModel(modelIdentifier) else {
            throw AudioError.invalidInput("Model '\(modelIdentifier)' is not a supported audio model.")
        }

        try await loadModel(from: modelIdentifier)
    }
}

// MARK: - Audio Errors

public enum AudioError: Error, LocalizedError {
    case modelLoadFailed(String)
    case audioLoadFailed(String)
    case transcriptionFailed(String)
    case invalidInput(String)
    case modelNotFound(String)

    public var errorDescription: String? {
        switch self {
        case let .modelLoadFailed(message):
            "Failed to load audio model: \(message)"
        case let .audioLoadFailed(message):
            "Failed to load audio: \(message)"
        case let .transcriptionFailed(message):
            "Transcription failed: \(message)"
        case let .invalidInput(message):
            "Invalid input: \(message)"
        case let .modelNotFound(message):
            message
        }
    }
}

private extension SpeechToTextRunner {
    func createSTT(for modelName: String) async throws -> any STTGenerationModel {
        let repo = resolveSTTRepo(modelName)
        let normalized = repo.lowercased()
        let cache = HubCache(cacheDirectory: ModelPaths.activeModelsDirectory)

        if normalized.contains("glmasr") || normalized.contains("glm-asr") {
            return try await GLMASRModel.fromPretrained(repo, cache: cache)
        }
        if normalized.contains("voxtral") {
            // Voxtral/Cohere fromPretrained in mlx-audio-swift do not accept a cache
            // override yet, so they fall back to the library's default HF cache.
            return try await VoxtralRealtimeModel.fromPretrained(repo)
        }
        if normalized.contains("cohere") {
            return try await CohereTranscribeModel.fromPretrained(repo)
        }
        if normalized.contains("parakeet") {
            return try await ParakeetModel.fromPretrained(repo, cache: cache)
        }
        if normalized.contains("firered") || normalized.contains("fire-red") {
            return try await FireRedASR2Model.fromPretrained(repo, cache: cache)
        }
        if normalized.contains("sensevoice") {
            return try await SenseVoiceModel.fromPretrained(repo, cache: cache)
        }
        if normalized.contains("qwen3-asr") || normalized.contains("qwen3_asr") {
            return try await Qwen3ASRModel.fromPretrained(repo, cache: cache)
        }

        throw AudioError.modelNotFound("Unsupported STT model: \(modelName)")
    }

    func resolveSTTRepo(_ modelName: String) -> String {
        let resolved = ModelAliasResolver.resolve(name: modelName)
        let normalized = resolved.lowercased()

        if normalized.hasPrefix("whisper-") ||
            normalized.hasPrefix("mlx-community/whisper") ||
            normalized.hasPrefix("funasr") ||
            normalized.hasPrefix("fun-asr") ||
            normalized.hasPrefix("mlx-community/fun-asr")
        {
            return "mlx-community/Qwen3-ASR-0.6B-4bit"
        }

        return resolved
    }

    func normalizeLanguage(_ language: String?) -> String? {
        guard let trimmed = language?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty
        else {
            return nil
        }

        switch trimmed.lowercased() {
        case "en",
             "english":
            return "English"
        case "chinese",
             "zh",
             "zh-cn",
             "zh-hans":
            return "Chinese"
        case "ja",
             "japanese":
            return "Japanese"
        default:
            return trimmed
        }
    }

    func makeTranscriptionResult(from output: STTOutput, fallbackLanguage: String?) -> TranscriptionResult {
        let segments = extractSegments(from: output)
        return TranscriptionResult(
            text: output.text,
            language: output.language ?? fallbackLanguage,
            segments: segments.isEmpty
                ? [TranscriptionResult.Segment(text: output.text)]
                : segments
        )
    }

    func extractSegments(from output: STTOutput) -> [TranscriptionResult.Segment] {
        guard let rawSegments = output.segments else {
            return []
        }

        return rawSegments.enumerated().compactMap { index, raw in
            guard let text = raw["text"] as? String else {
                return nil
            }

            return TranscriptionResult.Segment(
                id: intValue(raw["id"]) ?? index,
                seek: intValue(raw["seek"]) ?? 0,
                start: floatValue(raw["start"]) ?? 0,
                end: floatValue(raw["end"]) ?? 0,
                text: text,
                tokens: raw["tokens"] as? [Int],
                temperature: floatValue(raw["temperature"]) ?? 0,
                avgLogprob: floatValue(raw["avg_logprob"]) ?? 0,
                compressionRatio: floatValue(raw["compression_ratio"]) ?? 0,
                noSpeechProb: floatValue(raw["no_speech_prob"]) ?? 0
            )
        }
    }

    func intValue(_ value: Any?) -> Int? {
        switch value {
        case let value as Int:
            value
        case let value as Double:
            Int(value)
        case let value as Float:
            Int(value)
        case let value as NSNumber:
            value.intValue
        default:
            nil
        }
    }

    func floatValue(_ value: Any?) -> Float? {
        switch value {
        case let value as Float:
            value
        case let value as Double:
            Float(value)
        case let value as Int:
            Float(value)
        case let value as NSNumber:
            value.floatValue
        default:
            nil
        }
    }
}

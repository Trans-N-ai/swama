import Foundation
import HuggingFace
@preconcurrency import MLX
@preconcurrency import MLXAudioTTS
import MLXLMCommon

// MARK: - AudioResult

public enum AudioResult: Sendable {
    case samples([Float], Int, TimeInterval?)
    case file(URL, TimeInterval?)
}

// MARK: - TTSError

public enum TTSError: Error, LocalizedError {
    case invalidArgument(String)
    case invalidReferenceAudio(String)
    case invalidVoice(String)
    case modelNotLoaded
    case unsupportedStreamingGranularity(String)
    case voiceNotFound(String)
    case insufficientMemory(String)
    case cancelled
    case audioPlaybackFailed(String)
    case fileIOError(String)
    case generationFailed(String)
    case modelLoadFailed(String)

    public var errorDescription: String? {
        switch self {
        case let .audioPlaybackFailed(message),
             let .fileIOError(message),
             let .generationFailed(message),
             let .insufficientMemory(message),
             let .invalidArgument(message),
             let .invalidReferenceAudio(message),
             let .invalidVoice(message),
             let .modelLoadFailed(message),
             let .unsupportedStreamingGranularity(message),
             let .voiceNotFound(message):
            message
        case .modelNotLoaded:
            "TTS model is not loaded"
        case .cancelled:
            "TTS generation was cancelled"
        }
    }
}

// MARK: - TTSModelKind

public enum TTSModelKind: String, CaseIterable, Sendable {
    case orpheus
    case marvis
    case chatterbox
    case chatterboxTurbo = "chatterbox-turbo"
    case qwen3TTS = "qwen3-tts"
    case vyvo
    case fishSpeech = "fish-speech"
    case soprano
    case pocketTTS = "pocket-tts"
    case mossTTS = "moss-tts"
    case echoTTS = "echo-tts"

    // Legacy aliases kept so existing API model names continue resolving.
    case cosyVoice2 = "cosyvoice2"
    case cosyVoice3 = "cosyvoice3"
    case outetts
}

// MARK: - TTSModelResolution

public struct TTSModelResolution: Sendable {
    public let kind: TTSModelKind
    public let cacheKey: String
}

// MARK: - TTSModelResolver

public enum TTSModelResolver {
    private static let qwen3TTSRepo = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"

    public static let availableModels: [String] = [
        "orpheus",
        "marvis",
        "chatterbox",
        "chatterbox-turbo",
        "qwen3-tts",
        "vyvo",
        "fish-speech",
        "soprano",
        "pocket-tts",
        "moss-tts",
        "echo-tts",
        "cosyvoice2",
        "cosyvoice3",
        "outetts",
    ]

    public static func resolve(_ modelName: String) -> TTSModelResolution? {
        let normalized = modelName.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        switch normalized {
        case "mlx-community/orpheus-3b-0.1-ft-4bit",
             "mlx-community/orpheus-3b-0.1-ft-bf16",
             "orpheus":
            return resolution(.orpheus)
        case "marvis",
             "marvis-ai/marvis-tts-100m-v0.2-mlx-6bit",
             "marvis-ai/marvis-tts-250m-v0.2-mlx-8bit":
            return resolution(.marvis)
        case "chatterbox",
             "mlx-community/chatterbox-tts-q4":
            return resolution(.chatterbox)
        case "chatterbox_turbo",
             "chatterbox-turbo",
             "mlx-community/chatterbox-turbo-4bit",
             "mlx-community/chatterbox-turbo-tts-q4":
            return resolution(.chatterboxTurbo)
        case "qwen3-tts",
             "qwen3tts",
             qwen3TTSRepo.lowercased():
            return resolution(.qwen3TTS)
        case "mlx-community/vyvotts-en-beta-4bit",
             "vyvo",
             "vyvotts":
            return resolution(.vyvo)
        case "fish",
             "fish-audio",
             "fish-speech",
             "mlx-community/fish-audio-s2-pro-8bit":
            return resolution(.fishSpeech)
        case "mlx-community/soprano-80m-bf16",
             "soprano":
            return resolution(.soprano)
        case "mlx-community/pocket-tts",
             "pocket",
             "pocket-tts":
            return resolution(.pocketTTS)
        case "moss",
             "moss-tts",
             "openmoss-team/moss-tts":
            return resolution(.mossTTS)
        case "echo",
             "echo-tts",
             "mlx-community/echo-tts-base":
            return resolution(.echoTTS)
        case "cosy-voice2",
             "cosyvoice2",
             "mlx-community/cosyvoice2-0.5b-4bit":
            return resolution(.cosyVoice2)
        case "cosy-voice3",
             "cosyvoice3",
             "mlx-community/fun-cosyvoice3-0.5b-2512-4bit":
            return resolution(.cosyVoice3)
        case "mlx-community/llama-outetts-1.0-1b-4bit",
             "outetts":
            return resolution(.outetts)
        default:
            return resolveRepository(normalized)
        }
    }

    public static func voiceIDs(for kind: TTSModelKind) -> [String] {
        switch kind {
        case .orpheus:
            ["dan", "jess", "leo", "mia", "tara", "zac", "zoe"]
        case .marvis:
            ["conversational_a", "conversational_b"]
        case .qwen3TTS,
             .vyvo:
            ["en-us-1"]
        case .chatterbox,
             .chatterboxTurbo,
             .cosyVoice2,
             .cosyVoice3,
             .echoTTS,
             .fishSpeech,
             .mossTTS,
             .outetts,
             .pocketTTS,
             .soprano:
            []
        }
    }

    public static func repoIDs(for kind: TTSModelKind) -> [String] {
        [repository(for: kind)]
    }

    public static func repository(for kind: TTSModelKind) -> String {
        switch kind {
        case .orpheus:
            "mlx-community/orpheus-3b-0.1-ft-bf16"
        case .marvis:
            "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit"
        case .chatterbox,
             .chatterboxTurbo:
            "mlx-community/chatterbox-turbo-4bit"
        case .cosyVoice2,
             .cosyVoice3,
             .outetts,
             .qwen3TTS:
            qwen3TTSRepo
        case .vyvo:
            "mlx-community/VyvoTTS-EN-Beta-4bit"
        case .fishSpeech:
            "mlx-community/fish-audio-s2-pro-8bit"
        case .soprano:
            "mlx-community/Soprano-80M-bf16"
        case .pocketTTS:
            "mlx-community/pocket-tts"
        case .mossTTS:
            "OpenMOSS-Team/MOSS-TTS"
        case .echoTTS:
            "mlx-community/echo-tts-base"
        }
    }

    private static func resolution(_ kind: TTSModelKind) -> TTSModelResolution {
        TTSModelResolution(kind: kind, cacheKey: kind.rawValue)
    }

    private static func resolveRepository(_ normalized: String) -> TTSModelResolution? {
        if normalized.contains("qwen3-tts") {
            return resolution(.qwen3TTS)
        }
        if normalized.contains("vyvotts") {
            return resolution(.vyvo)
        }
        if normalized.contains("fish") {
            return resolution(.fishSpeech)
        }
        if normalized.contains("soprano") {
            return resolution(.soprano)
        }
        if normalized.contains("orpheus") {
            return resolution(.orpheus)
        }
        if normalized.contains("marvis") {
            return resolution(.marvis)
        }
        if normalized.contains("chatterbox") {
            return resolution(.chatterboxTurbo)
        }
        if normalized.contains("pocket") {
            return resolution(.pocketTTS)
        }
        if normalized.contains("moss") {
            return resolution(.mossTTS)
        }
        if normalized.contains("echo") {
            return resolution(.echoTTS)
        }
        return nil
    }
}

// MARK: - TTSRunner

public final class TTSRunner: @unchecked Sendable {
    private let kind: TTSModelKind
    private var model: (any SpeechGenerationModel)?

    public init(kind: TTSModelKind) {
        self.kind = kind
    }

    public func loadModel() async throws {
        if model != nil {
            return
        }

        do {
            // Use swama's models directory as the cache root so downloaded weights and
            // loaded weights resolve to the same place (see ModelPaths.audioModelDirectory).
            let cache = HubCache(cacheDirectory: ModelPaths.activeModelsDirectory)
            model = try await TTS.loadModel(
                modelRepo: TTSModelResolver.repository(for: kind),
                cache: cache
            )
        }
        catch {
            throw TTSError.modelLoadFailed(error.localizedDescription)
        }
    }

    public func generate(
        text: String,
        voice: String?,
        speed: Float?
    ) async throws -> AudioResult {
        if model == nil {
            try await loadModel()
        }

        guard let model else {
            throw TTSError.modelNotLoaded
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw TTSError.invalidArgument("Input text cannot be empty")
        }

        var parameters = model.defaultGenerationParameters
        if let speed, speed > 0, speed != 1 {
            parameters.maxTokens = scaledMaxTokens(parameters.maxTokens, speed: speed)
        }

        do {
            let audio = try await model.generate(
                text: trimmed,
                voice: resolveVoice(voice),
                refAudio: nil,
                refText: nil,
                language: nil,
                generationParameters: parameters
            )
            let samples = audio.asType(.float32).reshaped([-1]).asArray(Float.self)
            let duration = samples.isEmpty ? nil : TimeInterval(samples.count) / TimeInterval(model.sampleRate)
            return .samples(samples, model.sampleRate, duration)
        }
        catch is CancellationError {
            throw TTSError.cancelled
        }
        catch {
            throw TTSError.generationFailed(error.localizedDescription)
        }
    }

    private func resolveVoice(_ voice: String?) -> String? {
        let trimmed = voice?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let trimmed, !trimmed.isEmpty else {
            switch kind {
            case .orpheus:
                return "tara"
            case .marvis:
                return "conversational_a"
            case .vyvo:
                return "en-us-1"
            default:
                return nil
            }
        }

        return trimmed
    }

    private func scaledMaxTokens(_ maxTokens: Int?, speed: Float) -> Int? {
        guard let maxTokens else {
            return nil
        }

        let clamped = max(0.25, min(4.0, speed))
        return max(1, Int(Float(maxTokens) / clamped))
    }
}

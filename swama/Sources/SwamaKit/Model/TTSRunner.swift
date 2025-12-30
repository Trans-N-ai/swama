import Foundation
@preconcurrency import MLXAudio

// MARK: - TTSModelKind

public enum TTSModelKind: String, CaseIterable, Sendable {
    case orpheus
    case marvis
    case chatterbox
    case chatterboxTurbo
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
    public static let availableModels: [String] = [
        "orpheus",
        "marvis",
        "chatterbox",
        "chatterbox-turbo",
        "cosyvoice2",
        "cosyvoice3",
        "outetts",
    ]

    public static func resolve(_ modelName: String) -> TTSModelResolution? {
        let normalized = modelName.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        switch normalized {
        case "mlx-community/orpheus-3b-0.1-ft-4bit",
             "orpheus":
            return TTSModelResolution(kind: .orpheus, cacheKey: TTSModelKind.orpheus.rawValue)
        case "marvis",
             "marvis-ai/marvis-tts-100m-v0.2-mlx-6bit":
            return TTSModelResolution(kind: .marvis, cacheKey: TTSModelKind.marvis.rawValue)
        case "chatterbox",
             "mlx-community/chatterbox-tts-q4":
            return TTSModelResolution(kind: .chatterbox, cacheKey: TTSModelKind.chatterbox.rawValue)
        case "chatterbox_turbo",
             "chatterbox-turbo",
             "mlx-community/chatterbox-turbo-tts-q4":
            return TTSModelResolution(kind: .chatterboxTurbo, cacheKey: TTSModelKind.chatterboxTurbo.rawValue)
        case "cosy-voice2",
             "cosyvoice2",
             "mlx-community/cosyvoice2-0.5b-4bit":
            return TTSModelResolution(kind: .cosyVoice2, cacheKey: TTSModelKind.cosyVoice2.rawValue)
        case "cosy-voice3",
             "cosyvoice3",
             "mlx-community/fun-cosyvoice3-0.5b-2512-4bit":
            return TTSModelResolution(kind: .cosyVoice3, cacheKey: TTSModelKind.cosyVoice3.rawValue)
        case "mlx-community/llama-outetts-1.0-1b-4bit",
             "outetts":
            return TTSModelResolution(kind: .outetts, cacheKey: TTSModelKind.outetts.rawValue)
        default:
            return nil
        }
    }

    public static func voiceIDs(for kind: TTSModelKind) -> [String] {
        switch kind {
        case .orpheus:
            OrpheusEngine.Voice.allCases.map(\.rawValue).sorted()
        case .marvis:
            MarvisEngine.Voice.allCases.map(\.rawValue).sorted()
        case .chatterbox,
             .chatterboxTurbo,
             .cosyVoice2,
             .cosyVoice3,
             .outetts:
            []
        }
    }

    public static func repoIDs(for kind: TTSModelKind) -> [String] {
        switch kind {
        case .orpheus:
            [
                "mlx-community/orpheus-3b-0.1-ft-4bit",
                "mlx-community/snac_24khz",
            ]

        case .marvis:
            [
                "Marvis-AI/marvis-tts-100m-v0.2-MLX-6bit",
            ]

        case .chatterbox:
            [
                "mlx-community/Chatterbox-TTS-q4",
                "mlx-community/S3TokenizerV2",
            ]

        case .chatterboxTurbo:
            [
                "mlx-community/Chatterbox-Turbo-TTS-q4",
                "mlx-community/S3TokenizerV2",
            ]

        case .cosyVoice2:
            [
                "mlx-community/CosyVoice2-0.5B-4bit",
                "mlx-community/S3TokenizerV2",
            ]

        case .cosyVoice3:
            [
                "mlx-community/Fun-CosyVoice3-0.5B-2512-4bit",
                "mlx-community/S3TokenizerV3",
            ]

        case .outetts:
            [
                "mlx-community/Llama-OuteTTS-1.0-1B-4bit",
                "mlx-community/dac-speech-24khz-1.5kbps",
            ]
        }
    }
}

// MARK: - TTSRunner

@MainActor
public final class TTSRunner: @unchecked Sendable {
    private let kind: TTSModelKind
    private var engine: (any TTSEngine)?
    private var cosyVoice2Speaker: CosyVoice2Speaker?
    private var cosyVoice3Speaker: CosyVoice3Speaker?

    public init(kind: TTSModelKind) {
        self.kind = kind
    }

    public func loadModel() async throws {
        if let engine, engine.isLoaded {
            return
        }

        let engine = createEngine()
        try await engine.load()
        self.engine = engine
    }

    public func generate(
        text: String,
        voice: String?,
        speed: Float?
    ) async throws -> AudioResult {
        if engine == nil {
            try await loadModel()
        }

        guard let engine else {
            throw TTSError.modelNotLoaded
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw TTSError.invalidArgument("Input text cannot be empty")
        }

        switch kind {
        case .orpheus:
            let selectedVoice = try resolveOrpheusVoice(voice)
            guard let orpheus = engine as? OrpheusEngine else {
                throw TTSError.invalidArgument("Invalid Orpheus engine configuration")
            }

            return try await orpheus.generate(trimmed, voice: selectedVoice)

        case .marvis:
            _ = speed
            let selectedVoice = try resolveMarvisVoice(voice)
            guard let marvis = engine as? MarvisEngine else {
                throw TTSError.invalidArgument("Invalid Marvis engine configuration")
            }

            return try await marvis.generate(trimmed, voice: selectedVoice)

        case .chatterbox:
            guard let chatterbox = engine as? ChatterboxEngine else {
                throw TTSError.invalidArgument("Invalid Chatterbox engine configuration")
            }

            return try await chatterbox.generate(trimmed)

        case .chatterboxTurbo:
            guard let chatterboxTurbo = engine as? ChatterboxTurboEngine else {
                throw TTSError.invalidArgument("Invalid Chatterbox Turbo engine configuration")
            }

            return try await chatterboxTurbo.generate(trimmed)

        case .cosyVoice2:
            _ = speed
            guard let cosyVoice2 = engine as? CosyVoice2Engine else {
                throw TTSError.invalidArgument("Invalid CosyVoice2 engine configuration")
            }

            if cosyVoice2Speaker == nil {
                cosyVoice2.autoTranscribe = false
                let referenceURL = try await ensureCosyVoiceDefaultReferenceAudio()
                cosyVoice2Speaker = try await cosyVoice2.prepareSpeaker(from: referenceURL, transcription: nil)
            }
            return try await cosyVoice2.generate(trimmed, speaker: cosyVoice2Speaker)

        case .cosyVoice3:
            _ = speed
            guard let cosyVoice3 = engine as? CosyVoice3Engine else {
                throw TTSError.invalidArgument("Invalid CosyVoice3 engine configuration")
            }

            if cosyVoice3Speaker == nil {
                cosyVoice3.autoTranscribe = false
                let referenceURL = try await ensureCosyVoiceDefaultReferenceAudio()
                cosyVoice3Speaker = try await cosyVoice3.prepareSpeaker(from: referenceURL, transcription: nil)
            }
            return try await cosyVoice3.generate(trimmed, speaker: cosyVoice3Speaker)

        case .outetts:
            guard let outetts = engine as? OuteTTSEngine else {
                throw TTSError.invalidArgument("Invalid OuteTTS engine configuration")
            }

            return try await outetts.generate(trimmed)
        }
    }

    private func createEngine() -> any TTSEngine {
        switch kind {
        case .orpheus:
            OrpheusEngine()
        case .marvis:
            MarvisEngine()
        case .chatterbox:
            ChatterboxEngine()
        case .chatterboxTurbo:
            ChatterboxTurboEngine()
        case .cosyVoice2:
            CosyVoice2Engine()
        case .cosyVoice3:
            CosyVoice3Engine()
        case .outetts:
            OuteTTSEngine()
        }
    }

    private func resolveOrpheusVoice(_ voice: String?) throws -> OrpheusEngine.Voice {
        guard let voice, !voice.isEmpty else {
            return .tara
        }

        let normalized = voice.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        if let direct = OrpheusEngine.Voice(rawValue: normalized) {
            return direct
        }

        let available = TTSModelResolver.voiceIDs(for: .orpheus).joined(separator: ", ")
        throw TTSError.invalidVoice("Invalid voice '\(voice)'. Available: \(available)")
    }

    private func resolveMarvisVoice(_ voice: String?) throws -> MarvisEngine.Voice {
        guard let voice, !voice.isEmpty else {
            return .conversationalA
        }

        let normalized = voice.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        if let direct = MarvisEngine.Voice(rawValue: normalized) {
            return direct
        }

        let available = TTSModelResolver.voiceIDs(for: .marvis).joined(separator: ", ")
        throw TTSError.invalidVoice("Invalid voice '\(voice)'. Available: \(available)")
    }

    private func ensureCosyVoiceDefaultReferenceAudio() async throws -> URL {
        let cachesURL = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let directory = cachesURL.appendingPathComponent("swama/tts", isDirectory: true)
        let fileURL = directory.appendingPathComponent("cosyvoice_default.wav")

        if FileManager.default.fileExists(atPath: fileURL.path) {
            return fileURL
        }

        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let url = URL(string: "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav")!
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let httpResponse = response as? HTTPURLResponse,
              (200 ... 299).contains(httpResponse.statusCode)
        else {
            throw TTSError.invalidArgument("Failed to download default reference audio")
        }

        try data.write(to: fileURL, options: [.atomic])
        return fileURL
    }
}

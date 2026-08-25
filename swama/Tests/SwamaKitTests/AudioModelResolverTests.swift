@testable import SwamaKit
import Testing

@Suite("Audio Model Resolution")
struct AudioModelResolverTests {
    @Test func resolvesNativeWhisperAliases() {
        #expect(
            ModelAliasResolver.resolve(name: "whisper-base")
                == "mlx-community/whisper-base-4bit"
        )
        #expect(
            ModelAliasResolver.resolve(name: "whisper-large-v3-turbo-8bit")
                == "mlx-community/whisper-large-v3-turbo-8bit"
        )
    }

    @Test func recognizesNewSTTModelFamilies() {
        #expect(ModelAliasResolver.isAudioModel("moss-transcribe-diarize"))
        #expect(ModelAliasResolver.isAudioModel("openai/whisper-large-v3-turbo"))
        #expect(ModelAliasResolver.isAudioModel("mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"))
        #expect(ModelAliasResolver.isAudioModel("Mediform/canary-1b-v2-mlx-q8"))
        #expect(ModelAliasResolver.isAudioModel("facebook/wav2vec2-base-960h"))
    }

    @Test func resolvesNewTTSAliases() throws {
        let kokoro = try #require(TTSModelResolver.resolve("kokoro"))
        #expect(kokoro.kind == .kokoro)
        #expect(kokoro.repository == "mlx-community/Kokoro-82M-bf16")

        let irodori = try #require(TTSModelResolver.resolve("irodori-tts"))
        #expect(irodori.kind == .irodoriTTS)
        #expect(irodori.repository == "mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit")

        let omniVoice = try #require(TTSModelResolver.resolve("omnivoice"))
        #expect(omniVoice.kind == .omniVoice)
        #expect(omniVoice.repository == "mlx-community/OmniVoice-bf16")
    }

    @Test func preservesQwenTTSVariantRepository() throws {
        let repository = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
        let resolution = try #require(TTSModelResolver.resolve(repository))

        #expect(resolution.kind == .qwen3TTS)
        #expect(resolution.repository == repository)
        #expect(resolution.cacheKey == repository.lowercased())
    }

    @Test func distinguishesMOSSVariants() throws {
        let dialogue = try #require(TTSModelResolver.resolve("moss-ttsd"))
        let local = try #require(TTSModelResolver.resolve("moss-tts-local"))

        #expect(dialogue.kind == .mossTTSD)
        #expect(dialogue.repository == "OpenMOSS-Team/MOSS-TTSD-v1.0")
        #expect(local.kind == .mossTTSLocal)
        #expect(local.repository == "OpenMOSS-Team/MOSS-TTS-Local-Transformer")
    }
}


//
//  ModelAliases.swift
//  SwamaKit
//

import Foundation

public enum ModelAliasResolver {
    // MARK: Public

    /// Resolves a user-provided model name to its full Hugging Face model ID if an alias exists.
    ///
    /// - Parameter name: The model name or alias provided by the user.
    /// - Returns: The resolved full model ID, or the original name if no alias is found.
    public static func resolve(name: String) -> String {
        let lowercasedName = name.lowercased()

        // Check LLM aliases first
        if let resolvedName = aliases[lowercasedName] {
            return resolvedName
        }

        // Check STT (speech-to-text) model aliases
        if let resolvedName = sttAliases[lowercasedName] {
            return resolvedName
        }

        // Check TTS model aliases
        if let resolvedName = ttsAliases[lowercasedName] {
            return resolvedName
        }

        // If it's an STT/TTS model format, return as-is
        if isAudioModel(lowercasedName) || isTTSModel(lowercasedName) {
            return name
        }

        return name
    }

    /// Check if a model name is supported by MLXAudio transcription.
    public static func isAudioModel(_ modelName: String) -> Bool {
        let lowercasedName = modelName.lowercased()
        return lowercasedName.hasPrefix("whisper-") ||
            lowercasedName.hasPrefix("funasr-") ||
            sttModelMarkers.contains(where: lowercasedName.contains) ||
            sttAliases.keys.contains(lowercasedName) ||
            sttAliases.values.contains(where: { $0.lowercased() == lowercasedName })
    }

    /// Check if a model name is a TTS (Text-to-Speech) model
    public static func isTTSModel(_ modelName: String) -> Bool {
        let lowercasedName = modelName.lowercased()
        return ttsAliases.keys.contains(lowercasedName) ||
            ttsAliases.values.contains(where: { $0.lowercased() == lowercasedName })
    }

    /// Resolve an STT model name to an MLXAudio-friendly alias if possible.
    public static func resolveAudioModelName(_ modelName: String) -> String {
        let lowercasedName = modelName.lowercased()

        if sttAliases.keys.contains(lowercasedName) {
            return lowercasedName
        }

        if let alias = sttAliases.first(where: { $0.value.lowercased() == lowercasedName })?.key {
            return alias
        }

        return modelName
    }

    // MARK: Internal

    /// All keys should be lowercase for case-insensitive matching.
    static let aliases: [String: String] = [
        // DeepSeek Family
        "deepseek-r1": "mlx-community/DeepSeek-R1-0528-4bit",
        "deepseek-v3": "mlx-community/DeepSeek-V3-4bit",
        "deepseek-v2.5": "mlx-community/DeepSeek-V2.5-1210-4bit",
        "deepseek-coder": "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx",
        "deepseek-r1-8b": "mlx-community/DeepSeek-R1-0528-Qwen3-8B-8bit",

        // Qwen2.5 Family
        "qwen2.5": "mlx-community/Qwen2.5-7B-Instruct-4bit", // Default for "qwen2.5"
        "qwen2.5-0.5b": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "qwen2.5-1.5b": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        "qwen2.5-3b": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "qwen2.5-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "qwen2.5-14b": "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "qwen2.5-32b": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "qwen2.5-72b": "mlx-community/Qwen2.5-72B-Instruct-4bit",

        // Qwen3 Family
        "qwen3": "mlx-community/Qwen3-8B-4bit", // Default for "qwen3"
        "qwen3-30b": "mlx-community/Qwen3-30B-A3B-4bit",
        "qwen3-30b-2507": "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit",
        "qwen3-1.7b": "mlx-community/Qwen3-1.7B-4bit",
        "qwen3-32b": "mlx-community/Qwen3-32B-4bit",
        "qwen3-235b": "mlx-community/Qwen3-235B-A22B-4bit",

        // Qwen3.5 Family (multimodal, no explicit "vl" suffix in model names)
        "qwen3.5": "mlx-community/Qwen3.5-35B-A3B-4bit", // Default for "qwen3.5"
        "qwen3.5-0.8b": "mlx-community/Qwen3.5-0.8B-4bit",
        "qwen3.5-2b": "mlx-community/Qwen3.5-2B-4bit",
        "qwen3.5-4b": "mlx-community/Qwen3.5-4B-4bit",
        "qwen3.5-9b": "mlx-community/Qwen3.5-9B-4bit",
        "qwen3.5-27b": "mlx-community/Qwen3.5-27B-4bit",
        "qwen3.5-35b": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "qwen3.5-35b-a3b": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "qwen3.5-122b-a10b": "mlx-community/Qwen3.5-122B-A10B-4bit",
        "qwen3.5-397b-a17b": "mlx-community/Qwen3.5-397B-A17B-4bit",

        // Qwen3-VL Family (Vision-Language)
        "qwen3-vl": "mlx-community/Qwen3-VL-4B-Instruct-4bit", // Default for "qwen3-vl"
        "qwen3-vl-2b": "mlx-community/Qwen3-VL-2B-Instruct-4bit",
        "qwen3-vl-4b": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        "qwen3-vl-8b": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
        "qwen3-vl-32b": "mlx-community/Qwen3-VL-32B-Instruct-4bit",
        "qwen3-vl-30b": "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit",
        "qwen3-vl-235b": "mlx-community/Qwen3-VL-235B-A22B-Instruct-4bit",
        // Thinking variants
        "qwen3-vl-2b-thinking": "mlx-community/Qwen3-VL-2B-Thinking-4bit",
        "qwen3-vl-4b-thinking": "mlx-community/Qwen3-VL-4B-Thinking-4bit",
        "qwen3-vl-8b-thinking": "mlx-community/Qwen3-VL-8B-Thinking-4bit",
        "qwen3-vl-32b-thinking": "mlx-community/Qwen3-VL-32B-Thinking-4bit",
        "qwen3-vl-30b-thinking": "mlx-community/Qwen3-VL-30B-A3B-Thinking-4bit",
        "qwen3-vl-235b-thinking": "mlx-community/Qwen3-VL-235B-A22B-Thinking-3bit",

        // Gemma3 Famaly
        "gemma3": "mlx-community/gemma-3-4b-it-4bit", // Default for "gemma3"
        "gemma3-1b": "mlx-community/gemma-3-1b-it-4bit",
        "gemma3-4b": "mlx-community/gemma-3-4b-it-4bit",
        "gemma3-12b": "mlx-community/gemma-3-12b-it-4bit",
        "gemma3-27b": "mlx-community/gemma-3-27b-it-4bit",

        // Llama 3.x Family
        "llama3": "mlx-community/Llama-3-8B-Instruct-4bit", // Default for "llama3"
        "llama3-8b": "mlx-community/Llama-3-8B-Instruct-4bit",
        "llama3.2": "mlx-community/Llama-3.2-3B-Instruct-4bit", // Default for "llama3.2"
        "llama3.2-1b": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "llama3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "llama3.3": "mlx-community/Llama-3.3-70B-Instruct-4bit-DWQ", // Default for "llama3.3"
        "llama3.3-70b": "mlx-community/Llama-3.3-70B-Instruct-4bit-DWQ",

        // SmolLM Family
        "smollm": "mlx-community/SmolLM-135M-Instruct-4bit",

        // GPT-OSS Family
        "gpt-oss": "lmstudio-community/gpt-oss-20b-MLX-8bit", // Default for "gpt-oss"
        "gpt-oss-20b": "lmstudio-community/gpt-oss-20b-MLX-8bit",
        "gpt-oss-120b": "lmstudio-community/gpt-oss-120b-MLX-8bit",
    ]

    /// STT model aliases for MLXAudio.
    /// All keys should be lowercase for case-insensitive matching.
    static let sttAliases: [String: String] = [
        // Whisper is native again in mlx-audio-swift. Keep aliases quantized by default.
        "whisper": "mlx-community/whisper-large-v3-turbo-4bit",
        "whisper-tiny": "mlx-community/whisper-tiny-4bit",
        "whisper-tiny-4bit": "mlx-community/whisper-tiny-4bit",
        "whisper-tiny-8bit": "mlx-community/whisper-tiny-8bit",
        "whisper-tiny-fp16": "mlx-community/whisper-tiny-fp16",
        "whisper-base": "mlx-community/whisper-base-4bit",
        "whisper-base-4bit": "mlx-community/whisper-base-4bit",
        "whisper-base-8bit": "mlx-community/whisper-base-8bit",
        "whisper-base-fp16": "mlx-community/whisper-base-fp16",
        "whisper-small": "mlx-community/whisper-small-4bit",
        "whisper-small-4bit": "mlx-community/whisper-small-4bit",
        "whisper-small-8bit": "mlx-community/whisper-small-8bit",
        "whisper-small-fp16": "mlx-community/whisper-small-fp16",
        "whisper-medium": "mlx-community/whisper-medium-4bit",
        "whisper-medium-4bit": "mlx-community/whisper-medium-4bit",
        "whisper-medium-8bit": "mlx-community/whisper-medium-8bit",
        "whisper-medium-fp16": "mlx-community/whisper-medium-fp16",
        "whisper-large": "mlx-community/whisper-large-v3-4bit",
        "whisper-large-v3": "mlx-community/whisper-large-v3-4bit",
        "whisper-large-v3-4bit": "mlx-community/whisper-large-v3-4bit",
        "whisper-large-v3-8bit": "mlx-community/whisper-large-v3-8bit",
        "whisper-large-v3-fp16": "mlx-community/whisper-large-v3-fp16",
        "whisper-large-turbo": "mlx-community/whisper-large-v3-turbo-4bit",
        "whisper-large-v3-turbo": "mlx-community/whisper-large-v3-turbo-4bit",
        "whisper-large-v3-turbo-4bit": "mlx-community/whisper-large-v3-turbo-4bit",
        "whisper-large-v3-turbo-8bit": "mlx-community/whisper-large-v3-turbo-8bit",
        "whisper-large-v3-turbo-fp16": "mlx-community/whisper-large-v3-turbo-fp16",
        "whisper-tiny-en": "mlx-community/whisper-tiny.en-4bit",
        "whisper-base-en": "mlx-community/whisper-base.en-4bit",
        "whisper-small-en": "mlx-community/whisper-small.en-4bit",
        "whisper-medium-en": "mlx-community/whisper-medium.en-4bit",

        // FunASR models
        "funasr": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "funasr-mlt": "mlx-community/Qwen3-ASR-0.6B-4bit",

        // Native mlx-audio-swift STT models
        "qwen3-asr": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "qwen3-asr-0.6b": "mlx-community/Qwen3-ASR-0.6B-4bit",
        "qwen3-asr-1.7b": "mlx-community/Qwen3-ASR-1.7B-bf16",
        "glmasr": "mlx-community/GLM-ASR-Nano-2512-4bit",
        "glm-asr": "mlx-community/GLM-ASR-Nano-2512-4bit",
        "sensevoice": "mlx-community/SenseVoiceSmall",
        "parakeet": "mlx-community/parakeet-tdt-0.6b-v3",
        "voxtral": "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16",
        "cohere-transcribe": "beshkenadze/cohere-transcribe-03-2026-mlx-fp16",
        "moss-transcribe": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "moss-transcribe-diarize": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "nemotron-asr": "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit",
        "nemotron-asr-en": "animaslabs/nemotron-speech-streaming-en-0.6b-mlx-8bit",
        "canary": "Mediform/canary-1b-v2-mlx-q8",
        "moonshine": "UsefulSensors/moonshine-tiny",
        "wav2vec2": "facebook/wav2vec2-base-960h",
        "mms-asr": "facebook/mms-1b-fl102",
    ]

    /// TTS (Text-to-Speech) model aliases
    /// All keys should be lowercase for case-insensitive matching.
    static let ttsAliases: [String: String] = [
        // Orpheus
        "orpheus": "mlx-community/orpheus-3b-0.1-ft-bf16",

        // Marvis
        "marvis": "Marvis-AI/marvis-tts-250m-v0.2-MLX-8bit",

        // Chatterbox
        "chatterbox": "mlx-community/chatterbox-turbo-4bit",
        "chatterbox-turbo": "mlx-community/chatterbox-turbo-4bit",
        "chatterbox_turbo": "mlx-community/chatterbox-turbo-4bit",

        // New mlx-audio-swift TTS models
        "qwen3-tts": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "vyvo": "mlx-community/VyvoTTS-EN-Beta-4bit",
        "fish-speech": "mlx-community/fish-audio-s2-pro-8bit",
        "soprano": "mlx-community/Soprano-80M-bf16",
        "pocket-tts": "mlx-community/pocket-tts",
        "moss-tts": "OpenMOSS-Team/MOSS-TTS",
        "echo-tts": "mlx-community/echo-tts-base",
        "kokoro": "mlx-community/Kokoro-82M-bf16",
        "kitten-tts": "mlx-community/kitten-tts-mini-0.8",
        "irodori-tts": "mlx-community/Irodori-TTS-600M-v3-VoiceDesign-8bit",
        "omnivoice": "mlx-community/OmniVoice-bf16",
        "moss-ttsd": "OpenMOSS-Team/MOSS-TTSD-v1.0",
        "moss-tts-local": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",

        // Legacy names mapped to Qwen3-TTS to keep request compatibility.
        "outetts": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",

        // CosyVoice models
        "cosyvoice2": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "cosyvoice3": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    ]

    private static let sttModelMarkers = [
        "whisper",
        "qwen3-asr",
        "glm-asr",
        "glmasr",
        "sensevoice",
        "voxtral",
        "cohere",
        "parakeet",
        "firered",
        "fire-red",
        "moss-transcribe-diarize",
        "nemotron",
        "canary",
        "moonshine",
        "wav2vec",
        "mms-",
        "lasr",
        "granite-speech",
    ]
}

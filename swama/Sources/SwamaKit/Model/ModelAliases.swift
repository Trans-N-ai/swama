
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

        // Check audio model aliases
        if let resolvedName = audioAliases[lowercasedName] {
            return resolvedName
        }

        // If it's an audio model format, return as-is
        if isAudioModel(lowercasedName) {
            return name
        }

        return name
    }

    /// Check if a model name is supported by MLXAudio transcription
    public static func isAudioModel(_ modelName: String) -> Bool {
        let lowercasedName = modelName.lowercased()
        return lowercasedName.hasPrefix("whisper-") ||
            lowercasedName.hasPrefix("funasr-") ||
            audioAliases.keys.contains(lowercasedName) ||
            audioAliases.values.contains(modelName)
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

    /// Audio model aliases for MLXAudio (Whisper, FunASR)
    /// All keys should be lowercase for case-insensitive matching.
    static let audioAliases: [String: String] = [
        // Whisper models - default to 4bit quantization for balance of quality and size
        "whisper-tiny": "mlx-community/whisper-tiny-4bit",
        "whisper-base": "mlx-community/whisper-base-4bit",
        "whisper-small": "mlx-community/whisper-small-4bit",
        "whisper-medium": "mlx-community/whisper-medium-4bit",
        "whisper-large": "mlx-community/whisper-large-v3-4bit",
        "whisper-large-v3": "mlx-community/whisper-large-v3-4bit",
        "whisper-large-turbo": "mlx-community/whisper-large-v3-turbo-4bit",
        "whisper": "mlx-community/whisper-large-v3-4bit",

        // English-only variants
        "whisper-tiny-en": "mlx-community/whisper-tiny.en-4bit",
        "whisper-base-en": "mlx-community/whisper-base.en-4bit",
        "whisper-small-en": "mlx-community/whisper-small.en-4bit",
        "whisper-medium-en": "mlx-community/whisper-medium.en-4bit",

        // FunASR models
        "funasr": "mlx-community/Fun-ASR-Nano-2512-4bit",
        "funasr-mlt": "mlx-community/Fun-ASR-MLT-Nano-2512-4bit",
    ]
}


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
        if let resolvedName = aliases[lowercasedName] {
            return resolvedName
        }
        // Check WhisperKit aliases
        if let resolvedName = whisperKitAliases[lowercasedName] {
            return resolvedName
        }
        return name
    }

    /// Check if a model name is a WhisperKit model
    public static func isWhisperKitModel(_ modelName: String) -> Bool {
        let lowercasedName = modelName.lowercased()
        return lowercasedName.hasPrefix("whisper-") ||
            lowercasedName.hasPrefix("whisperkit-") ||
            whisperKitAliases.values.contains(modelName)
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
        "qwen3-1.7b": "mlx-community/Qwen3-1.7B-4bit",
        "qwen3-32b": "mlx-community/Qwen3-32B-4bit",
        "qwen3-235b": "mlx-community/Qwen3-235B-A22B-4bit",

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
    ]

    /// WhisperKit model aliases mapping user-friendly names to HuggingFace folder names
    static let whisperKitAliases: [String: String] = [
        // OpenAI standard naming (primary)
        "whisper-tiny": "openai_whisper-tiny",
        "whisper-tiny.en": "openai_whisper-tiny.en",
        "whisper-base": "openai_whisper-base",
        "whisper-base.en": "openai_whisper-base.en",
        "whisper-small": "openai_whisper-small",
        "whisper-small.en": "openai_whisper-small.en",
        "whisper-medium": "openai_whisper-medium",
        "whisper-medium.en": "openai_whisper-medium.en",
        "whisper-large-v2": "openai_whisper-large-v2",
        "whisper-large-v3": "openai_whisper-large-v3",
        "whisper-large": "openai_whisper-large-v3", // Latest large model alias
    ]
}

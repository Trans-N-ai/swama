import AVFoundation
import Foundation
@preconcurrency import WhisperKit

// MARK: - TranscriptionResponseFormat

/// Response format for transcription output
public enum TranscriptionResponseFormat {
    case simple // Just text
    case verboseJson // Text with timing and metadata
}

// MARK: - TranscriptionOutput

/// Unified transcription output
public enum TranscriptionOutput: Sendable {
    case simple(String)
    case detailed([TranscriptionResult])
}

// MARK: - WhisperKitRunner

/// WhisperKit-based implementation for Whisper speech recognition
public class WhisperKitRunner: @unchecked Sendable {
    private var whisperKit: WhisperKit?
    private var isRunning = false

    public init() {}

    /// Load WhisperKit model from a specific folder
    /// - Parameter modelFolder: Path to the model folder
    public func loadModel(from modelFolder: String) async throws {
        let config = WhisperKitConfig(
            modelFolder: modelFolder,
            verbose: true
        )

        self.whisperKit = try await WhisperKit(config)
    }

    /// Validate and prepare audio file for transcription
    /// - Parameter url: URL to audio file
    /// - Returns: Path to the prepared audio file (may be converted)
    private func validateAndPrepareAudio(at url: URL) async throws -> String {
        // Check if file exists
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw WhisperError.transcriptionFailed("Audio file not found: \(url.path)")
        }

        // Basic file size check
        let fileSize = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        if fileSize == 0 {
            throw WhisperError.transcriptionFailed("Audio file is empty")
        }

        // Let WhisperKit handle the actual audio processing
        return url.path
    }

    /// Transcribe audio file with unified options (main interface)
    /// - Parameters:
    ///   - url: URL to audio file
    ///   - language: Language code (e.g., "en", "zh", "ja") or nil for auto-detection
    ///   - temperature: Sampling temperature (0.0-1.0)
    ///   - responseFormat: Response format for output
    /// - Returns: Full transcription results array with timing information
    public func transcribe(
        audioFile url: URL,
        language: String? = nil,
        temperature: Float = 0.0,
        responseFormat: TranscriptionResponseFormat = .simple
    ) async throws -> TranscriptionOutput {
        // Prevent concurrent runs
        while isRunning {
            try await Task.sleep(nanoseconds: 10_000_000) // 10ms
        }

        isRunning = true
        defer { isRunning = false }

        guard let whisperKit else {
            throw WhisperError.modelLoadFailed("WhisperKit not initialized")
        }

        // Validate audio file first
        let audioPath = try await validateAndPrepareAudio(at: url)

        // Create DecodingOptions based on parameters
        let options = DecodingOptions(
            verbose: responseFormat == .verboseJson,
            task: .transcribe, // Always transcribe, never translate
            language: language,
            temperature: temperature,
            detectLanguage: language == nil, // Auto-detect if no language specified
            withoutTimestamps: responseFormat == .simple,
            wordTimestamps: responseFormat == .verboseJson
        )

        // Get detailed results with better error handling
        do {
            let results = try await whisperKit.transcribe(audioPath: audioPath, decodeOptions: options)

            // Return appropriate format
            switch responseFormat {
            case .simple:
                let text = results.compactMap(\.text).joined(separator: " ")
                return .simple(Self.cleanTranscriptionText(text))

            case .verboseJson:
                // Clean text in each segment
                let cleanedResults = results.map { result in
                    var cleanedResult = result
                    cleanedResult.segments = result.segments.map { segment in
                        var cleanedSegment = segment
                        cleanedSegment.text = Self.cleanTranscriptionText(segment.text)
                        return cleanedSegment
                    }
                    return cleanedResult
                }
                return .detailed(cleanedResults)
            }
        }
        catch {
            // Provide more specific error messages
            let errorMessage = error.localizedDescription
            if errorMessage.contains("1954115647") || errorMessage.contains("coreaudio.avfaudio") {
                throw WhisperError
                    .transcriptionFailed(
                        "Audio encoding error: The audio file format is not supported or the file is corrupted. Please try converting to WAV format (16kHz, mono) and try again. Original error: \(errorMessage)"
                    )
            }
            else {
                throw WhisperError.transcriptionFailed("Transcription failed: \(errorMessage)")
            }
        }
    }

    // MARK: - Helper Methods

    /// Clean up Whisper special tokens from transcription text
    private static func cleanTranscriptionText(_ text: String) -> String {
        // Remove Whisper special tokens like <|startoftranscript|>, <|en|>, <|transcribe|>, timestamps, etc.
        let pattern = #"<\|[^|]*\|>"#
        let regex = try! NSRegularExpression(pattern: pattern, options: [])
        let range = NSRange(text.startIndex ..< text.endIndex, in: text)
        let cleanedText = regex.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: "")

        // Trim whitespace and return
        return cleanedText.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Convenience methods for model path integration with Swama
public extension WhisperKitRunner {
    /// Load model from Swama's model directory structure
    /// - Parameter modelIdentifier: Model identifier (e.g. "whisper-tiny", "whisper-base")
    func loadModel(_ modelIdentifier: String) async throws {
        // Always use ModelAliasResolver for WhisperKit models
        guard ModelAliasResolver.isWhisperKitModel(modelIdentifier) else {
            throw WhisperError
                .invalidInput(
                    "Model '\(modelIdentifier)' is not a WhisperKit model. Use whisper-tiny, whisper-base, etc."
                )
        }

        // Check if model exists locally
        guard whisperKitModelExistsLocally(modelIdentifier) else {
            // Provide helpful error message with download instructions
            let availableModels = availableWhisperKitModels()
            let availableModelsList = availableModels.isEmpty ? "None" : availableModels.joined(separator: ", ")

            throw WhisperError.modelNotFound("""
            WhisperKit model '\(modelIdentifier)' not found locally.

            Available local models: \(availableModelsList)

            To download this model, use:
                swama pull \(modelIdentifier)

            Or download manually using the WhisperKit CLI.
            """)
        }

        // Load the local model
        let modelDir = getWhisperKitModelDirectory(for: modelIdentifier)
        print("Loading local WhisperKit model from: \(modelDir.path)")

        try await loadModel(from: modelDir.path)
        print("✅ Successfully loaded local WhisperKit model")
    }
}

// MARK: - Whisper Errors

public enum WhisperError: Error, LocalizedError {
    case modelLoadFailed(String)
    case audioLoadFailed(String)
    case transcriptionFailed(String)
    case invalidInput(String)
    case modelNotFound(String)

    public var errorDescription: String? {
        switch self {
        case let .modelLoadFailed(message):
            "Failed to load Whisper model: \(message)"
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

// MARK: - WhisperKit Helper Methods

/// WhisperKit-specific helper methods
extension WhisperKitRunner {
    /// Get the local directory path for a WhisperKit model
    private func getWhisperKitModelDirectory(for modelName: String) -> URL {
        let resolvedName = ModelAliasResolver.whisperKitAliases[modelName.lowercased()] ?? modelName
        return ModelPaths.getModelDirectory(for: "whisperkit/\(resolvedName)")
    }

    /// Check if a WhisperKit model exists locally
    private func whisperKitModelExistsLocally(_ modelName: String) -> Bool {
        let modelDir = getWhisperKitModelDirectory(for: modelName)
        let requiredFiles = ["AudioEncoder.mlmodelc", "TextDecoder.mlmodelc", "MelSpectrogram.mlmodelc", "config.json"]

        return requiredFiles.allSatisfy { fileName in
            let filePath = modelDir.appendingPathComponent(fileName)
            return FileManager.default.fileExists(atPath: filePath.path)
        }
    }

    /// List all locally available WhisperKit models
    private func availableWhisperKitModels() -> [String] {
        let whisperKitModelsDirectory = ModelPaths.getModelDirectory(for: "whisperkit")

        guard FileManager.default.fileExists(atPath: whisperKitModelsDirectory.path) else {
            return []
        }

        do {
            let modelDirs = try FileManager.default.contentsOfDirectory(
                at: whisperKitModelsDirectory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )

            return modelDirs.compactMap { modelDir in
                guard modelDir.hasDirectoryPath else {
                    return nil
                }

                let modelName = modelDir.lastPathComponent
                return whisperKitModelExistsLocally(modelName) ? modelName : nil
            }
        }
        catch {
            return []
        }
    }
}

private extension URL {
    var hasDirectoryPath: Bool {
        (try? resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
    }
}

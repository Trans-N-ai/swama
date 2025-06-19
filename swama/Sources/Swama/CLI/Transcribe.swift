import ArgumentParser
import Foundation
import MLX
@preconcurrency import MLXLMCommon
import SwamaKit
@preconcurrency import WhisperKit

// MARK: - ResponseFormat

enum ResponseFormat: String, CaseIterable, ExpressibleByArgument {
    case simple
    case json
    case verbose

    var defaultValueDescription: String {
        "simple"
    }

    static var allValueStrings: [String] {
        allCases.map(\.rawValue)
    }
}

// MARK: - Transcribe

struct Transcribe: AsyncParsableCommand {
    static let configuration: CommandConfiguration = .init(
        abstract: "Transcribe audio file to text using Whisper"
    )

    @Argument(help: "Audio file path (WAV format recommended)")
    var audioFile: String

    @Option(name: [.customShort("m"), .long], help: "Whisper model name or alias")
    var model: String = "whisper-base"

    @Option(
        name: [.customShort("l"), .long],
        help: "Language code (e.g., 'en', 'zh', 'ja'). Auto-detect if not specified"
    )
    var language: String?

    @Option(name: [.customShort("f"), .long], help: "Response format: simple, json, verbose")
    var format: ResponseFormat = .simple

    @Option(name: [.customShort("t"), .long], help: "Sampling temperature (0.0-1.0)")
    var temperature: Float = 0.0

    @Option(name: [.customShort("p"), .long], help: "Optional prompt to guide transcription")
    var prompt: String?

    @Flag(name: .long, help: "Show detailed output with timestamps (equivalent to --format verbose)")
    var verbose: Bool = false

    func run() async throws {
        let audioURL = URL(fileURLWithPath: audioFile)

        // Check if file exists
        guard FileManager.default.fileExists(atPath: audioFile) else {
            print("Error: Audio file not found at \(audioFile)")
            throw ExitCode.failure
        }

        // Validate temperature range
        guard temperature >= 0.0, temperature <= 1.0 else {
            print("❌ Error: Temperature must be between 0.0 and 1.0")
            throw ExitCode.failure
        }

        // Determine final response format (verbose flag overrides format option)
        let finalFormat: TranscriptionResponseFormat = verbose ? .verboseJson : {
            switch format {
            case .simple: .simple
            case .json: .simple // CLI treats json same as simple for now
            case .verbose: .verboseJson
            }
        }()

        print("🎤 Loading Whisper model: \(model)")
        if let lang = language {
            print("🌍 Language: \(lang)")
        }
        if temperature > 0.0 {
            print("🌡️ Temperature: \(temperature)")
        }
        if let promptText = prompt {
            print("💭 Prompt: \(promptText)")
        }

        // Check if this is a WhisperKit model
        guard ModelAliasResolver.isWhisperKitModel(model) else {
            print("❌ Error: '\(model)' is not a valid WhisperKit model.")
            print("   Use whisper-tiny, whisper-base, whisper-small, whisper-medium, or whisper-large")
            throw ExitCode.failure
        }

        do {
            // Load WhisperKit model
            print("📥 Loading model...")
            let runner = WhisperKitRunner()

            // Load using WhisperKit-specific logic
            try await runner.loadModel(model)
            print("✅ Model loaded successfully")

            print("🎧 Processing audio file...")

            // Use the new unified transcribe method with user parameters
            let result = try await runner.transcribe(
                audioFile: audioURL,
                language: language,
                temperature: temperature,
                responseFormat: finalFormat
            )

            // Handle different output formats
            try await handleOutput(result: result, format: format, verbose: verbose)
        }
        catch {
            print("❌ Transcription failed: \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    // MARK: - Output Handling

    private func handleOutput(
        result: TranscriptionOutput,
        format: ResponseFormat,
        verbose: Bool
    ) async throws {
        switch result {
        case let .simple(text):
            if format == .json {
                // Output as JSON
                let jsonOutput = ["text": text]
                let jsonData = try JSONSerialization.data(withJSONObject: jsonOutput, options: .prettyPrinted)
                print(String(data: jsonData, encoding: .utf8) ?? "")
            }
            else {
                // Simple text output
                print("\n📝 Transcription:")
                print("================")
                print(text)
                print("================")
            }

        case let .detailed(results):
            if verbose || format == .verbose {
                // Detailed output with timestamps
                print("\n📝 Detailed Transcription:")
                print("==========================")

                for (index, transcriptionResult) in results.enumerated() {
                    print("\n--- Result \(index + 1) ---")
                    print("Language: \(transcriptionResult.language)")

                    for segment in transcriptionResult.segments {
                        let startTime = formatTime(segment.start)
                        let endTime = formatTime(segment.end)
                        print("[\(startTime) -> \(endTime)] \(segment.text)")
                    }
                }
                print("==========================")
            }
            else if format == .json {
                // JSON output for detailed results
                let segments = results.flatMap(\.segments).map { segment in
                    [
                        "id": segment.id,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    ] as [String: Any]
                }

                let jsonOutput = [
                    "text": results.compactMap(\.text).joined(separator: " "),
                    "segments": segments
                ] as [String: Any]

                let jsonData = try JSONSerialization.data(withJSONObject: jsonOutput, options: .prettyPrinted)
                print(String(data: jsonData, encoding: .utf8) ?? "")
            }
            else {
                // Simple text output from detailed results
                let transcription = results.compactMap(\.text).joined(separator: " ")
                print("\n📝 Transcription:")
                print("================")
                print(transcription)
                print("================")
            }
        }
    }

    private func formatTime(_ seconds: Float) -> String {
        let totalSeconds = Int(seconds)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let secs = totalSeconds % 60
        let milliseconds = Int((seconds - Float(totalSeconds)) * 1000)

        if hours > 0 {
            return String(format: "%d:%02d:%02d.%03d", hours, minutes, secs, milliseconds)
        }
        else {
            return String(format: "%02d:%02d.%03d", minutes, secs, milliseconds)
        }
    }
}

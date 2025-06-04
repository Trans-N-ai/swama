import ArgumentParser
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import SwamaKit

struct Run: AsyncParsableCommand {
    static let configuration: CommandConfiguration = .init(
        abstract: "Run a local model with a prompt using MLX"
    )

    @Argument(help: "Model name or alias, e.g. qwen3 or mlx-community/Llama-3.2-1B-Instruct-4bit")
    var modelName: String

    @Argument(help: "Prompt to run")
    var prompt: String

    @Option(name: [.customShort("t"), .long], help: "Sampling temperature")
    var temperature: Float = 0.6

    @Option(name: [.long], help: "Top-p (nucleus sampling)")
    var topP: Float = 1.0

    @Option(name: [.customShort("n"), .long], help: "Maximum number of tokens to generate")
    var maxTokens: Int?

    @Option(name: [.long], help: "Repetition penalty")
    var repetitionPenalty: Float?

    func run() async throws {
        let resolvedModelName = ModelAliasResolver.resolve(name: modelName)

        // Check if model exists locally
        let localModels = ModelManager.models()
        var modelExists = true
        if !localModels.contains(where: { $0.id == resolvedModelName }) {
            modelExists = false
            fputs("Model \(resolvedModelName) not found locally.\n", stdout)
            fflush(stdout)
            do {
                try await ModelDownloader.downloadModel(resolvedModelName: resolvedModelName)
            }
            catch {
                fputs("Failed to download model '\(resolvedModelName)': \(error.localizedDescription)\n", stderr)
                // Re-throw the error to stop execution if download fails, as the model is needed to run.
                throw error
            }
        }

        // Animation for model loading and response generation
        let animatedMessagePrefix = "Generating response"
        var stopAnimationSignal = false
        let spinnerFrames = ["/", "-", "\\", "|"]
        var animationDisplayTask: Task<Void, Never>?

        // Defer block to ensure animation line is cleared if an error occurs or scope is exited prematurely
        defer {
            if animationDisplayTask != nil, !stopAnimationSignal {
                stopAnimationSignal = true
                animationDisplayTask?.cancel() // Request cancellation

                // Perform final cleanup of the animation line
                let cleanupMessageSample =
                    "\(animatedMessagePrefix)... \(spinnerFrames[0])  " // Base message + ... + spinner + few spaces
                let lineToClear = String(
                    repeating: " ",
                    count: cleanupMessageSample.utf8.count + 5
                ) // Extra margin for safety
                fputs("\r\(lineToClear)\r", stdout)
                fflush(stdout)
            }
        }

        animationDisplayTask = Task.detached {
            var frameIndex = 0
            let messagePart = "\(animatedMessagePrefix)... " // e.g., "Generating response... "

            while !stopAnimationSignal, !Task.isCancelled {
                let currentFrameChar = spinnerFrames[frameIndex % spinnerFrames.count]
                // Print: carriage return, message, spinner char, then a space to clear any previous wider char.
                fputs("\r\(messagePart)\(currentFrameChar) ", stdout)
                fflush(stdout)
                frameIndex += 1
                do {
                    // Sleep for a short duration. Task.sleep is cancellation-aware.
                    try await Task.sleep(nanoseconds: 120_000_000) // 120ms
                }
                catch {
                    // If sleep is cancelled (e.g., task is cancelled), break the loop.
                    break
                }
            }
            // The animation task itself doesn't do the final clear;
            // that's handled by the main thread's explicit cleanup or the defer block.
        }

        let container = try await loadModelContainer(modelName: resolvedModelName)
        let runner = ModelRunner(container: container)

        let output = try await runner.run(
            prompt: prompt,
            parameters: .init(
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty
            )
        )

        // Explicitly stop and clear animation BEFORE printing the final output
        if animationDisplayTask != nil { // Check if animation was actually started
            stopAnimationSignal = true
            animationDisplayTask?.cancel()

            // Short pause to allow the animation task to process cancellation and stop printing.
            // This helps prevent the final clear from racing with the animation's last frame.
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms

            // Perform the definitive clear of the animation line
            let cleanupMessageSample = "\(animatedMessagePrefix)... \(spinnerFrames[0])  "
            let lineToClear = String(repeating: " ", count: cleanupMessageSample.utf8.count + 5)
            fputs("\r\(lineToClear)\r", stdout)
            fflush(stdout)
        }

        print(output)
    }
}

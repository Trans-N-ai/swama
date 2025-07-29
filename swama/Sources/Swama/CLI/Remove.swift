import ArgumentParser
import Foundation
import SwamaKit

struct Remove: AsyncParsableCommand {
    static let configuration: CommandConfiguration = .init(
        commandName: "rm",
        abstract: "Remove a model from local storage"
    )

    @Argument(
        help: "Model name or alias to remove, e.g. qwen3-30b, whisperkit-base, or mlx-community/Llama-3.2-1B-Instruct-4bit"
    )
    var model: String

    @Flag(name: .shortAndLong, help: "Force removal without confirmation prompt")
    var force: Bool = false

    func run() async throws {
        // Resolve model name using the same logic as other commands
        let resolvedModelName = ModelAliasResolver.resolve(name: model)

        // Show resolved name if different from input
        if model != resolvedModelName {
            print("Info: Resolved model alias '\(model)' to '\(resolvedModelName)'")
        }

        // Check if model exists
        guard ModelPaths.modelExistsLocally(resolvedModelName) else {
            print("Error: Model '\(resolvedModelName)' not found locally.")
            throw ExitCode.failure
        }

        // Get model info for confirmation
        let models = ModelManager.models()
        let modelInfo = models.first { $0.id == resolvedModelName }

        // Confirmation prompt (unless --force is used)
        if !force {
            print("Are you sure you want to remove model '\(resolvedModelName)'?")
            if let modelInfo {
                let sizeStr = ByteCountFormatter.string(fromByteCount: modelInfo.sizeInBytes, countStyle: .file)
                print("Size: \(sizeStr)")
            }
            print("This action cannot be undone. Type 'y' or 'yes' to confirm:")

            guard let input = readLine()?.lowercased(),
                  input == "y" || input == "yes"
            else {
                print("Removal cancelled.")
                return
            }
        }

        // Remove model from disk
        do {
            let wasRemoved = try ModelPaths.removeModel(resolvedModelName)
            if wasRemoved {
                print("Successfully removed model '\(resolvedModelName)'")

                // Note: ModelPool cleanup is skipped to avoid MLX initialization issues
                // Models will be automatically evicted from memory when needed
            }
            else {
                print("Error: Model '\(resolvedModelName)' not found on disk.")
                throw ExitCode.failure
            }
        }
        catch {
            print("Error removing model '\(resolvedModelName)': \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }
}

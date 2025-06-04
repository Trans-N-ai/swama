import ArgumentParser
import Foundation
import SwamaKit

struct Pull: AsyncParsableCommand {
    static let configuration: CommandConfiguration = .init(
        abstract: "Pull MLX model from HuggingFace with resume support"
    )

    @Argument(help: "Model name or alias, e.g. qwen3-30b or mlx-community/Llama-3.2-1B-Instruct-4bit")
    var model: String

    func run() async throws {
        let resolvedModelName = ModelAliasResolver.resolve(name: model)
        if model != resolvedModelName {
            // Use fputs to stdout to behave like print, ensuring it appears on a new line
            fputs("Info: Resolved model alias '\(model)' to '\(resolvedModelName)'\n", stdout)
            fflush(stdout)
        }
        // The main print "Pulling model..." and completion messages are handled by ModelDownloader
        try await ModelDownloader.downloadModel(resolvedModelName: resolvedModelName)
    }
}

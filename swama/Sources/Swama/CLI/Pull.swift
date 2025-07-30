import ArgumentParser
import Foundation
import SwamaKit

struct Pull: AsyncParsableCommand {
    static let configuration: CommandConfiguration = .init(
        abstract: "Pull model from HuggingFace (MLX or WhisperKit models)"
    )

    @Argument(help: "Model name or alias, e.g. qwen3-30b, whisperkit-base, or mlx-community/Llama-3.2-1B-Instruct-4bit")
    var model: String

    func run() async throws {
        _ = try await ModelDownloader.fetchModel(modelName: model)
    }
}

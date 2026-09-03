import ArgumentParser
import Foundation
import SwamaKit

struct Pull: SwamaLoggedCommand {
    static let configuration: CommandConfiguration = .init(
        abstract: "Pull model from HuggingFace (supports LLM, audio STT, and TTS models)"
    )

    @Argument(help: "Model name or alias, e.g. qwen3-30b or mlx-community/Llama-3.2-1B-Instruct-4bit")
    var model: String

    func runLogged() async throws {
        _ = try await ModelDownloader.fetchModel(modelName: model)
    }
}

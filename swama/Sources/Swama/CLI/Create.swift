import ArgumentParser
import Foundation
import SwamaKit

struct Create: AsyncParsableCommand {
    static let configuration: CommandConfiguration = .init(
        abstract: "Create a model entry from a user-specified path and name. The model metadata (.swama-meta.json) will be saved to ~/.swama/models/<name>, allowing the model to be run later using this name."
    )

    @Argument(help: "Path to user-provided path, e.g.  '/path/to/model'")
    var path: String

    @Option(name: .shortAndLong, help: "Output directory for the created model or project")
    var name: String

    func run() async throws {
        print("Creating model from path: \(path) with name: \(name)")

        try await ModelCreator.run(from: path, name: name)

        print("Model created successfully at \(ModelPaths.activeModelsDirectory.appendingPathComponent(name).path)")
    }
}

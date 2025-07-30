import Foundation

public enum ModelCreator {
    public static func run(from path: String, name: String) async throws {
        ModelDownloader.printMessage("Creating model from path: \(path) with name: \(name)")

        let modelDir = ModelPaths.preferredModelsDirectory.appendingPathComponent(name)
        let sourceURL = URL(fileURLWithPath: path)

        // Check if the model directory already exists
        if FileManager.default.fileExists(atPath: modelDir.path) {
            throw NSError(
                domain: "ModelCreatorError",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Model directory already exists at \(modelDir.path)"]
            )
        }

        // Create the model directory
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true, attributes: nil)

        // Write metadata
        try ModelDownloader.writeModelMetadata(modelName: name, modelDir: sourceURL)
    }
}

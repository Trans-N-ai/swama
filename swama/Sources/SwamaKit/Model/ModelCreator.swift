import Foundation

public enum ModelCreator {
    public static func run(from path: String, name: String) async throws {
        ModelDownloader.printMessage("Creating model from path: \(path) with name: \(name)")

        let modelDir = ModelPaths.activeModelsDirectory.appendingPathComponent(name)
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

        // Copy all files from source to target directory
        let fileManager = FileManager.default
        let sourceFiles = try fileManager.contentsOfDirectory(at: sourceURL, includingPropertiesForKeys: nil)

        for sourceFile in sourceFiles {
            let fileName = sourceFile.lastPathComponent
            let destinationFile = modelDir.appendingPathComponent(fileName)
            try fileManager.copyItem(at: sourceFile, to: destinationFile)
        }

        // Write metadata to the target model directory
        try ModelDownloader.writeModelMetadata(modelName: name, modelDir: modelDir)
    }
}

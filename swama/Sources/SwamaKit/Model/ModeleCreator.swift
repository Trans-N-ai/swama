import Foundation

public enum ModeleCreator {
    public static func createdModel(from path: String, name: String) async throws {
        ModelDownloader.printMessage("Creating model from path: \(path) with name: \(name)")

        let modelDir = ModelPaths.localModelsDirectory.appendingPathComponent(name)
        let metaPath = modelDir.appendingPathComponent(".swama-meta.json")
        let modelPath = URL(fileURLWithPath: path)

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
        try writeUserModelMetadata(modelName: name, modelPath: modelPath, metaPath: metaPath)
    }

    static func writeUserModelMetadata(modelName: String, modelPath: URL, metaPath: URL) throws {
        let size = try ModelDownloader.calculateFolderSize(at: modelPath)
        let created = Int(Date().timeIntervalSince1970)
        let metadata: [String: Any] = [
            "id": modelName,
            "object": "model",
            "created": created,
            "owned_by": "swama",
            "size_in_bytes": size,
            "path": modelPath.path
        ]

        let data = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted])
        try data.write(to: metaPath)

        ModelDownloader.printMessage("📝 Metadata written to .swama-meta.json")
    }
}
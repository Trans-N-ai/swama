import Foundation
@testable import SwamaRuntime
import Testing

// MARK: - ModelPathsValidationTests

/// Positive cases for the model ID and relative path grammar. The existing boundary tests only
/// check that bad input is rejected, so a validator that rejects everything (as a Swift 6.4
/// Release build of the unapplied `CharacterSet.controlCharacters.contains` reference did)
/// passed them. Run these in Release too: that miscompile does not reproduce in Debug.
@Suite("Model path validation")
struct ModelPathsValidationTests {
    @Test(arguments: [
        "mlx-community/Qwen3.5-0.8B-4bit",
        "Qwen/Qwen3-4B-MLX-4bit",
        "org/model",
        "qwen3",
        "gemma3_4b.v2",
    ])
    func acceptsValidModelIdentifiers(_ modelName: String) {
        #expect(ModelPaths.isValidModelIdentifier(modelName))
    }

    @Test(arguments: [
        "",
        " org/model",
        "/org/model",
        "org\\model",
        "org/../model",
        "a/b/c",
        "org/mo\u{0007}del",
        "org/mo\u{0085}del",
    ])
    func rejectsInvalidModelIdentifiers(_ modelName: String) {
        #expect(ModelPaths.isValidModelIdentifier(modelName) == false)
    }

    @Test func acceptsContainedRelativePath() throws {
        let root = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-modelpaths-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let url = try ModelPaths.containedURL(in: root, relativePath: "org/model/config.json")
        #expect(url.standardizedFileURL.path.hasPrefix(root.standardizedFileURL.path))
        #expect(url.lastPathComponent == "config.json")
    }

    @Test func rejectsRelativePathWithControlCharacter() {
        let root = FileManager.default.temporaryDirectory
        #expect(throws: ModelPathError.self) {
            _ = try ModelPaths.containedURL(in: root, relativePath: "org/mo\u{0007}del")
        }
    }
}

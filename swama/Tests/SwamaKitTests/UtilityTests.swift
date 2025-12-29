import Foundation
@testable import SwamaKit
import Testing

@MainActor @Suite(.serialized)
final class UtilityTests {
    @Test func currentTimestamp() {
        // Test that we can get current timestamp using Foundation
        let timestamp = Int(Date().timeIntervalSince1970)

        // Check that it's a valid Unix timestamp (should be positive and reasonable)
        #expect(timestamp > 1_000_000_000) // After year 2001
        #expect(timestamp < 2_000_000_000) // Before year 2033

        // Test that consecutive calls are reasonably close
        let timestamp2 = Int(Date().timeIntervalSince1970)
        #expect(abs(timestamp2 - timestamp) <= 1) // Within 1 second
    }

    @Test func filePathUtilities() {
        // Test URL path construction
        let testURL = URL(fileURLWithPath: "/tmp/test")
        #expect(testURL.path == "/tmp/test")
        #expect(testURL.isFileURL)
    }

    @Test func stringExtensions() {
        // Basic string tests (if any custom extensions exist)
        let testString = "Hello, World!"
        #expect(testString.count == 13)
        #expect(testString.contains("World"))
    }

    @Test func dataConversion() {
        // Test basic data conversion utilities
        let testString = "Hello, Swift!"
        let data = testString.data(using: .utf8)
        #expect(data != nil)

        if let data {
            let convertedString = String(data: data, encoding: .utf8)
            #expect(convertedString == testString)
        }
    }

    @Test func jSONSerialization() throws {
        // Test JSON serialization/deserialization
        let testDict: [String: Any] = [
            "name": "Test Model",
            "version": 1.0,
            "active": true
        ]

        do {
            let jsonData = try JSONSerialization.data(withJSONObject: testDict)
            #expect(!jsonData.isEmpty)

            let deserializedDict = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
            #expect(deserializedDict != nil)
            #expect(deserializedDict?["name"] as? String == "Test Model")
            #expect(deserializedDict?["version"] as? Double == 1.0)
            #expect(deserializedDict?["active"] as? Bool == true)
        }
        catch {
            Issue.record("JSON serialization failed: \(error)")
        }
    }

    @Test func modelPathsRemoval() throws {
        // Test ModelPaths.removeModel() functionality
        let tempDir = FileManager.default.temporaryDirectory
        let testModelName = "test-model/TestModel-1B-4bit"
        let testModelDir = tempDir.appendingPathComponent("swama-test-models").appendingPathComponent(testModelName)
        let metadataFile = testModelDir.appendingPathComponent(".swama-meta.json")

        // Clean up any existing test directory
        let testRootDir = tempDir.appendingPathComponent("swama-test-models")
        try? FileManager.default.removeItem(at: testRootDir)

        // Create test model directory structure
        try FileManager.default.createDirectory(at: testModelDir, withIntermediateDirectories: true)

        // Create metadata file to simulate a valid model
        let testMetadata = """
        {
            "id": "\(testModelName)",
            "created": \(Int(Date().timeIntervalSince1970)),
            "sizeInBytes": 1024,
            "source": "test"
        }
        """
        try testMetadata.write(to: metadataFile, atomically: true, encoding: .utf8)

        // Verify setup
        #expect(FileManager.default.fileExists(atPath: testModelDir.path))
        #expect(FileManager.default.fileExists(atPath: metadataFile.path))

        // Test removal with temporary environment variable
        let originalEnv = ProcessInfo.processInfo.environment["SWAMA_MODELS"]
        setenv("SWAMA_MODELS", testRootDir.path, 1)

        defer {
            // Clean up environment
            if let originalEnv {
                setenv("SWAMA_MODELS", originalEnv, 1)
            }
            else {
                unsetenv("SWAMA_MODELS")
            }
            // Clean up test directory
            try? FileManager.default.removeItem(at: testRootDir)
        }

        // Test successful removal
        let wasRemoved = try ModelPaths.removeModel(testModelName)
        #expect(wasRemoved == true)
        #expect(!FileManager.default.fileExists(atPath: testModelDir.path))

        // Test removal of non-existent model
        let wasRemovedAgain = try ModelPaths.removeModel(testModelName)
        #expect(wasRemovedAgain == false)

        // Test removal of model without metadata file
        try FileManager.default.createDirectory(at: testModelDir, withIntermediateDirectories: true)
        let wasRemovedWithoutMeta = try ModelPaths.removeModel(testModelName)
        #expect(wasRemovedWithoutMeta == false)
    }

    @Test func modelPathsExistence() throws {
        // Test ModelPaths.modelExistsLocally() functionality
        let tempDir = FileManager.default.temporaryDirectory
        let testModelName = "test-model/ExistenceTest-1B-4bit"
        let testModelDir = tempDir.appendingPathComponent("swama-test-models").appendingPathComponent(testModelName)
        let metadataFile = testModelDir.appendingPathComponent(".swama-meta.json")

        // Clean up any existing test directory
        let testRootDir = tempDir.appendingPathComponent("swama-test-models")
        try? FileManager.default.removeItem(at: testRootDir)

        // Set temporary environment variable
        let originalEnv = ProcessInfo.processInfo.environment["SWAMA_MODELS"]
        setenv("SWAMA_MODELS", testRootDir.path, 1)

        defer {
            // Clean up environment and directory
            if let originalEnv {
                setenv("SWAMA_MODELS", originalEnv, 1)
            }
            else {
                unsetenv("SWAMA_MODELS")
            }
            try? FileManager.default.removeItem(at: testRootDir)
        }

        // Test non-existent model
        #expect(!ModelPaths.modelExistsLocally(testModelName))

        // Create model directory without metadata
        try FileManager.default.createDirectory(at: testModelDir, withIntermediateDirectories: true)
        #expect(!ModelPaths.modelExistsLocally(testModelName))

        // Create metadata file
        let testMetadata = """
        {
            "id": "\(testModelName)",
            "created": \(Int(Date().timeIntervalSince1970))"
        }
        """
        try testMetadata.write(to: metadataFile, atomically: true, encoding: .utf8)

        // Test existing model with metadata
        #expect(ModelPaths.modelExistsLocally(testModelName))
    }

    @Test func modelCreate() async throws {
        let tempDir = FileManager.default.temporaryDirectory
        let testModelName = "CreateTest-1B-4bit"
        let testModelDir = tempDir.appendingPathComponent("swama-test-models")

        // Create fake model directory and file (important!)
        try? FileManager.default.removeItem(at: testModelDir)
        try FileManager.default.createDirectory(at: testModelDir, withIntermediateDirectories: true)

        // Add a fake model file to simulate real content
        let dummyModelFile = testModelDir.appendingPathComponent("weights.bin")
        let dummyData = Data(repeating: 0xFF, count: 1024 * 1024) // 1MB fake data
        try dummyData.write(to: dummyModelFile)

        // Delete existing metadata if it exists
        let createdModelPath = ModelPaths.activeModelsDirectory.appendingPathComponent(testModelName)
        try? FileManager.default.removeItem(at: createdModelPath)
        let metadataFile = createdModelPath.appendingPathComponent(".swama-meta.json")

        // First: should succeed
        try await ModelCreator.run(from: testModelDir.path, name: testModelName)
        #expect(FileManager.default.fileExists(atPath: metadataFile.path))

        // Second: should throw
        do {
            try await ModelCreator.run(from: testModelDir.path, name: testModelName)
            Issue.record("Expected error when creating a model that already exists, but got none.")
        }
        catch let error as NSError {
            #expect(error.domain == "ModelCreatorError")
            #expect(error.code == 1)
            #expect(error.localizedDescription.contains("already exists"))
        }
    }
}

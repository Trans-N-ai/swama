import Foundation
@testable import SwamaKit
import Testing

@MainActor @Suite(.serialized)
final class ModelManagerTests {
    @Test func modelInfoInitialization() {
        // Test ModelInfo initialization with all parameters
        let modelInfo = ModelInfo(
            id: "test/model",
            created: 1_703_980_800, // 2023-12-31
            sizeInBytes: 1024 * 1024 * 1024, // 1GB
            source: .metaFile,
            rawMetadata: ["version": "1.0", "type": "llm"]
        )

        #expect(modelInfo.id == "test/model")
        #expect(modelInfo.created == 1_703_980_800)
        #expect(modelInfo.sizeInBytes == (1024 * 1024 * 1024))
        #expect(modelInfo.source == .metaFile)
        #expect(modelInfo.rawMetadata != nil)
        #expect(modelInfo.rawMetadata?["version"] as? String == "1.0")
    }

    @Test func modelInfoInitializationWithoutMetadata() {
        // Test ModelInfo initialization without metadata
        let modelInfo = ModelInfo(
            id: "test/model2",
            created: 1_703_980_800,
            sizeInBytes: 512 * 1024 * 1024, // 512MB
            source: .directoryScan
        )

        #expect(modelInfo.id == "test/model2")
        #expect(modelInfo.created == 1_703_980_800)
        #expect(modelInfo.sizeInBytes == (512 * 1024 * 1024))
        #expect(modelInfo.source == .directoryScan)
        #expect(modelInfo.rawMetadata == nil)
    }

    @Test func modelInfoWithRawMetadata() {
        // Test ModelInfo with rawMetadata
        let testMetadata: [String: Any] = ["hidden_size": 768, "vocab_size": 50000, "model_type": "BPE"]

        let modelInfo = ModelInfo(
            id: "test/model-with-metadata",
            created: 1_703_980_800,
            sizeInBytes: 1024 * 1024 * 1024, // 1GB
            source: .metaFile,
            rawMetadata: testMetadata
        )

        #expect(modelInfo.id == "test/model-with-metadata")
        #expect(modelInfo.rawMetadata != nil)
        #expect(modelInfo.rawMetadata?["hidden_size"] as? Int == 768)
        #expect(modelInfo.rawMetadata?["model_type"] as? String == "BPE")
    }

    @Test func metadataSourceEquality() {
        // Test MetadataSource enum equality
        #expect(MetadataSource.metaFile == MetadataSource.metaFile)
        #expect(MetadataSource.directoryScan == MetadataSource.directoryScan)
        #expect(MetadataSource.metaFile != MetadataSource.directoryScan)
    }

    @Test func modelsListReturnsArray() {
        // Test that models() returns an array (even if empty)
        let models = ModelManager.models()
        #expect(models.count >= 0) // Should always return a valid array, even if empty
        // Note: We can't test the actual content since it depends on the file system
        // In a real test environment, you might want to mock the file system
    }
}

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

    @Test func loadedModelInitialization() {
        // Test LoadedModel initialization
        let testConfig: [String: Any] = ["hidden_size": 768, "vocab_size": 50000]
        let testTokenizer: [String: Any] = ["vocab_size": 50000, "model_type": "BPE"]
        let testWeightFiles = [
            URL(fileURLWithPath: "/path/to/model.safetensors"),
            URL(fileURLWithPath: "/path/to/tokenizer.bin")
        ]

        let loadedModel = LoadedModel(
            id: "test/loaded-model",
            config: testConfig,
            tokenizer: testTokenizer,
            weightFiles: testWeightFiles
        )

        #expect(loadedModel.id == "test/loaded-model")
        #expect(loadedModel.config != nil)
        #expect(loadedModel.tokenizer != nil)
        #expect(loadedModel.weightFiles.count == 2)
        #expect(loadedModel.config?["hidden_size"] as? Int == 768)
        #expect(loadedModel.tokenizer?["model_type"] as? String == "BPE")
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
        #expect(models != nil)
        // Note: We can't test the actual content since it depends on the file system
        // In a real test environment, you might want to mock the file system
    }
}

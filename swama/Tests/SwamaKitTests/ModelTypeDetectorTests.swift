import Foundation
@testable import SwamaKit
import Testing

@Suite("Model Type Detection")
struct ModelTypeDetectorTests {
    @Test func detectVLMByOmniName() {
        #expect(ModelTypeDetector.isVLMModelName("mlx-community/Qwen3.5-Omni-7B-4bit"))
    }

    @Test func detectVLMByConfigWithoutVLSuffix() {
        let config: [String: Any] = [
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "image_token_id": 248_056,
            "vision_config": [
                "hidden_size": 1152
            ]
        ]

        #expect(ModelTypeDetector.isVLMModelConfig(config))
    }

    @Test func doNotDetectTextOnlyConfigAsVLM() {
        let config: [String: Any] = [
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 4096
        ]

        #expect(!ModelTypeDetector.isVLMModelConfig(config))
    }

    @Test func resolveQwen35Aliases() {
        #expect(ModelAliasResolver.resolve(name: "qwen3.5") == "mlx-community/Qwen3.5-35B-A3B-4bit")
        #expect(ModelAliasResolver.resolve(name: "qwen3.5-0.8b") == "mlx-community/Qwen3.5-0.8B-4bit")
        #expect(ModelAliasResolver.resolve(name: "qwen3.5-2b") == "mlx-community/Qwen3.5-2B-4bit")
        #expect(ModelAliasResolver.resolve(name: "qwen3.5-4b") == "mlx-community/Qwen3.5-4B-4bit")
        #expect(ModelAliasResolver.resolve(name: "qwen3.5-9b") == "mlx-community/Qwen3.5-9B-4bit")
        #expect(ModelAliasResolver.resolve(name: "qwen3.5-27b") == "mlx-community/Qwen3.5-27B-4bit")
        #expect(ModelAliasResolver.resolve(name: "qwen3.5-35b-a3b") == "mlx-community/Qwen3.5-35B-A3B-4bit")
        #expect(ModelAliasResolver.resolve(name: "qwen3.5-122b-a10b") == "mlx-community/Qwen3.5-122B-A10B-4bit")
        #expect(ModelAliasResolver.resolve(name: "qwen3.5-397b-a17b") == "mlx-community/Qwen3.5-397B-A17B-4bit")
    }
}

import Foundation
import MLXLMCommon
@testable import SwamaKit
import Testing
import Tokenizers

@Suite(
    "Local tokenizer bridge",
    .enabled(if: ProcessInfo.processInfo.environment["SWAMA_TEST_DECISIONS_MODEL"] == "1")
)
struct LocalTokenizerLoaderTests {
    @Test func realQwenTokenizerPreservesTemplateAndTokenSemantics() async throws {
        let directory = FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent(".swama/models/mlx-community/Qwen3.5-0.8B-MLX-4bit")
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        let bridge = try await LocalTokenizerLoader().load(from: directory)
        for text in ["中文工单: invoice 42", "<|im_start|>user\nYes/no!"] {
            for includeSpecial in [false, true] {
                let tokens = upstream.encode(text: text, addSpecialTokens: includeSpecial)
                #expect(bridge.encode(text: text, addSpecialTokens: includeSpecial) == tokens)
                for skipSpecial in [false, true] {
                    #expect(bridge.decode(tokenIds: tokens, skipSpecialTokens: skipSpecial) ==
                        upstream.decode(tokens: tokens, skipSpecialTokens: skipSpecial)
                    )
                }
            }
        }
        #expect(bridge.bosToken == upstream.bosToken)
        #expect(bridge.eosToken == upstream.eosToken)
        #expect(bridge.unknownToken == upstream.unknownToken)
        for token in ["yes", "Yes", "A", upstream.eosToken ?? ""] {
            #expect(bridge.convertTokenToId(token) == upstream.convertTokenToId(token))
            if let id = upstream.convertTokenToId(token) {
                #expect(bridge.convertIdToToken(id) == upstream.convertIdToToken(id))
            }
        }
        let messages: [[String: any Sendable]] = [
            ["role": "system", "content": "You handle support tickets."],
            ["role": "user", "content": "Choose A or B."]
        ]
        let tools: [[String: any Sendable]] = [
            ["type": "function", "function": ["name": "route", "description": "Route a ticket"]]
        ]
        let context: [String: any Sendable] = ["enable_thinking": false]
        #expect(try bridge.applyChatTemplate(messages: messages, tools: tools, additionalContext: context) ==
            upstream.applyChatTemplate(messages: messages, tools: tools, additionalContext: context)
        )
    }

    @Test func missingTemplateKeepsTheMLXErrorContract() async throws {
        let source = FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent(".swama/models/mlx-community/Qwen3.5-0.8B-MLX-4bit")
        let directory = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-tokenizer-bridge-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.copyItem(
            at: source.appendingPathComponent("tokenizer.json"),
            to: directory.appendingPathComponent("tokenizer.json")
        )
        var config = try #require(JSONSerialization.jsonObject(
            with: Data(contentsOf: source.appendingPathComponent("tokenizer_config.json"))
        ) as? [String: Any])
        config.removeValue(forKey: "chat_template")
        try JSONSerialization.data(withJSONObject: config).write(
            to: directory.appendingPathComponent("tokenizer_config.json")
        )
        let bridge = try await LocalTokenizerLoader().load(from: directory)
        do {
            _ = try bridge.applyChatTemplate(
                messages: [["role": "user", "content": "Hello"]], tools: nil, additionalContext: nil
            )
            Issue.record("a tokenizer without a template must report the MLX error")
        }
        catch MLXLMCommon.TokenizerError.missingChatTemplate {}
    }
}

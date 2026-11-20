import Foundation
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon
import MLXNN
@testable import SwamaKit
@testable import SwamaRuntime
import Testing

// MARK: - GenerationEventCompatibilityTests

@Suite("Generation event compatibility", .serialized)
struct GenerationEventCompatibilityTests {
    @Test(arguments: [false, true])
    func reasoningHasBoundariesInResultsAndStreaming(cacheEnabled: Bool) async throws {
        try await withCacheSetting(cacheEnabled) {
            let text = "<think>Plan first.</think>Answer."
            for runtime in [false, true] {
                let result = try await run(text, runtime: runtime, reasoning: true)
                #expect(result.output == text)
                #expect(result.raw == text)
                #expect(result.stream == text)
                #expect(result.tokens > 0)
            }
        }
    }

    @Test(arguments: [false, true])
    func reasoningBudgetEndsWithAClosedBoundary(cacheEnabled: Bool) async throws {
        try await withCacheSetting(cacheEnabled) {
            for runtime in [false, true] {
                let result = try await run(
                    "<think>Still thinking.",
                    runtime: runtime,
                    reasoning: true,
                    lengthStop: true
                )
                #expect(result.output == "<think>Still thinking.</think>")
                #expect(result.stream == result.output)
                #expect(result.stop == .length)
            }
        }
    }

    @Test(arguments: [false, true])
    func incompleteToolOutputPreservesTextAndUsage(cacheEnabled: Bool) async throws {
        try await withCacheSetting(cacheEnabled) {
            for runtime in [false, true] {
                let result = try await run(
                    "<tool_call>{\"name\":\"weather\",\"arguments\":",
                    runtime: runtime,
                    lengthStop: true
                )
                #expect(result.tools.isEmpty)
                #expect(result.output.contains("weather"))
                #expect(result.output.contains("arguments"))
                #expect(result.stream == result.output)
                #expect(result.stop == .length)
                #expect(result.tokens > 0)
            }
        }
    }

    @Test(arguments: [false, true])
    func undeclaredToolIsTextAndNeverAnExecutableCall(cacheEnabled: Bool) async throws {
        try await withCacheSetting(cacheEnabled) {
            for runtime in [false, true] {
                let result = try await run(
                    "<tool_call>{\"name\":\"other\",\"arguments\":{}}</tool_call>",
                    runtime: runtime
                )
                #expect(result.tools.isEmpty)
                #expect(result.output.contains("other"))
                #expect(result.stream == result.output)
                #expect(result.tokens > 0)
            }
        }
    }

    @Test(arguments: [false, true])
    func invalidToolArgumentsPreserveTextWithoutExecuting(cacheEnabled: Bool) async throws {
        try await withCacheSetting(cacheEnabled) {
            for runtime in [false, true] {
                let result = try await run(
                    "<tool_call>{\"name\":\"weather\",\"arguments\":{}}</tool_call>",
                    runtime: runtime, strictValidation: true
                )
                #expect(result.tools.isEmpty)
                #expect(result.toolCallbacks.isEmpty)
                #expect(result.output.contains("weather"))
                #expect(result.stream == result.output)
                #expect(result.tokens > 0)
            }
        }
    }

    @Test(arguments: [false, true])
    func declaredToolKeepsItsStructuredCallback(cacheEnabled: Bool) async throws {
        try await withCacheSetting(cacheEnabled) {
            for runtime in [false, true] {
                let result = try await run(
                    "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Tokyo\"}}</tool_call>",
                    runtime: runtime
                )
                #expect(result.tools == ["weather"])
                #expect(result.toolCallbacks == ["weather"])
                #expect(result.output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                #expect(result.tokens > 0)
            }
        }
    }

    private func withCacheSetting(_ enabled: Bool, operation: () async throws -> Void) async throws {
        let prior = ProcessInfo.processInfo.environment["SWAMA_PROMPT_CACHE"]
        setenv("SWAMA_PROMPT_CACHE", enabled ? "1" : "0", 1)
        defer {
            if let prior {
                setenv("SWAMA_PROMPT_CACHE", prior, 1)
            }
            else {
                unsetenv("SWAMA_PROMPT_CACHE")
            }
        }
        try await operation()
    }

    private func run(
        _ text: String,
        runtime: Bool,
        reasoning: Bool = false,
        lengthStop: Bool = false, strictValidation: Bool = false
    ) async throws -> EventSnapshot {
        if let path = ProcessInfo.processInfo.environment["SWAMA_TEST_METALLIB"] {
            try #require(FileManager.default.fileExists(atPath: path))
            GPU.metallib = URL(fileURLWithPath: path)
        }
        let tokenizer = EventByteTokenizer()
        let config = ModelConfiguration(
            id: "event-script-\(UUID().uuidString)", toolCallFormat: .json,
            reasoningConfig: reasoning ? .init(
                startDelimiter: "<think>",
                endDelimiter: "</think>",
                promptStrategy: .alwaysOn
            ) : nil
        )
        let container = ModelContainer(context: .init(
            configuration: config, model: EventScriptModel(text: text),
            processor: EventInputProcessor(), tokenizer: tokenizer
        ))
        let schema: [String: any Sendable] = [
            "type": "object", "properties": ["city": ["type": "string"]], "required": ["city"]
        ]
        let function: [String: any Sendable] = ["name": "weather", "parameters": schema]
        let tools: [[String: any Sendable]] = [["type": "function", "function": function]]
        let input = UserInput(chat: [.user("test")], tools: tools)
        var parameters = GenerateParameters(maxTokens: text.utf8.count + (lengthStop ? 0 : 2), temperature: 0)
        if strictValidation {
            parameters.toolCallPolicy.validation = .strict
        }
        let events = EventCallbacks()
        if runtime {
            let value = try await SwamaRuntime.ModelRunner(container: container, promptCacheStore: .init()).runChat(
                userInput: input, parameters: parameters,
                onToken: { events.text($0) }, onToolCall: { events.tool($0.function.name) }
            )
            return EventSnapshot(
                output: value.output,
                raw: value.rawText,
                stream: events.output,
                tools: value.toolCalls.map(\.function.name),
                toolCallbacks: events.names,
                tokens: value.completionInfo?.generationTokenCount ?? 0,
                stop: value.completionInfo?.stopReason
            )
        }
        let value = try await SwamaKit.ModelRunner(container: container, promptCacheStore: .init()).runChat(
            userInput: input, parameters: parameters,
            onToken: { events.text($0) }, onToolCall: { events.tool($0.function.name) }
        )
        return EventSnapshot(
            output: value.output,
            raw: value.rawText,
            stream: events.output,
            tools: value.toolCalls.map(\.function.name),
            toolCallbacks: events.names,
            tokens: value.completionInfo?.generationTokenCount ?? 0,
            stop: value.completionInfo?.stopReason
        )
    }
}

// MARK: - EventSnapshot

private struct EventSnapshot {
    let output: String
    let raw: String
    let stream: String
    let tools: [String]
    let toolCallbacks: [String]
    let tokens: Int
    let stop: GenerateStopReason?
}

// MARK: - EventCallbacks

private final class EventCallbacks: @unchecked Sendable {
    private let lock: NSLock = .init()
    private var pieces = ""
    private var calls: [String] = .init()
    func text(_ text: String) { lock.withLock { pieces += text } }
    func tool(_ name: String) { lock.withLock { calls.append(name) } }
    var output: String { lock.withLock { pieces } }
    var names: [String] { lock.withLock { calls } }
}

// MARK: - EventByteTokenizer

private struct EventByteTokenizer: MLXLMCommon.Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens _: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        String(decoding: tokenIds.filter { $0 > 0 && $0 < 128 }.map(UInt8.init), as: UTF8.self)
    }

    func convertTokenToId(_ token: String) -> Int? { token == "<eos>" ? 0 : nil }
    func convertIdToToken(_ id: Int) -> String? { id == 0 ? "<eos>" : decode(tokenIds: [id], skipSpecialTokens: false) }
    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] { [255] }
}

// MARK: - EventInputProcessor

private struct EventInputProcessor: UserInputProcessor {
    func prepare(input _: UserInput) throws -> LMInput { LMInput(tokens: MLXArray([255])) }
}

// MARK: - EventScriptModel

private final class EventScriptModel: Module, LLMModel, KVCacheDimensionProvider {
    let script: [Int]
    var kvHeads: [Int] { [1] }
    var loraLayers: [Module] { [] }
    init(text: String) { script = text.utf8.map(Int.init); super.init() }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let count = inputs.dim(1)
        let previous = cache?.first?.offset ?? 0
        let keys = MLXArray.zeros([1, 1, count, 1])
        for layer in cache ?? [] {
            _ = layer.update(keys: keys, values: keys)
        }
        var logits = [Float](repeating: -100, count: count * 256)
        for row in 0 ..< count {
            let index = previous + row
            let next = index < script.count ? script[index] : 0
            logits[row * 256 + next] = 100
        }
        return MLXArray(logits, [1, count, 256])
    }
}

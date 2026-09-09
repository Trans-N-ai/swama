@preconcurrency import MLXLMCommon
import NIOCore
@testable import SwamaServer
import Testing
import struct Tokenizers.ToolSpec

@Suite("Legacy completion source compatibility")
struct LegacyCompletionsCompatibilityTests {
    @Test func publicHelperSignaturesRemainCallable() {
        let nonStream: (
            Channel,
            String,
            [MLXLMCommon.Chat.Message],
            String,
            GenerateParameters,
            [ToolSpec]?
        ) async throws -> Void = CompletionsHandler.sendNonStreamResponse
        let stream: (
            Channel,
            String,
            [MLXLMCommon.Chat.Message],
            String,
            GenerateParameters,
            [ToolSpec]?
        ) async throws -> Void = CompletionsHandler.sendStreamResponse

        _ = nonStream
        _ = stream
    }
}

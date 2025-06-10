import Foundation
@testable import SwamaKit
import Testing

@MainActor @Suite(.serialized)
final class EmbeddingRunnerTests {
    @Test func embeddingGeneration() async throws {
        // TODO: Add tests for embedding generation
        // This would require a test model or mock
    }

    @Test func embeddingInputParsing() throws {
        // Test EmbeddingInput parsing
        let stringInput = EmbeddingInput.string("Hello world")
        #expect(stringInput.asStringArray == ["Hello world"])

        let arrayInput = EmbeddingInput.strings(["Hello", "World"])
        #expect(arrayInput.asStringArray == ["Hello", "World"])
    }

    @Test func embeddingError() {
        let error = EmbeddingError.noEmbeddingGenerated
        #expect(error.errorDescription == "No embedding was generated")
    }
}

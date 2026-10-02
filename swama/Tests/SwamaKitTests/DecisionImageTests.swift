import CoreImage
import Foundation
import MLX
import MLXLMCommon
@testable import SwamaCore
@testable import SwamaKit
import Testing

// MARK: - DecisionImageRequestTests

/// Task #81: images in decisions. The request shape and the engine's refusals, without a model.
@Suite("Decision images: request and validation")
struct DecisionImageRequestTests {
    private let questions: [DecisionQuestion] = [.yesNo(id: "q", question: "Is it red?")]

    @Test func requestsWithoutImagesAreUnchanged() {
        let plain = DecisionRequest(model: .init("org/model"), input: "text", questions: questions)
        #expect(plain.images.isEmpty)
        #expect(plain.imageMaxDimension == nil)
        #expect(plain == DecisionRequest(
            model: .init("org/model"), input: "text", questions: questions, images: [], imageMaxDimension: nil
        ))
    }

    @Test func engineRefusesBadImageRequestsBeforeTheBackend() async throws {
        let backend = ImageRecordingBackend()
        let engine = SwamaEngine(backend: backend)
        let png = try decisionTestPNG(red: 1, green: 0, blue: 0)
        let tooMany = Array(repeating: DecisionImage(data: png, mediaType: "image/png"), count: 5)
        let cases: [(DecisionRequest, SwamaError.Code)] = [
            (.init(model: .init("org/model"), input: "text", questions: questions, images: tooMany), .invalidRequest),
            (.init(
                model: .init("org/model"), input: "text", questions: questions,
                images: [.init(data: Data(), mediaType: "image/png")]
            ), .invalidImage),
            (.init(
                model: .init("org/model"), input: "text", questions: questions,
                images: [.init(data: png, mediaType: "image/png")], imageMaxDimension: 8
            ), .invalidRequest)
        ]
        for (request, code) in cases {
            do {
                _ = try await engine.decide(request)
                Issue.record("A bad image request reached the backend")
            }
            catch let error as SwamaError {
                #expect(error.code == code)
            }
        }
        #expect(await backend.calls == 0)
    }

    @Test func imagesMayStandWithoutTextAndReachTheBackendInOrder() async throws {
        let backend = ImageRecordingBackend()
        let engine = SwamaEngine(backend: backend)
        let red = try decisionTestPNG(red: 1, green: 0, blue: 0)
        let blue = try decisionTestPNG(red: 0, green: 0, blue: 1)
        let images = [
            DecisionImage(data: red, mediaType: "image/png"),
            DecisionImage(data: blue, mediaType: "image/png")
        ]
        _ = try await engine.decide(.init(model: .init("org/model"), input: "", questions: questions, images: images))
        #expect(await backend.lastImages == images)
        // Without images a blank input keeps today's refusal.
        await #expect(throws: SwamaError.self) {
            _ = try await engine.decide(.init(model: .init("org/model"), input: "", questions: questions))
        }
    }

    @Test func chatInputPlacesImagesBeforeTheTextInOrder() throws {
        let red = try decisionTestPNG(red: 1, green: 0, blue: 0)
        let blue = try decisionTestPNG(red: 0, green: 0, blue: 1)
        let input = try decisionImageUserInput(content: "Which colour?", images: [red, blue], processing: .init())
        guard case let .chat(messages) = input.prompt else {
            Issue.record("Expected a chat prompt")
            return
        }

        #expect(messages.count == 1)
        #expect(messages[0].role == .user)
        #expect(messages[0].content == "Which colour?")
        #expect(messages[0].images.count == 2)
        #expect(input.additionalContext?["enable_thinking"] as? Bool == false)
        #expect(throws: DecisionScoringError.self) {
            _ = try decisionImageUserInput(content: "x", images: [Data("not an image".utf8)], processing: .init())
        }
    }
}

// MARK: - DecisionImageModelTests

/// Task #81 with real models (`SWAMA_TEST_DECISIONS_MODEL=1`): the scored prompt is the chat prompt,
/// prepared prompts are never reused for images, and text-only models refuse images.
@Suite(
    "Decision images with a model",
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["SWAMA_TEST_DECISIONS_MODEL"] == "1")
)
struct DecisionImageModelTests {
    private let visionModel = "mlx-community/Qwen3.5-0.8B-MLX-4bit"
    private let textModel = "mlx-community/SmolLM-135M-Instruct-4bit"
    private let content = "Is the following true? The image is red.\nAnswer with yes or no only."
    private let labels = ["yes", "no"]
    private let pool: ModelPool = .init()

    /// M3: the token ids a decision scores are exactly what the chat path prepares for the same message.
    @Test func scoredPromptIsTheChatPathPrompt() async throws {
        let red = try decisionTestPNG(red: 1, green: 0, blue: 0)
        let processing = MLXLMCommon.UserInput.Processing(resize: .init(width: 448, height: 448))
        let scored = try await score(visionModel, images: [red], processing: processing)
        let container = try #require(await pool.loadedContainer(modelName: visionModel))
        let chatInput = try decisionImageUserInput(content: content, images: [red], processing: processing)
        let chatIDs = try await container.prepare(input: chatInput).text.tokens.flattened().asArray(Int.self)
        #expect(scored.promptTokenIDs == chatIDs)

        let text = try await score(visionModel, images: [], processing: processing)
        #expect(scored.promptTokenIDs.count > text.promptTokenIDs.count)
        #expect(scored.labelTokenIDs == text.labelTokenIDs)
        await pool.clearCache()
    }

    /// M5: two different images in swapped order give different probabilities.
    @Test func imageOrderReachesTheModel() async throws {
        let red = try decisionTestPNG(red: 1, green: 0, blue: 0)
        let blue = try decisionTestPNG(red: 0, green: 0, blue: 1)
        let ab = try await score(visionModel, images: [red, blue])
        let ba = try await score(visionModel, images: [blue, red])
        #expect(ab.promptTokenIDs.count == ba.promptTokenIDs.count)
        #expect(ab.labelLogProbs != ba.labelLogProbs)
        await pool.clearCache()
    }

    /// M6: a text-only prepared prompt is never used for a request with images.
    @Test func preparedTextPromptIsNotReusedForImages() async throws {
        let red = try decisionTestPNG(red: 1, green: 0, blue: 0)
        let text = try await score(visionModel, images: [])
        let container = try #require(await pool.loadedContainer(modelName: visionModel))
        let prepared = try await prepareDecisionPrompt(container: container, content: content, contextLimit: 8192)
        let withImage = try await score(visionModel, images: [red], prepared: prepared)
        #expect(prepared.promptIDs == text.promptTokenIDs)
        #expect(withImage.promptTokenIDs != prepared.promptIDs)
        #expect(withImage.promptTokenIDs.count > prepared.promptIDs.count)
        await pool.clearCache()
    }

    /// M2: capability comes from the loaded container; a text-only model refuses images.
    @Test func textOnlyModelRefusesImages() async throws {
        let red = try decisionTestPNG(red: 1, green: 0, blue: 0)
        do {
            _ = try await score(textModel, images: [red])
            Issue.record("A text-only model accepted an image")
        }
        catch let error as DecisionScoringError {
            guard case .imagesNotSupported = error else {
                Issue.record("Unexpected error \(error)")
                return
            }
        }
        _ = try await score(textModel, images: [])
        await pool.clearCache()
    }

    private func score(
        _ model: String,
        images: [Data],
        processing: MLXLMCommon.UserInput.Processing = .init(resize: .init(width: 448, height: 448)),
        prepared: PreparedDecisionPrompt? = nil
    ) async throws -> DecisionLogits {
        let content = content
        let labels = labels
        return try await pool.run(modelName: model) { runner in
            try await runner.scoreDecision(
                content: content, labels: labels, contextLimit: 8192, prepared: prepared,
                images: images, imageProcessing: processing
            )
        }
    }
}

// MARK: - ImageRecordingBackend

private actor ImageRecordingBackend: SwamaEngineBackend {
    private(set) var calls = 0
    private(set) var lastImages: [DecisionImage] = []
    func decide(_ request: DecisionRequest) async throws -> DecisionResponse {
        calls += 1
        lastImages = request.images
        return .init(model: request.model, answers: [:], usage: .init(promptTokens: 0, completionTokens: 0))
    }

    func generate(
        _: GenerationRequest,
        onEvent _: (@Sendable (GenerationEvent) async throws -> Void)?
    ) async throws -> GenerationResponse {
        throw SwamaError(code: .backendFailure, message: "Unexpected generation")
    }

    func embed(_: EmbeddingRequest) async throws -> SwamaCore.EmbeddingResponse {
        throw SwamaError(code: .backendFailure, message: "Unexpected embedding")
    }

    func models() async throws -> [SwamaCore.ModelInfo] { [] }
    func fetch(_ model: ModelID) async throws -> ModelID { model }
    func remove(_: ModelID) async throws {}
    func clearCache(for _: ModelID) async {}
    func clearCache() async {}
}

/// A 64×64 solid-colour PNG.
func decisionTestPNG(red: CGFloat, green: CGFloat, blue: CGFloat) throws -> Data {
    let image = CIImage(color: CIColor(red: red, green: green, blue: blue)).cropped(to: CGRect(
        x: 0,
        y: 0,
        width: 64,
        height: 64
    ))
    let context = CIContext()
    guard let data = context.pngRepresentation(
        of: image, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!
    )
    else {
        throw DecisionScoringError.invalidImage
    }

    return data
}

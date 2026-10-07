import CoreImage
import Foundation
import MLX
import MLXLMCommon
@testable import SwamaCore
@testable import SwamaKit
@testable import SwamaRuntime
@testable import SwamaServer
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
        let input = try SwamaKit.decisionImageUserInput(
            content: "Which colour?",
            images: [red, blue],
            processing: .init()
        )
        guard case let .chat(messages) = input.prompt else {
            Issue.record("Expected a chat prompt")
            return
        }

        #expect(messages.count == 1)
        #expect(messages[0].role == .user)
        #expect(messages[0].content == "Which colour?")
        #expect(messages[0].images.count == 2)
        #expect(input.additionalContext?["enable_thinking"] as? Bool == false)
        #expect(throws: SwamaKit.DecisionScoringError.self) {
            _ = try SwamaKit.decisionImageUserInput(
                content: "x",
                images: [Data("not an image".utf8)],
                processing: .init()
            )
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
    private let pool: SwamaKit.ModelPool = .init()

    /// M3: the token ids a decision scores are what the real HTTP chat path (`LegacyServerCoreBackend.makeUserInput`)
    /// prepares for the same user message, under the same resize policy. A decision always turns thinking off;
    /// the chat input below gets the same switch.
    @Test func scoredPromptIsTheKitChatPathPrompt() async throws {
        let red = try decisionTestPNG(red: 1, green: 0, blue: 0)
        let backend = LegacyServerCoreBackend(modelPool: pool)
        let message = SwamaCore.Message(role: .user, content: [.text(content), .imageData(red, mediaType: "image/png")])
        let defaultInput = try backend.makeUserInput([message], tools: [], modelName: visionModel)
        var chatInput = try backend.makeUserInput([message], tools: [], modelName: visionModel)
        chatInput.additionalContext = ["enable_thinking": false]
        // The chat path's own policy for Qwen3.5 with media: resize to 1344.
        let processing = chatInput.processing
        #expect(processing.resize == CGSize(width: 1344, height: 1344))
        let scored = try await score(visionModel, images: [red], processing: processing)
        let container = try #require(await pool.loadedContainer(modelName: visionModel))
        let chatDefault = try await container.prepare(input: defaultInput)
            .text
            .tokens
            .flattened()
            .asArray(Int.self)
        let chatIDs = try await container.prepare(input: chatInput).text.tokens.flattened().asArray(Int.self)
        #expect(scored.promptTokenIDs == chatIDs)
        // For this model (0.8B) the chat path's default render already equals thinking-off; on 9B it does not.
        // A decision always
        // forces it off, so both chat renders must match what the decision scored.
        #expect(chatDefault == chatIDs)

        let text = try await score(visionModel, images: [], processing: processing)
        #expect(scored.promptTokenIDs.count > text.promptTokenIDs.count)
        #expect(scored.labelTokenIDs == text.labelTokenIDs)
        await pool.clearCache()
    }

    /// M3 for the library path: Runtime's decision scorer against Runtime's own chat input builder
    /// (`RuntimeCoreEngine.makeUserInput`, model-default resize), thinking switched off as above.
    @Test func scoredPromptIsTheRuntimeChatPathPrompt() async throws {
        let red = try decisionTestPNG(red: 1, green: 0, blue: 0)
        let engine = RuntimeCoreEngine()
        var chatInput = try engine.makeUserInput(
            [.init(
                role: .user,
                content: [.text(content), .imageData(red, mediaType: "image/png")],
                toolCalls: [],
                toolCallID: nil
            )],
            tools: []
        )
        chatInput.additionalContext = ["enable_thinking": false]
        let runtimePool = SwamaRuntime.ModelPool()
        let content = content
        let labels = labels
        let processing = chatInput.processing
        let scored = try await runtimePool.run(modelName: visionModel) { runner in
            try await runner.scoreDecision(
                content: content, labels: labels, contextLimit: 8192, images: [red], imageProcessing: processing
            )
        }
        let container = try #require(await runtimePool.loadedContainer(modelName: visionModel))
        let chatIDs = try await container.prepare(input: chatInput).text.tokens.flattened().asArray(Int.self)
        #expect(scored.promptTokenIDs == chatIDs)
        await runtimePool.clearCache()
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
        let prepared = try await SwamaKit.prepareDecisionPrompt(
            container: container,
            content: content,
            contextLimit: 8192
        )
        let withImage = try await score(visionModel, images: [red], prepared: prepared)
        #expect(prepared.promptIDs == text.promptTokenIDs)
        #expect(withImage.promptTokenIDs != prepared.promptIDs)
        #expect(withImage.promptTokenIDs.count > prepared.promptIDs.count)
        await pool.clearCache()
    }

    /// The chat path's multimodal safety limit (4096 tokens) applies to image decisions too: four images at
    /// the Qwen3.5 default resize (about 5000 tokens) are refused, the same images at 512 px are scored.
    @Test func imageDecisionsKeepTheMultimodalContextLimit() async throws {
        let red = try decisionTestPNG(red: 1, green: 0, blue: 0)
        let four = Array(repeating: red, count: 4)
        do {
            _ = try await score(visionModel, images: four, processing: .init(resize: .init(width: 1344, height: 1344)))
            Issue.record("Four 1344 px images passed the multimodal context limit")
        }
        catch let error as SwamaKit.DecisionScoringError {
            guard case .contextLimitExceeded = error else {
                Issue.record("Unexpected error \(error)")
                return
            }
        }
        let small = try await score(
            visionModel,
            images: four,
            processing: .init(resize: .init(width: 512, height: 512))
        )
        #expect(small.promptTokenIDs.count < 4096)
        await pool.clearCache()
    }

    /// M2: capability comes from the loaded container; a text-only model refuses images.
    @Test func textOnlyModelRefusesImages() async throws {
        let red = try decisionTestPNG(red: 1, green: 0, blue: 0)
        do {
            _ = try await score(textModel, images: [red])
            Issue.record("A text-only model accepted an image")
        }
        catch let error as SwamaKit.DecisionScoringError {
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
        prepared: SwamaKit.PreparedDecisionPrompt? = nil
    ) async throws -> SwamaKit.DecisionLogits {
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
        throw SwamaKit.DecisionScoringError.invalidImage
    }

    return data
}

// MARK: - DecisionResizeBoundTests

@Suite("Decision images: resize bound")
struct DecisionResizeBoundTests {
    private func bound(requested: Int? = nil, qwen: Bool = false, sizes: [CGSize?]) -> Int? {
        LegacyServerCoreBackend.decisionResizeBound(requested: requested, appliesQwenDefault: qwen, imageSizes: sizes)
    }

    @Test func noBoundMeansNoResize() {
        #expect(bound(sizes: []) == nil)
        #expect(bound(sizes: [CGSize(width: 512, height: 341)]) == nil)
    }

    @Test func qwenDefaultIsCappedAtTheLargestImage() {
        #expect(bound(qwen: true, sizes: [CGSize(width: 512, height: 341)]) == 512)
        #expect(bound(qwen: true, sizes: [CGSize(width: 2000, height: 1000)]) == 1344)
    }

    @Test func requestedBoundIsCappedAtTheLargestImage() {
        #expect(bound(requested: 512, sizes: [CGSize(width: 300, height: 200)]) == 300)
        #expect(bound(requested: 512, sizes: [CGSize(width: 1280, height: 720)]) == 512)
    }

    @Test func unreadableSizeKeepsTheUncappedBound() {
        #expect(bound(qwen: true, sizes: [nil]) == 1344)
        #expect(bound(requested: 512, sizes: [CGSize(width: 300, height: 200), nil]) == 512)
    }

    @Test func tinyImagesAreStillEnlargedToTheMinimumShortSide() {
        // The vision processor refuses sides below its patch factor, so tiny images keep being enlarged.
        #expect(bound(qwen: true, sizes: [CGSize(width: 8, height: 8)]) == 64)
        #expect(bound(qwen: true, sizes: [CGSize(width: 16, height: 32)]) == 128)
        #expect(bound(requested: 512, sizes: [CGSize(width: 40, height: 20)]) == 128)
        // An extreme aspect ratio cannot reach the minimum within the bound; the bound wins, as before.
        #expect(bound(qwen: true, sizes: [CGSize(width: 2000, height: 10)]) == 1344)
        // With several images, the box must keep every image's short side.
        #expect(bound(qwen: true, sizes: [CGSize(width: 900, height: 600), CGSize(width: 16, height: 32)]) == 900)
        #expect(bound(qwen: true, sizes: [CGSize(width: 200, height: 200), CGSize(width: 300, height: 30)]) == 640)
    }

    @Test func largestOfSeveralImagesWins() {
        #expect(bound(requested: 1344, sizes: [CGSize(width: 400, height: 300), CGSize(width: 900, height: 600)]) ==
            900
        )
    }
}

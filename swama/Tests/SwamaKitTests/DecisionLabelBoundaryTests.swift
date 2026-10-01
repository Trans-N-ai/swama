import Foundation
@testable import SwamaKit
@testable import SwamaRuntime
import Testing

@Suite("Decision label boundaries")
struct DecisionLabelBoundaryTests {
    private let prompt = "message<B>tail"
    private let ids = [9, 50, 60]
    private var encodings: [String: [Int]] {
        ["tail": [60], "tailA": [60, 1], "tailB": [60, 2], "message<B>tailA": [9, 50, 60, 1]]
    }

    @Test func supportedBoundaryPreservesAllLabelsInBothPaths() throws {
        let encode = { encodings[$0] ?? [] }
        let kit = SwamaKit.decisionLabelContext(
            prompt: prompt,
            promptIDs: ids,
            labels: ["A", "B"],
            boundaryTokens: [50: "<B>"],
            encode: encode
        )
        let runtime = SwamaRuntime.decisionLabelContext(
            prompt: prompt,
            promptIDs: ids,
            labels: ["A", "B"],
            boundaryTokens: [50: "<B>"],
            encode: encode
        )
        #expect(kit.text == "tail")
        #expect(kit.tokenIDs == [60])
        #expect(runtime.text == kit.text)
        #expect(runtime.tokenIDs == kit.tokenIDs)
        let labels = try SwamaKit
            .decisionLabelIDs(promptIDs: kit.tokenIDs, labels: ["A", "B"]) { encode(kit.text + $0) }
        #expect(labels == [1, 2])
    }

    @Test func missingAndOverlappingBoundariesUseFullPrompt() {
        // A longer added token could swallow the marker when a label is appended.
        for tokens in [[:], [50: "<B>", 51: "prefix<B>tailB"]] {
            let kit = SwamaKit.decisionLabelContext(
                prompt: prompt,
                promptIDs: ids,
                labels: ["A", "B"],
                boundaryTokens: tokens
            ) { encodings[$0] ?? [] }
            let runtime = SwamaRuntime.decisionLabelContext(
                prompt: prompt,
                promptIDs: ids,
                labels: ["A", "B"],
                boundaryTokens: tokens
            ) { encodings[$0] ?? [] }
            #expect(kit.text == prompt && kit.tokenIDs == ids)
            #expect(runtime.text == prompt && runtime.tokenIDs == ids)
        }
    }

    @Test func mismatchedSuffixOrWholePromptCrossCheckFallsBack() {
        for broken in ["tail", "message<B>tailA"] {
            var table = encodings
            table[broken] = [999]
            let kit = SwamaKit.decisionLabelContext(
                prompt: prompt,
                promptIDs: ids,
                labels: ["A", "B"],
                boundaryTokens: [50: "<B>"]
            ) { table[$0] ?? [] }
            let runtime = SwamaRuntime.decisionLabelContext(
                prompt: prompt,
                promptIDs: ids,
                labels: ["A", "B"],
                boundaryTokens: [50: "<B>"]
            ) { table[$0] ?? [] }
            #expect(kit.text == prompt && kit.tokenIDs == ids)
            #expect(runtime.text == prompt && runtime.tokenIDs == ids)
        }
    }

    @Test func suffixOptimizationStillRejectsMergingOrDuplicateLabels() throws {
        for badB in [[61, 2], [60, 1], [60, 2, 3]] {
            #expect(throws: (any Error).self) {
                try SwamaKit.decisionLabelIDs(promptIDs: [60], labels: ["A", "B"]) { $0 == "A" ? [60, 1] : badB }
            }
            #expect(throws: (any Error).self) {
                try SwamaRuntime.decisionLabelIDs(promptIDs: [60], labels: ["A", "B"]) { $0 == "A" ? [60, 1] : badB }
            }
        }
    }

    private func metadata(
        prefix: Bool = false,
        strip: Bool = false,
        normalizer: String = "NFC",
        duplicate: Bool = false
    ) throws -> Data {
        let token: [String: Any] = ["id": 50, "content": "<B>", "lstrip": strip, "rstrip": false]
        return try JSONSerialization.data(withJSONObject: [
            "normalizer": ["type": normalizer],
            "pre_tokenizer": ["type": "Sequence",
                              "pretokenizers": [["type": "Split"], ["type": "ByteLevel", "add_prefix_space": prefix]]],
            "added_tokens": duplicate ? [token, token] : [token],
        ])
    }

    @Test func metadataGateAcceptsSupportedLayoutAndRejectsUnsafeLayouts() throws {
        let valid = try metadata()
        #expect(SwamaKit.decisionBoundaryTokens(from: valid) == [50: "<B>"])
        #expect(SwamaRuntime.decisionBoundaryTokens(from: valid) == [50: "<B>"])
        for invalid in try [
            metadata(prefix: true),
            metadata(strip: true),
            metadata(normalizer: "Other"),
            metadata(duplicate: true),
            Data("{}".utf8),
            Data("invalid".utf8)
        ] {
            #expect(SwamaKit.decisionBoundaryTokens(from: invalid).isEmpty)
            #expect(SwamaRuntime.decisionBoundaryTokens(from: invalid).isEmpty)
        }
    }
}

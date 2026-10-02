import CoreGraphics
import CoreImage
import CoreText
import Foundation
import MLXLMCommon
@testable import SwamaKit
import Testing

// MARK: - DecisionImageProbeTests

/// Task #81 probes on a real vision model, without any HTTP interface. Opt-in only:
///   SWAMA_DECISION_IMAGE_PROBE=<model id>  SWAMA_DECISION_IMAGE_PROBE_OUT=<file.json>
/// Writes accuracy (with image, without image, swapped image), order probes and latency rows.
/// The gates live in CRITERIA-task81-decision-images.md; this test only measures and records.
@Suite(
    "Decision image probes",
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["SWAMA_DECISION_IMAGE_PROBE"] != nil)
)
struct DecisionImageProbeTests {
    private let model = ProcessInfo.processInfo.environment["SWAMA_DECISION_IMAGE_PROBE"] ?? ""
    private let outPath = ProcessInfo.processInfo
        .environment["SWAMA_DECISION_IMAGE_PROBE_OUT"] ?? "/tmp/decision-image-probe.json"
    private let pool: ModelPool = .init()

    private static let colours: [(String, CGFloat, CGFloat, CGFloat)] = [
        ("red", 0.9, 0.1, 0.1), ("green", 0.1, 0.75, 0.2), ("blue", 0.1, 0.2, 0.9),
        ("yellow", 0.95, 0.9, 0.1), ("purple", 0.55, 0.15, 0.7), ("orange", 1.0, 0.55, 0.05)
    ]

    @Test func measure() async throws {
        var report: [String: Any] = ["model": model]
        let colourNames = Self.colours.map(\.0)
        let digitNames = (0 ... 9).map { "digit_\($0)" }
        var rng = SplitMix(seed: 81)

        // Colour probes: 30, each a solid colour; options in a shuffled order per probe.
        var colourProbes: [(image: Data, target: String, options: [String])] = []
        for i in 0 ..< 30 {
            let c = Self.colours[i % Self.colours.count]
            try colourProbes.append((solidPNG(c.1, c.2, c.3), c.0, rng.shuffled(colourNames)))
        }
        // Digit probes: 30, a large black digit on white.
        var digitProbes: [(image: Data, target: String, options: [String])] = []
        for i in 0 ..< 30 {
            let d = (i * 7 + 3) % 10
            try digitProbes.append((digitPNG(d), "digit_\(d)", rng.shuffled(digitNames)))
        }

        for (name, probes, question) in [
            ("colour", colourProbes, "Which colour fills the image?"),
            ("digit", digitProbes, "Which digit is shown in the image?")
        ] {
            var withImage = 0, withoutImage = 0, swapped = 0
            var ms: [Double] = []
            for (k, probe) in probes.enumerated() {
                let content = choicePrompt(question: question, options: probe.options)
                let (hit, t) = try await pick(content: content, options: probe.options, images: [probe.image])
                ms.append(t)
                withImage += hit == probe.target ? 1 : 0
                let (blind, _) = try await pick(content: content, options: probe.options, images: [])
                withoutImage += blind == probe.target ? 1 : 0
                // Swapped control: another probe's image whose answer differs.
                let other = probes[(k + 1) % probes.count]
                let (wrong, _) = try await pick(content: content, options: probe.options, images: [other.image])
                swapped += (wrong == probe.target && other.target != probe.target) ? 1 : 0
            }
            let count = Double(probes.count)
            var row: [String: Any] = [:]
            row["probes"] = probes.count
            row["with_image"] = Double(withImage) / count
            row["without_image"] = Double(withoutImage) / count
            row["swapped_image"] = Double(swapped) / count
            row["median_ms"] = median(ms)
            report[name] = row
        }

        // Order probes: two different colours; ask for the first or the second image.
        var orderHits = 0, orderTotal = 0
        for i in 0 ..< 20 {
            let a = Self.colours[i % 6], b = Self.colours[(i + 2) % 6]
            let which = i % 2 == 0 ? "first" : "second"
            let target = which == "first" ? a.0 : b.0
            let options = rng.shuffled(colourNames)
            let content = choicePrompt(question: "What colour is the \(which) image?", options: options)
            let (hit, _) = try await pick(
                content: content, options: options, images: [solidPNG(a.1, a.2, a.3), solidPNG(b.1, b.2, b.3)]
            )
            orderHits += hit == target ? 1 : 0
            orderTotal += 1
        }
        report["order"] = ["probes": orderTotal, "accuracy": Double(orderHits) / Double(orderTotal)]

        // M10: an image question with 30 options takes the two-letter labels from #169.
        let pairUsable = try await pool.run(modelName: model) { runner in try await runner.decisionPairLabels() }
        let pairLabels = try decisionPairLabelPrefix(count: 30, usable: pairUsable)
        let distractors = (0 ..< 20).map { "word_\($0)" }
        var pairHits = 0
        for i in 0 ..< 20 {
            let d = (i * 3 + 1) % 10
            let options = rng.shuffled(digitNames + distractors)
            let lines = zip(pairLabels, options).map { "\($0). \($1)" }.joined(separator: "\n")
            let content = "Which digit is shown in the image?\n\(lines)\nAnswer with the two-letter label only."
            let scored = try await score(
                content: content, labels: pairLabels, images: [digitPNG(d)], processing: defaultProcessing()
            )
            let best = scored.labelLogProbs.enumerated().max { $0.element < $1.element }!.offset
            pairHits += options[best] == "digit_\(d)" ? 1 : 0
        }
        report["pair_labels_30"] = ["probes": 20, "accuracy": Double(pairHits) / 20]

        // Latency and prompt tokens: text only, 1 and 4 images, default policy and 512 px.
        var latency: [[String: Any]] = []
        let photo = try digitPNG(7, size: 1024)
        for (label, images, processing) in [
            ("text", [Data](), MLXLMCommon.UserInput.Processing()),
            ("1 image default", [photo], defaultProcessing()),
            ("1 image 512", [photo], .init(resize: .init(width: 512, height: 512))),
            ("4 images default", Array(repeating: photo, count: 4), defaultProcessing())
        ] {
            var ms: [Double] = []
            var tokens = 0
            let content = choicePrompt(question: "Which digit is shown?", options: digitNames)
            _ = try await pick(content: content, options: digitNames, images: images, processing: processing)
            for _ in 0 ..< 5 {
                let start = Date()
                let scored = try await score(
                    content: content,
                    labels: letters(digitNames.count),
                    images: images,
                    processing: processing
                )
                ms.append(Date().timeIntervalSince(start) * 1000)
                tokens = scored.promptTokenIDs.count
            }
            latency.append(["condition": label, "median_ms": median(ms), "prompt_tokens": tokens])
        }
        report["latency"] = latency
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outPath))
        print(String(decoding: data, as: UTF8.self))
        await pool.clearCache()
    }

    // MARK: Helpers

    /// The HTTP path's default resize for Qwen3.5 (the chat path's policy); others use the model's own.
    private func defaultProcessing() -> MLXLMCommon.UserInput.Processing {
        model.lowercased().contains("qwen3.5") ? .init(resize: .init(width: 1344, height: 1344)) : .init()
    }

    private func letters(_ n: Int) -> [String] { (0 ..< n).map { String(UnicodeScalar(65 + $0)!) } }

    private func choicePrompt(question: String, options: [String]) -> String {
        let lines = zip(letters(options.count), options).map { "\($0). \($1)" }.joined(separator: "\n")
        return "\(question)\n\(lines)\nAnswer with the letter only."
    }

    private func pick(
        content: String, options: [String], images: [Data],
        processing: MLXLMCommon.UserInput.Processing? = nil
    ) async throws -> (String, Double) {
        let start = Date()
        let scored = try await score(
            content: content, labels: letters(options.count), images: images,
            processing: processing ?? defaultProcessing()
        )
        let ms = Date().timeIntervalSince(start) * 1000
        let best = scored.labelLogProbs.enumerated().max { $0.element < $1.element }!.offset
        return (options[best], ms)
    }

    private func score(
        content: String, labels: [String], images: [Data], processing: MLXLMCommon.UserInput.Processing
    ) async throws -> DecisionLogits {
        try await pool.run(modelName: model) { runner in
            try await runner.scoreDecision(
                content: content, labels: labels, contextLimit: 32768, images: images, imageProcessing: processing
            )
        }
    }

    private func median(_ values: [Double]) -> Double {
        let s = values.sorted()
        return s.isEmpty ? 0 : (s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2)
    }

    private func solidPNG(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat) throws -> Data {
        try decisionTestPNG(red: r, green: g, blue: b)
    }

    private func digitPNG(_ digit: Int, size: Int = 256) throws -> Data {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(
            data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0, space: space,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        else {
            throw DecisionScoringError.invalidImage
        }

        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        let font = CTFontCreateWithName("Helvetica-Bold" as CFString, CGFloat(size) * 0.8, nil)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(
                red: 0,
                green: 0,
                blue: 0,
                alpha: 1
            )
        ]
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: "\(digit)", attributes: attributes))
        let bounds = CTLineGetImageBounds(line, ctx)
        ctx.textPosition = CGPoint(
            x: (CGFloat(size) - bounds.width) / 2 - bounds.minX,
            y: (CGFloat(size) - bounds.height) / 2 - bounds.minY
        )
        CTLineDraw(line, ctx)
        guard let image = ctx.makeImage(),
              let data = CIContext().pngRepresentation(of: CIImage(cgImage: image), format: .RGBA8, colorSpace: space)
        else {
            throw DecisionScoringError.invalidImage
        }

        return data
    }
}

// MARK: - SplitMix

/// Deterministic shuffles for reproducible probes.
private struct SplitMix {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    mutating func shuffled<T>(_ values: [T]) -> [T] {
        var a = values
        for i in stride(from: a.count - 1, to: 0, by: -1) {
            let j = Int(next() % UInt64(i + 1))
            a.swapAt(i, j)
        }
        return a
    }
}

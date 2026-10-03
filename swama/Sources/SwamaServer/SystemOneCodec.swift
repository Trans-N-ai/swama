import Foundation
import ImageIO
import SwamaCore

// MARK: - SystemOneRequest

struct SystemOneRequest: Sendable {
    struct Question: Sendable {
        let id: String
        let kind: DecisionKind
        let names: [String]
        let legend: [SystemOneJSON]
    }

    let decision: DecisionRequest
    let questions: [Question]

    /// The largest request body this route reads (Cloudflare Clef's limit).
    static let maximumBodyBytes = 13 * 1024 * 1024

    static func parse(_ data: Data) throws -> SystemOneRequest {
        let root = try SystemOneJSON.parse(data)
        guard case .object = root else {
            throw invalid("The SystemOne request must be an object.")
        }

        let model = try requiredString(root.member("model"), field: "model")
        guard let state = root.member("state"), isText(state) else {
            throw invalid("state must be text, an object, or an array.")
        }
        guard case let .object(rawQuestions)? = root.member("questions"), !rawQuestions.isEmpty else {
            throw invalid("questions must be a non-empty map.")
        }

        // The published contract permits unknown top-level extension fields. SGLang
        // explicitly refuses these Decisions-only fields instead of ignoring them.
        for name in ["temperature", "prompt_format_version", "return_prompt_token_ids"] {
            if let value = root.member(name), value != .null {
                throw invalid("\(name) is not part of this API; use /v1/decisions for it.")
            }
        }
        // Clef's video extension is not served here; refusing it beats answering without it.
        if let video = root.member("video"), video != .null {
            throw invalid("video is not supported on this backend; send images instead.")
        }
        let images = try SystemOneImages.parse(root.member("images"))
        if let kwargs = root.member("chat_template_kwargs") {
            try allow(kwargs, ["enable_thinking"], field: "chat_template_kwargs")
            if let value = kwargs.member("enable_thinking"), value != .bool(false) {
                throw invalid("Decisions require enable_thinking=false.")
            }
        }

        var decisions = [DecisionQuestion](), metadata = [Question]()
        for member in rawQuestions {
            let id = member.name, raw = member.value
            try allow(raw, ["type", "instructions", "criteria"], field: "question '\(id)'")
            let kind = try requiredString(raw.member("type"), field: "question '\(id)' type")
            let rawInstructions = raw.member("instructions")
            let instructions = try optionalText(rawInstructions, field: "instructions") ?? ""
            guard !isBlank(rawInstructions) else {
                throw invalid(
                    "Non-empty instructions are required on this decision backend. " +
                        "Supported bounds: a choice needs at least two options; a score takes 2 to 10 levels."
                )
            }

            switch kind {
            case "noul":
                var yes: String?, no: String?
                if let criteria = raw.member("criteria"), criteria != .null {
                    try allow(criteria, ["true", "false"], field: "noul criteria")
                    yes = try optionalText(criteria.member("true"), field: "criteria.true")
                    no = try optionalText(criteria.member("false"), field: "criteria.false")
                }
                decisions.append(.yesNo(id: id, question: instructions, yes: yes, no: no))
                metadata.append(.init(id: id, kind: .yesNo, names: ["yes", "no"], legend: []))

            case "choice":
                guard case let .object(criteria)? = raw.member("criteria") else {
                    throw invalid("choice criteria must be a map of names to descriptions.")
                }
                guard criteria.count >= 2 else {
                    throw invalid("a choice needs at least two options on this backend.")
                }

                // More than 26 options take the model's two-letter labels, AA to ZZ.
                guard criteria.count <= 26 * 26
                else {
                    throw invalid("This backend supports at most 676 options per choice.")
                }

                let options = try criteria.map {
                    try DecisionOption(name: $0.name, description: optionalText($0.value, field: "choice description"))
                }
                decisions.append(.choice(id: id, question: instructions, options: options))
                metadata.append(.init(id: id, kind: .choice, names: options.map(\.name), legend: []))

            case "score":
                guard case let .array(criteria)? = raw.member("criteria") else {
                    throw invalid("score criteria must be an ordered array.")
                }
                guard (2 ... 10).contains(criteria.count) else {
                    throw invalid("a score takes 2 to 10 levels on this backend.")
                }

                let levels = try criteria.map { value -> String in
                    guard isText(value), !isBlank(value) else {
                        throw invalid("Score criteria entries must be non-empty text, objects, or arrays.")
                    }

                    return value.text
                }
                decisions.append(.score(id: id, question: instructions, levels: levels))
                metadata.append(.init(id: id, kind: .score, names: levels.indices.map(String.init), legend: criteria))

            default: throw invalid("Question type must be noul, choice, or score.")
            }
        }
        return .init(
            // SGLang accepts any state, including an empty one; /v1/decisions still requires input.
            // More than 26 options take the model's two-letter labels.
            decision: .init(
                model: .init(model), input: state.text, questions: decisions,
                allowsBlankInput: true, allowsPairLabels: true, images: images
            ),
            questions: metadata
        )
    }

    func response(_ result: DecisionResponse) throws -> SystemOneJSON {
        var answers = [(String, SystemOneJSON)]()
        for question in questions {
            guard let answer = result.answers[question.id], answer.type == question.kind,
                  answer.labelMass.isFinite,
                  Set(answer.probabilities.keys) == Set(question.names),
                  answer.probabilities.values.allSatisfy({ $0.isFinite && (0 ... 1).contains($0) })
            else {
                throw backendFailure("The model returned an invalid SystemOne answer.")
            }

            let probabilities = SystemOneJSON.object(question.names.map { ($0, .number(answer.probabilities[$0]!)) })
            let value: SystemOneJSON
            switch question.kind {
            case .yesNo:
                value = .object([
                    ("type", .string("noul")),
                    ("noul", .number(answer.probabilities["yes"]!)),
                    ("x_label_mass", .number(answer.labelMass))
                ])

            case .choice:
                guard let choice = answer.choice, question.names.contains(choice),
                      let confidence = answer.confidence, confidence.isFinite
                else {
                    throw backendFailure("The model returned an invalid choice answer.")
                }

                value = .object([
                    ("type", .string("choice")),
                    ("choice", .string(choice)),
                    ("confidence", .number(confidence)),
                    ("probabilities", probabilities),
                    ("x_label_mass", .number(answer.labelMass))
                ])

            case .score:
                guard let score = answer.score, score.isFinite,
                      let confidence = answer.confidence, confidence.isFinite
                else {
                    throw backendFailure("The model returned an invalid score answer.")
                }

                value = .object([
                    ("type", .string("score")),
                    ("score", .number(score)),
                    ("confidence", .number(confidence)),
                    ("legend", .object(Array(zip(question.names, question.legend)))),
                    ("probabilities", probabilities),
                    ("x_label_mass", .number(answer.labelMass))
                ])
            }
            answers.append((question.id, value))
        }
        return .object([
            ("model", .string(result.model.rawValue)),
            ("answers", .object(answers)),
            ("usage", .object([
                ("input_tokens", .integer(result.usage.promptTokens)),
                ("output_tokens", .integer(0))
            ]))
        ])
    }

    private static func allow(_ value: SystemOneJSON, _ allowed: Set<String>, field: String) throws {
        guard case let .object(members) = value else {
            throw invalid("\(field) must be an object.")
        }

        if let unknown = members.first(where: { !allowed.contains($0.name) }) {
            throw invalid("Unknown \(field) field '\(unknown.name)'.")
        }
    }

    private static func requiredString(_ value: SystemOneJSON?, field: String) throws -> String {
        guard case let .string(text)? = value, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw invalid("\(field) must be a non-empty string.")
        }

        return text
    }

    private static func optionalText(_ value: SystemOneJSON?, field: String) throws -> String? {
        guard let value, value != .null else {
            return nil
        }
        guard isText(value) else {
            throw invalid("\(field) must be text, an object, an array, or null.")
        }

        return value.text
    }

    private static func isText(_ value: SystemOneJSON) -> Bool {
        switch value {
        case .array,
             .object,
             .string: true
        default: false
        }
    }

    private static func isBlank(_ value: SystemOneJSON?) -> Bool {
        switch value {
        case nil,
             .null: true
        case let .string(text): text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case let .object(members): members.isEmpty
        case let .array(values): values.isEmpty
        default: false
        }
    }

    private static func invalid(_ message: String) -> DecisionWireError { .invalid(message) }
    private func backendFailure(_ message: String) -> SwamaError { .init(code: .backendFailure, message: message) }
}

// MARK: - SystemOneImages

/// Cloudflare Clef's System One extension: an optional `images` array placed before the state.
/// Each item is a base64 data URL string or `{"content_type", "base64"}`; remote URLs are refused.
enum SystemOneImages {
    static let maximumCount = 4
    static let maximumImageBytes = 4 * 1024 * 1024
    static let maximumTotalBytes = 8 * 1024 * 1024
    static let maximumPixels = 16_000_000
    static let mediaTypes: Set<String> = ["image/png", "image/jpeg", "image/webp"]

    static func parse(_ value: SystemOneJSON?) throws -> [DecisionImage] {
        guard let value, value != .null else {
            return []
        }
        guard case let .array(items) = value else {
            throw DecisionWireError.invalid("images must be an array.")
        }
        guard items.count <= maximumCount else {
            throw DecisionWireError.invalid("A request accepts at most \(maximumCount) images.")
        }

        var images = [DecisionImage](), total = 0
        for (index, item) in items.enumerated() {
            let (mediaType, base64) = try mediaTypeAndBase64(item, index: index)
            guard let data = Data(base64Encoded: base64), !data.isEmpty else {
                throw DecisionWireError.invalid("images[\(index)] is not valid base64.")
            }

            // Judge the format by its bytes; a declaration alone is not enough.
            guard let actual = sniffedMediaType(data) else {
                throw DecisionWireError.invalid("images[\(index)] must be PNG, JPEG, or WebP data.")
            }
            guard actual == mediaType else {
                throw DecisionWireError.invalid("images[\(index)] is declared \(mediaType) but contains \(actual).")
            }
            guard data.count <= maximumImageBytes else {
                throw DecisionWireError.invalid("images[\(index)] is larger than 4 MiB.")
            }

            total += data.count
            guard total <= maximumTotalBytes else {
                throw DecisionWireError.invalid("images exceed 8 MiB in total.")
            }
            guard let pixels = pixelCount(data) else {
                throw DecisionWireError.invalid("images[\(index)] is not a decodable image.")
            }
            guard pixels <= maximumPixels else {
                throw DecisionWireError.invalid("images[\(index)] is larger than 16 megapixels.")
            }

            images.append(.init(data: data, mediaType: mediaType))
        }
        return images
    }

    private static func mediaTypeAndBase64(_ item: SystemOneJSON, index: Int) throws -> (String, String) {
        switch item {
        case let .string(text):
            guard text.lowercased().hasPrefix("data:") else {
                throw DecisionWireError.invalid(
                    "images[\(index)] must be a base64 data URL; remote URLs are not accepted."
                )
            }
            guard let comma = text.firstIndex(of: ","),
                  let header = Optional(text[text.index(text.startIndex, offsetBy: 5) ..< comma]),
                  header.lowercased().hasSuffix(";base64")
            else {
                throw DecisionWireError.invalid("images[\(index)] must be a base64 data URL.")
            }

            let mediaType = String(header.dropLast(";base64".count)).lowercased()
            guard mediaTypes.contains(mediaType) else {
                throw DecisionWireError.invalid("images[\(index)] must be PNG, JPEG, or WebP.")
            }

            return (mediaType, String(text[text.index(after: comma)...]))

        case let .object(members):
            if let unknown = members.first(where: { !["content_type", "base64"].contains($0.name) }) {
                throw DecisionWireError.invalid("Unknown images[\(index)] field '\(unknown.name)'.")
            }
            guard case let .string(contentType)? = item.member("content_type"),
                  case let .string(base64)? = item.member("base64")
            else {
                throw DecisionWireError.invalid("images[\(index)] needs string content_type and base64 fields.")
            }

            let mediaType = contentType.lowercased()
            guard mediaTypes.contains(mediaType) else {
                throw DecisionWireError.invalid("images[\(index)] must be PNG, JPEG, or WebP.")
            }

            return (mediaType, base64)

        default:
            throw DecisionWireError.invalid("images[\(index)] must be a data URL string or an object.")
        }
    }

    /// The media type from the file signature: PNG, JPEG, or RIFF/WEBP; nil for anything else.
    static func sniffedMediaType(_ data: Data) -> String? {
        let bytes = [UInt8](data.prefix(12))
        if bytes.starts(with: [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
            return "image/png"
        }
        if bytes.starts(with: [0xFF, 0xD8, 0xFF]) {
            return "image/jpeg"
        }
        if bytes.count == 12, bytes.starts(with: Array("RIFF".utf8)), Array(bytes[8 ..< 12]) == Array("WEBP".utf8) {
            return "image/webp"
        }
        return nil
    }

    /// Width times height from the image header, without decoding pixels; nil if it is not an image.
    static func pixelCount(_ data: Data) -> Int? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              CGImageSourceGetCount(source) > 0,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0
        else {
            return nil
        }

        return width * height
    }
}

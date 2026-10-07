import Foundation
import SwamaCore

// MARK: - OpenAIDecisionRequest

/// OpenAI Decisions wire format on `/v1/decisions` (openai-openapi 4a4020d8): an ordered `questions` array of
/// `predicate`, `choice` and `score` questions with optional names, mapped onto Core's yes/no, choice and score
/// questions. Parsing is strict: unknown fields are refused, so the removed SGLang format 1 shape gets a 400.
struct OpenAIDecisionRequest: Sendable {
    /// A typed choice value. A string and a boolean with the same text are distinct on the wire.
    enum ChoiceValue: Hashable, Sendable {
        case string(String)
        case bool(Bool)

        /// The option name Core sees; booleans become "true" and "false".
        var coreName: String {
            switch self {
            case let .string(text): text
            case let .bool(flag): flag ? "true" : "false"
            }
        }

        var json: SystemOneJSON {
            switch self {
            case let .string(text): .string(text)
            case let .bool(flag): .bool(flag)
            }
        }
    }

    struct Question: Sendable {
        /// Internal Core id, `q<index>`; never sent back.
        let id: String
        let name: String?
        let kind: DecisionKind
        /// Choice values in request order.
        let values: [ChoiceValue]
        /// Score level labels in request order.
        let labels: [String]
    }

    let decision: DecisionRequest
    let questions: [Question]

    static let maximumQuestions = 200
    static let maximumChoices = 255
    static let maximumSafetyIdentifierLength = 128

    static func parse(_ body: [String: JSONValue]) throws -> OpenAIDecisionRequest {
        if usesRemovedSGLangShape(body) {
            throw invalid(
                "The SGLang prompt format 1 request shape was removed from /v1/decisions; "
                    + "send the OpenAI Decisions format (questions with type, name and instructions)."
            )
        }
        try allow(body, ["model", "input", "questions", "safety_identifier"])
        guard case let .string(model)? = body["model"],
              !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw invalid("model must be a non-empty string.")
        }

        // Accepted for compatibility and ignored: Swama has no per-user policy.
        if let value = body["safety_identifier"], value != .null {
            guard case let .string(identifier) = value,
                  identifier.unicodeScalars.count <= maximumSafetyIdentifierLength
            else {
                throw invalid("safety_identifier must be a string of at most 128 characters, or null.")
            }
        }

        let (input, images) = try parseInput(body["input"])
        guard case let .array(rawQuestions)? = body["questions"],
              (1 ... maximumQuestions).contains(rawQuestions.count)
        else {
            throw invalid("questions must be an array of 1–200 questions.")
        }

        var decisions = [DecisionQuestion](), metadata = [Question](), names = Set<String>()
        for (index, raw) in rawQuestions.enumerated() {
            let (decision, question) = try parseQuestion(raw, index: index)
            if let name = question.name {
                guard names.insert(name).inserted else {
                    throw invalid("Duplicate question name '\(name)'.")
                }
            }
            decisions.append(decision)
            metadata.append(question)
        }
        return .init(
            // Like SystemOne, a blank input is rendered as is; the questions may be about the images alone.
            // More than 26 choices take the model's two-letter labels.
            decision: .init(
                model: .init(model), input: input, questions: decisions,
                allowsBlankInput: true,
                allowsPairLabels: metadata.contains { $0.values.count > 26 },
                images: images
            ),
            questions: metadata
        )
    }

    private static func parseQuestion(_ value: JSONValue, index: Int) throws -> (DecisionQuestion, Question) {
        let field = "questions[\(index)]"
        guard case let .object(question) = value else {
            throw invalid("\(field) must be an object.")
        }
        guard case let .string(type)? = question["type"] else {
            throw invalid("\(field) type must be predicate, choice, or score.")
        }

        switch type {
        case "predicate": try allow(question, ["type", "name", "instructions"])
        case "choice": try allow(question, ["type", "name", "instructions", "choices"])
        case "score": try allow(question, ["type", "name", "instructions", "levels"])
        default: throw invalid("\(field) type must be predicate, choice, or score.")
        }

        var name: String?
        if let value = question["name"] {
            guard case let .string(text) = value else {
                throw invalid("\(field) name must be a string.")
            }

            name = text
        }
        guard case let .string(instructions)? = question["instructions"],
              !instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw invalid("\(field) instructions must be a non-empty string.")
        }

        let id = "q\(index)"
        switch type {
        case "choice":
            guard case let .array(rawChoices)? = question["choices"],
                  (2 ... maximumChoices).contains(rawChoices.count)
            else {
                throw invalid("\(field) choices must contain 2–255 entries.")
            }

            var values = [ChoiceValue](), options = [DecisionOption]()
            for raw in rawChoices {
                guard case let .object(choice) = raw else {
                    throw invalid("Each choice must be an object.")
                }

                try allow(choice, ["value", "description"])
                let value: ChoiceValue =
                    switch choice["value"] {
                    case let .string(text)?: .string(text)
                    case let .bool(flag)?: .bool(flag)
                    default: throw invalid("\(field) choice value must be a string or a boolean.")
                    }
                guard !values.contains(value) else {
                    throw invalid("\(field) has the duplicate choice value \(value.json.json).")
                }

                // Core option names are text, so "true" and true would answer under the same name.
                guard !values.contains(where: { $0.coreName == value.coreName }) else {
                    throw invalid(
                        "\(field) has both the string and the boolean \(value.coreName); they cannot be told apart."
                    )
                }

                values.append(value)
                try options.append(.init(name: value.coreName, description: optionalString(
                    choice["description"],
                    field: "choice description"
                )))
            }
            return (
                .choice(id: id, question: instructions, options: options),
                .init(id: id, name: name, kind: .choice, values: values, labels: [])
            )

        case "score":
            guard case let .array(rawLevels)? = question["levels"], (2 ... 10).contains(rawLevels.count) else {
                throw invalid("\(field) levels must contain 2–10 entries.")
            }

            var labels = [String](), levels = [String]()
            for raw in rawLevels {
                guard case let .object(level) = raw else {
                    throw invalid("Each score level must be an object.")
                }

                try allow(level, ["label", "description"])
                guard case let .string(label)? = level["label"],
                      !label.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                else {
                    throw invalid("\(field) level label must be a non-empty string.")
                }

                labels.append(label)
                if let description = try optionalString(level["description"], field: "level description"),
                   !description.isEmpty
                {
                    levels.append("\(label): \(description)")
                }
                else {
                    levels.append(label)
                }
            }
            return (
                .score(id: id, question: instructions, levels: levels),
                .init(id: id, name: name, kind: .score, values: [], labels: labels)
            )

        default:
            return (
                .yesNo(id: id, question: instructions),
                .init(id: id, name: name, kind: .yesNo, values: [], labels: [])
            )
        }
    }

    /// The text joined with newlines in order, and the images in order. Only user messages with `input_text` and
    /// `input_image` parts are served; images must be data URLs.
    private static func parseInput(_ value: JSONValue?) throws -> (String, [DecisionImage]) {
        switch value {
        case let .string(text)?:
            return (text, [])

        case let .array(messages)?:
            var texts = [String](), sources = [(field: String, mediaType: String, base64: String)]()
            for (messageIndex, raw) in messages.enumerated() {
                let field = "input[\(messageIndex)]"
                guard case let .object(message) = raw else {
                    throw invalid("\(field) must be a message object.")
                }

                try allow(message, ["role", "content", "type"])
                guard message["role"] == .string("user") else {
                    throw invalid("\(field) role must be user; other roles are not supported.")
                }

                if let type = message["type"], type != .string("message") {
                    throw invalid("\(field) type must be message.")
                }

                switch message["content"] {
                case let .string(text)?:
                    texts.append(text)

                case let .array(parts)?:
                    for (partIndex, rawPart) in parts.enumerated() {
                        let partField = "\(field).content[\(partIndex)]"
                        guard case let .object(part) = rawPart else {
                            throw invalid("\(partField) must be an object.")
                        }

                        switch part["type"] {
                        case .string("input_text")?:
                            try allow(part, ["type", "text"])
                            guard case let .string(text)? = part["text"] else {
                                throw invalid("\(partField) text must be a string.")
                            }

                            texts.append(text)

                        case .string("input_image")?:
                            try allow(part, ["type", "image_url", "detail"])
                            guard case let .string(url)? = part["image_url"] else {
                                throw invalid("\(partField) image_url must be a string.")
                            }

                            // The detail level is accepted and ignored; images use the chat path's resizing.
                            if let detail = part["detail"], detail != .null {
                                guard case let .string(level) = detail,
                                      ["low", "high", "auto", "original"].contains(level)
                                else {
                                    throw invalid("\(partField) detail must be low, high, auto, original, or null.")
                                }
                            }
                            let (mediaType, base64) = try SystemOneImages.dataURL(url, field: partField)
                            sources.append((partField, mediaType, base64))

                        default:
                            throw invalid("\(partField) type must be input_text or input_image.")
                        }
                    }

                default:
                    throw invalid("\(field) content must be a string or an array of parts.")
                }
            }
            guard sources.count <= SystemOneImages.maximumCount else {
                throw invalid("A request accepts at most \(SystemOneImages.maximumCount) images.")
            }

            var images = [DecisionImage](), total = 0
            for source in sources {
                try images.append(SystemOneImages.decode(
                    source.base64,
                    mediaType: source.mediaType,
                    field: source.field,
                    total: &total
                ))
            }
            return (texts.joined(separator: "\n"), images)

        default:
            throw invalid("input must be a string or an array of user messages.")
        }
    }

    func response(_ result: DecisionResponse) throws -> SystemOneJSON {
        var answers = [SystemOneJSON]()
        for question in questions {
            let names: [String] =
                switch question.kind {
                case .yesNo: ["yes", "no"]
                case .choice: question.values.map(\.coreName)
                case .score: question.labels.indices.map(String.init)
                }
            guard let answer = result.answers[question.id], answer.type == question.kind,
                  Set(answer.probabilities.keys) == Set(names),
                  answer.probabilities.values.allSatisfy({ $0.isFinite && (0 ... 1).contains($0) })
            else {
                throw backendFailure("The model returned an invalid decision answer.")
            }

            let name: SystemOneJSON = question.name.map { .string($0) } ?? .null
            switch question.kind {
            case .yesNo:
                // p(yes) as Core reports it, conditional on the two labels; not renormalized again.
                answers.append(.object([
                    ("type", .string("predicate")),
                    ("name", name),
                    ("probability", .number(answer.probabilities["yes"]!))
                ]))

            case .choice:
                guard let choice = answer.choice,
                      let chosen = question.values.first(where: { $0.coreName == choice }),
                      let confidence = answer.confidence, confidence.isFinite
                else {
                    throw backendFailure("The model returned an invalid choice answer.")
                }

                answers.append(.object([
                    ("type", .string("choice")),
                    ("name", name),
                    ("choice", chosen.json),
                    ("probabilities", .array(question.values.map {
                        .object([("value", $0.json), ("probability", .number(answer.probabilities[$0.coreName]!))])
                    })),
                    ("confidence", .number(confidence))
                ]))

            case .score:
                guard let score = answer.score, score.isFinite,
                      let confidence = answer.confidence, confidence.isFinite
                else {
                    throw backendFailure("The model returned an invalid score answer.")
                }

                answers.append(.object([
                    ("type", .string("score")),
                    ("name", name),
                    ("score", .number(score)),
                    ("probabilities", .array(question.labels.enumerated().map { index, label in
                        .object([
                            ("value", .integer(index)),
                            ("label", .string(label)),
                            ("probability", .number(answer.probabilities[String(index)]!))
                        ])
                    })),
                    ("confidence", .number(confidence))
                ]))
            }
        }
        return .object([
            ("model", .string(result.model.rawValue)),
            ("answers", .array(answers)),
            ("usage", .object([
                ("input_tokens", .integer(result.usage.promptTokens)),
                (
                    "input_tokens_details",
                    .object([("cached_tokens", .integer(0)), ("cache_write_tokens", .integer(0))])
                ),
                ("output_tokens", .integer(0)),
                ("output_tokens_details", .object([("reasoning_tokens", .integer(0))])),
                ("total_tokens", .integer(result.usage.totalTokens))
            ]))
        ])
    }

    private static func usesRemovedSGLangShape(_ body: [String: JSONValue]) -> Bool {
        let removedRequestKeys = [
            "temperature",
            "chat_template_kwargs",
            "prompt_format_version",
            "return_prompt_token_ids"
        ]
        if removedRequestKeys.contains(where: { body[$0] != nil }) {
            return true
        }
        guard case let .array(questions)? = body["questions"] else {
            return false
        }

        let removedQuestionKeys = ["id", "question", "options"]
        for case let .object(question) in questions {
            if removedQuestionKeys.contains(where: { question[$0] != nil }) || question["type"] == .string("yes_no") {
                return true
            }
        }
        return false
    }

    private static func allow(_ object: [String: JSONValue], _ keys: Set<String>) throws {
        // Sorted so the reported field does not depend on dictionary order.
        if let unknown = object.keys.sorted().first(where: { !keys.contains($0) }) {
            throw invalid("Unknown decision field '\(unknown)'.")
        }
    }

    private static func optionalString(_ value: JSONValue?, field: String) throws -> String? {
        guard let value else {
            return nil
        }
        guard case let .string(text) = value else {
            throw invalid("\(field) must be a string.")
        }

        return text
    }

    private static func invalid(_ message: String) -> DecisionWireError { .invalid(message) }
    private func backendFailure(_ message: String) -> SwamaError { .init(code: .backendFailure, message: message) }
}

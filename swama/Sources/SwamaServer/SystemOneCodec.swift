import Foundation
import SwamaCore

struct SystemOneRequest: Sendable {
    struct Question: Sendable {
        let id: String
        let kind: DecisionKind
        let names: [String]
        let legend: [SystemOneJSON]
    }

    let decision: DecisionRequest
    let questions: [Question]

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
                allowsBlankInput: true, allowsPairLabels: true
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

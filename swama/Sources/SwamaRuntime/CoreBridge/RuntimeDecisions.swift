import Foundation

// MARK: - RuntimeDecisionOption

package struct RuntimeDecisionOption: Sendable {
    package let name: String
    package let description: String?

    package init(name: String, description: String?) {
        self.name = name
        self.description = description
    }
}

// MARK: - RuntimeDecisionQuestion

package enum RuntimeDecisionQuestion: Sendable {
    case choice(id: String, question: String, options: [RuntimeDecisionOption])
    case score(id: String, question: String, levels: [String])
    case yesNo(id: String, question: String, yes: String?, no: String?)

    package var id: String {
        switch self {
        case let .choice(id, _, _),
             let .score(id, _, _),
             let .yesNo(id, _, _, _): id
        }
    }

    var type: String {
        switch self {
        case .choice: "choice"
        case .score: "score"
        case .yesNo: "yes_no"
        }
    }

    var names: [String] {
        switch self {
        case let .choice(_, _, options): options.map(\.name)
        case let .score(_, _, levels): levels.indices.map(String.init)
        case .yesNo: ["yes", "no"]
        }
    }

    /// Choices with more than 26 options have no fixed labels; they take the model's pair labels.
    var labels: [String] {
        switch self {
        case let .choice(_, _, options) where options.count > 26: []
        case let .choice(_, _, options):
            options.indices.map { String(UnicodeScalar(65 + $0)!) }
        case let .score(_, _, levels): levels.indices.map(String.init)
        case .yesNo: ["yes", "no"]
        }
    }

    /// SGLang prompt format 1. Keep this versioned wording separate from HTTP JSON parsing.
    func content(input: String, labels: [String]? = nil) -> String {
        let labels = labels ?? self.labels
        var lines = [input, ""]
        switch self {
        case let .choice(_, question, options):
            lines.append("Question: \(question)")
            for (label, option) in zip(labels, options) {
                if let description = option.description, !description.isEmpty {
                    lines.append("\(label): \(option.name) - \(description)")
                }
                else {
                    lines.append("\(label): \(option.name)")
                }
            }
            lines.append("Answer with the letter of one option only.")

        case let .score(_, question, levels):
            lines.append("Question: \(question)")
            for (label, level) in zip(labels, levels) {
                lines.append("\(label): \(level)")
            }
            lines.append("Answer with the number of one level only.")

        case let .yesNo(_, question, yes, no):
            lines.append("Is the following true? \(question)")
            if let yes, !yes.isEmpty {
                lines.append("yes: \(yes)")
            }
            if let no, !no.isEmpty {
                lines.append("no: \(no)")
            }
            lines.append("Answer with yes or no only.")
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - RuntimeDecisionRequest

package struct RuntimeDecisionRequest: Sendable {
    package let model: String
    package let input: String
    package let questions: [RuntimeDecisionQuestion]
    package let temperature: Double
    package let returnPromptTokenIDs: Bool

    package init(
        model: String,
        input: String,
        questions: [RuntimeDecisionQuestion],
        temperature: Double,
        returnPromptTokenIDs: Bool
    ) {
        self.model = model
        self.input = input
        self.questions = questions
        self.temperature = temperature
        self.returnPromptTokenIDs = returnPromptTokenIDs
    }
}

// MARK: - RuntimeDecisionAnswer

package struct RuntimeDecisionAnswer: Sendable {
    package let type: String
    package let probabilities: [String: Double]
    package let labelMass: Double
    package let choice: String?
    package let score: Double?
    package let confidence: Double?
    package let promptTokenIDs: [Int]?
    package let labelTokenIDs: [Int]?
}

// MARK: - RuntimeDecisionResult

package struct RuntimeDecisionResult: Sendable {
    package let model: String
    package let answers: [String: RuntimeDecisionAnswer]
    package let promptTokens: Int
}

/// Model-independent scoring calculation. `logProbs` are normalized against the full vocabulary.
func decisionAnswer(
    question: RuntimeDecisionQuestion,
    logProbs: [Double],
    temperature: Double,
    promptTokenIDs: [Int]?,
    labelTokenIDs: [Int]?
) throws -> RuntimeDecisionAnswer {
    guard logProbs.count == question.names.count,
          logProbs.allSatisfy({ !$0.isNaN && $0 != .infinity }),
          temperature.isFinite,
          temperature > 0
    else {
        throw DecisionScoringError.invalidLogits
    }

    let maximum = logProbs.max()!
    guard maximum.isFinite else {
        throw DecisionScoringError.invalidLogits
    }

    let weights = logProbs.map { exp(($0 - maximum) / temperature) }
    let denominator = weights.reduce(0, +)
    let probabilities = weights.map { $0 / denominator }
    let mass = logProbs.map { exp($0) }.reduce(0, +)
    guard denominator.isFinite, denominator > 0, mass.isFinite, (0 ... 1.000_001).contains(mass) else {
        throw DecisionScoringError.invalidLogits
    }

    let names = question.names
    let named = Dictionary(uniqueKeysWithValues: zip(names, probabilities))
    // SGLang SystemOne confidence, pinned to eb9c9ee99d47bf4c526a06cd84da59cd9cf4e2a5.
    // Normalize only for this derived value; preserve the reported probabilities and label mass.
    let probabilityTotal = probabilities.reduce(0, +)
    let q = probabilities.map { $0 / probabilityTotal }
    let count = Double(q.count)
    let top = q.indices.reduce(0) { q[$1] > q[$0] ? $1 : $0 }
    let choice: String?
    let score: Double?
    let confidence: Double?
    switch question {
    case .choice:
        let maxIndex = probabilities.indices.reduce(0) { probabilities[$1] > probabilities[$0] ? $1 : $0 }
        choice = names[maxIndex]
        score = nil
        confidence = min(1, max(0, (count * q[top] - 1) / (count - 1)))

    case .score:
        choice = nil
        score = probabilities.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element }
        let spread = q.enumerated().reduce(0) { $0 + $1.element * abs(Double($1.offset - top)) }
        let midpoint = (count - 1) / 2
        let uniformSpread = q.indices.reduce(0.0) { $0 + abs(Double($1) - midpoint) } / count
        confidence = max(0, 1 - spread / uniformSpread)

    case .yesNo:
        choice = nil
        score = nil
        confidence = nil
    }
    return .init(
        type: question.type,
        probabilities: named,
        labelMass: mass,
        choice: choice,
        score: score,
        confidence: confidence,
        promptTokenIDs: promptTokenIDs,
        labelTokenIDs: labelTokenIDs
    )
}

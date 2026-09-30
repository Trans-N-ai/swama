import Foundation
import SwamaCore

extension DecisionQuestion {
    var decisionLabels: [String] {
        switch self {
        case let .choice(_, _, options): options.indices.map { String(UnicodeScalar(65 + $0)!) }
        case let .score(_, _, levels): levels.indices.map(String.init)
        case .yesNo: ["yes", "no"]
        }
    }

    var decisionNames: [String] {
        switch self {
        case let .choice(_, _, options): options.map(\.name)
        case let .score(_, _, levels): levels.indices.map(String.init)
        case .yesNo: ["yes", "no"]
        }
    }

    var decisionKind: DecisionKind {
        switch self {
        case .choice: .choice
        case .score: .score
        case .yesNo: .yesNo
        }
    }

    func decisionPrompt(input: String) -> String {
        var lines = [input, ""]
        switch self {
        case let .choice(_, question, options):
            lines.append("Question: \(question)")
            for (label, option) in zip(decisionLabels, options) {
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
            for (label, level) in zip(decisionLabels, levels) {
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

    func decisionAnswer(
        logProbs: [Double],
        temperature: Double,
        promptTokenIDs: [Int]?,
        labelTokenIDs: [Int]?
    ) throws -> DecisionAnswer {
        let names = decisionNames
        guard names.count == logProbs.count,
              logProbs.allSatisfy({ !$0.isNaN && $0 != .infinity }),
              temperature.isFinite, temperature > 0
        else {
            throw SwamaError(code: .backendFailure, message: "The model returned invalid decision logits.")
        }

        let maximum = logProbs.max()!
        guard maximum.isFinite else {
            throw SwamaError(code: .backendFailure, message: "The model returned invalid decision logits.")
        }

        let weights = logProbs.map { exp(($0 - maximum) / temperature) }
        let total = weights.reduce(0, +)
        let probabilities = weights.map { $0 / total }
        let mass = logProbs.map { exp($0) }.reduce(0, +)
        guard total.isFinite, total > 0, mass.isFinite, (0 ... 1.000_001).contains(mass) else {
            throw SwamaError(code: .backendFailure, message: "The model returned invalid decision logits.")
        }

        let choice: String?
        let score: Double?
        switch self {
        case .choice:
            let maxIndex = probabilities.indices.reduce(0) { probabilities[$1] > probabilities[$0] ? $1 : $0 }
            choice = names[maxIndex]
            score = nil

        case .score:
            choice = nil
            score = probabilities.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element }

        case .yesNo:
            choice = nil
            score = nil
        }
        return .init(
            type: decisionKind,
            probabilities: Dictionary(uniqueKeysWithValues: zip(names, probabilities)),
            labelMass: mass,
            choice: choice,
            score: score,
            promptTokenIDs: promptTokenIDs,
            labelTokenIDs: labelTokenIDs
        )
    }
}

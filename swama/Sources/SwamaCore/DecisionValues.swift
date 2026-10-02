import Foundation

// MARK: - DecisionOption

/// One candidate in a choice decision. Names are returned as probability keys.
public struct DecisionOption: Hashable, Sendable {
    public init(name: String, description: String? = nil) {
        self.name = name
        self.description = description
    }

    public let name: String
    public let description: String?
}

// MARK: - DecisionQuestion

/// The three question forms served by `/v1/decisions` prompt format 1.
public enum DecisionQuestion: Hashable, Sendable {
    case choice(id: String, question: String, options: [DecisionOption])
    case score(id: String, question: String, levels: [String])
    case yesNo(id: String, question: String, yes: String? = nil, no: String? = nil)

    public var id: String {
        switch self {
        case let .choice(id, _, _),
             let .score(id, _, _),
             let .yesNo(id, _, _, _): id
        }
    }
}

// MARK: - DecisionRequest

public struct DecisionRequest: Hashable, Sendable {
    public init(
        model: ModelID,
        input: String,
        questions: [DecisionQuestion],
        temperature: Double = 1,
        returnPromptTokenIDs: Bool = false,
        allowsBlankInput: Bool = false,
        allowsPairLabels: Bool = false
    ) {
        self.model = model
        self.input = input
        self.questions = questions
        self.temperature = temperature
        self.returnPromptTokenIDs = returnPromptTokenIDs
        self.allowsBlankInput = allowsBlankInput
        self.allowsPairLabels = allowsPairLabels
    }

    public let model: ModelID
    public let input: String
    public let questions: [DecisionQuestion]
    public let temperature: Double
    public let returnPromptTokenIDs: Bool
    /// Allows a blank input, rendered as is. Off by default, so library callers keep requiring input.
    public let allowsBlankInput: Bool
    /// Allows choices with more than 26 options, labeled with the model's two-letter labels.
    /// Off by default, so library callers keep the 2–26 option limit.
    public let allowsPairLabels: Bool
}

// MARK: - DecisionKind

public enum DecisionKind: String, Codable, Hashable, Sendable {
    case choice
    case score
    case yesNo = "yes_no"
}

// MARK: - DecisionAnswer

public struct DecisionAnswer: Hashable, Sendable {
    public init(
        type: DecisionKind,
        probabilities: [String: Double],
        labelMass: Double,
        choice: String? = nil,
        score: Double? = nil,
        confidence: Double? = nil,
        promptTokenIDs: [Int]? = nil,
        labelTokenIDs: [Int]? = nil
    ) {
        self.type = type
        self.probabilities = probabilities
        self.labelMass = labelMass
        self.choice = choice
        self.score = score
        self.confidence = confidence
        self.promptTokenIDs = promptTokenIDs
        self.labelTokenIDs = labelTokenIDs
    }

    public let type: DecisionKind
    public let probabilities: [String: Double]
    public let labelMass: Double
    public let choice: String?
    public let score: Double?
    /// Concentration among candidate labels, not a calibrated probability of correctness.
    public let confidence: Double?
    public let promptTokenIDs: [Int]?
    public let labelTokenIDs: [Int]?
}

// MARK: - DecisionResponse

public struct DecisionResponse: Hashable, Sendable {
    public init(model: ModelID, answers: [String: DecisionAnswer], usage: Usage) {
        self.model = model
        self.answers = answers
        self.usage = usage
    }

    public let model: ModelID
    public let answers: [String: DecisionAnswer]
    public let usage: Usage
    public let promptFormatVersion = 1
}

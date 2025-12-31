import Foundation

public enum ContextLimitError: Error, LocalizedError {
    case exceededAfterTrimming(limit: Int, promptTokens: Int)

    public var errorDescription: String? {
        switch self {
        case let .exceededAfterTrimming(limit, promptTokens):
            "Prompt still exceeds context limit (\(promptTokens) > \(limit)) after trimming."
        }
    }
}

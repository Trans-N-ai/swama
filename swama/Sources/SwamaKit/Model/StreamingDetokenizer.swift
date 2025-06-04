import Tokenizers

public struct StreamingDetokenizer {
    // MARK: Lifecycle

    public init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    // MARK: Public

    /// Appends a new token to the stream.
    /// If a complete string chunk can be formed, it is returned.
    /// Otherwise, nil is returned.
    public mutating func append(token: Int) -> String? {
        buffer.append(token)

        let decoded = tokenizer.decode(tokens: buffer, skipSpecialTokens: false)

        // Check if the decoded string has replacement characters (incomplete token)
        if decoded.contains("\u{fffd}") {
            return nil
        }

        // Buffer was successfully decoded → flush to tokens
        tokens.append(contentsOf: buffer)
        buffer.removeAll()

        let fullDecoded = tokenizer.decode(tokens: tokens, skipSpecialTokens: true)
        let delta = fullDecoded.dropFirst(lastDecoded.count)

        lastDecoded = fullDecoded
        return String(delta)
    }

    // MARK: Private

    private let tokenizer: Tokenizer
    private var tokens: [Int] = []
    private var lastDecoded: String = ""
    private var buffer: [Int] = []
}

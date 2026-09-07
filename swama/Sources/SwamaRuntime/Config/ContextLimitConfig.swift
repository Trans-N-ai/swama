import Foundation

/// Shared configuration for context window limits.
/// Defaults to 16_384 tokens unless overridden by the SWAMA_CONTEXT_LIMIT environment variable
/// or updated at runtime (CLI flag / macOS UI).
package actor ContextLimitConfig {
    package static let shared: ContextLimitConfig = .init()

    package enum Constants {
        package static let defaultLimit = 16384
    }

    private var limit: Int

    private init() {
        if let envValue = ProcessInfo.processInfo.environment["SWAMA_CONTEXT_LIMIT"],
           let parsed = Int(envValue),
           parsed > 0
        {
            limit = parsed
        }
        else {
            limit = Constants.defaultLimit
        }
    }

    /// Returns the current context limit (tokens for the prompt).
    package func currentLimit() -> Int {
        limit
    }

    /// Updates the context limit. Values <= 0 are ignored.
    package func updateLimit(_ newValue: Int) {
        guard newValue > 0 else {
            return
        }

        limit = newValue
    }
}

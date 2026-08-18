import Foundation

/// Whether the OpenAI-compatible chat path reuses KV-cache state between requests on the same
/// model (longest-common-prefix restore against `PromptCacheStore`, suffix-only prefill).
///
/// On by default. Set `SWAMA_PROMPT_CACHE=0` to force every request through the uncached
/// full-prefill path -- an escape hatch for A/B latency comparisons, or to rule out a
/// cache-related issue in the field without a redeploy. The environment is re-read on every
/// access (rather than cached once at process start) so the switch can be toggled within a
/// single process, which is also what lets it be exercised directly in tests; the cost is one
/// dictionary lookup per chat request, not per token.
public enum PromptCacheConfig {
    public static var isEnabled: Bool {
        ProcessInfo.processInfo.environment["SWAMA_PROMPT_CACHE"] != "0"
    }
}

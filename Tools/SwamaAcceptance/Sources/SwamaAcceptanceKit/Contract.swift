import Foundation

// MARK: - AcceptanceContract

struct AcceptanceContract: Codable, Sendable {
    let schemaVersion: Int
    let build: BuildContract
    let models: [ModelContract]
    let benchmark: BenchmarkContract
    let reliability: ReliabilityContract
    let architecture: ArchitectureContract

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case build
        case models
        case benchmark
        case reliability
        case architecture
    }

    static func load(from url: URL) throws -> Self {
        do {
            return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        }
        catch {
            throw AcceptanceFailure.unknown("cannot decode contract \(url.path): \(error)")
        }
    }
}

// MARK: - BuildContract

struct BuildContract: Codable, Sendable {
    let productTimeoutSeconds: Double
    let externalFixtureTimeoutSeconds: Double
    let externalFixtureTimeoutRetryLimit: Int

    enum CodingKeys: String, CodingKey {
        case productTimeoutSeconds = "product_timeout_seconds"
        case externalFixtureTimeoutSeconds = "external_fixture_timeout_seconds"
        case externalFixtureTimeoutRetryLimit = "external_fixture_timeout_retry_limit"
    }
}

// MARK: - ModelContract

struct ModelContract: Codable, Sendable {
    let id: String
    let role: String
    let benchmark: Bool
}

// MARK: - BenchmarkContract

struct BenchmarkContract: Codable, Sendable {
    let maxTokens: Int
    let rounds: Int
    let warmupRounds: Int
    let minimumMeasuredRounds: Int
    let prompt: String
    let comparison: ComparisonContract

    enum CodingKeys: String, CodingKey {
        case maxTokens = "max_tokens"
        case rounds
        case warmupRounds = "warmup_rounds"
        case minimumMeasuredRounds = "minimum_measured_rounds"
        case prompt
        case comparison
    }
}

// MARK: - ComparisonContract

struct ComparisonContract: Codable, Sendable {
    let ttftRatioMax: Double
    let ttftAbsoluteSlackMilliseconds: Double
    let tokensPerSecondRatioMin: Double
    let peakResidentRatioMax: Double
    let peakResidentAbsoluteSlackBytes: UInt64

    enum CodingKeys: String, CodingKey {
        case ttftRatioMax = "ttft_ratio_max"
        case ttftAbsoluteSlackMilliseconds = "ttft_absolute_slack_ms"
        case tokensPerSecondRatioMin = "tokens_per_second_ratio_min"
        case peakResidentRatioMax = "peak_resident_ratio_max"
        case peakResidentAbsoluteSlackBytes = "peak_resident_absolute_slack_bytes"
    }
}

// MARK: - ReliabilityContract

struct ReliabilityContract: Codable, Sendable {
    let cancelMaxTokens: Int
    let cancelObservedTokenCeiling: Int
    let cancellationLatencyMillisecondsMax: Double
    let concurrentRequests: Int
    let concurrentMaxTokens: Int
    let repeatRounds: Int
    let repeatMaxTokens: Int
    let switchCycles: Int
    let switchMaxTokens: Int
    let releaseSettleMilliseconds: Int
    let releaseRSSRatioMax: Double
    let releaseRSSAbsoluteSlackBytes: UInt64
    let disconnectRecoveryTTFTMillisecondsMax: Double
    let timeoutSeconds: Double

    enum CodingKeys: String, CodingKey {
        case cancelMaxTokens = "cancel_max_tokens"
        case cancelObservedTokenCeiling = "cancel_observed_token_ceiling"
        case cancellationLatencyMillisecondsMax = "cancellation_latency_ms_max"
        case concurrentRequests = "concurrent_requests"
        case concurrentMaxTokens = "concurrent_max_tokens"
        case repeatRounds = "repeat_rounds"
        case repeatMaxTokens = "repeat_max_tokens"
        case switchCycles = "switch_cycles"
        case switchMaxTokens = "switch_max_tokens"
        case releaseSettleMilliseconds = "release_settle_ms"
        case releaseRSSRatioMax = "release_rss_ratio_max"
        case releaseRSSAbsoluteSlackBytes = "release_rss_absolute_slack_bytes"
        case disconnectRecoveryTTFTMillisecondsMax = "disconnect_recovery_ttft_ms_max"
        case timeoutSeconds = "timeout_seconds"
    }
}

// MARK: - ArchitectureContract

struct ArchitectureContract: Codable, Sendable {
    let legacyForbiddenImportAllowlist: [String]
    let legacyPublicMLXLeakAllowlist: [String]
    let goalCoreTarget: String
    let goalForbiddenImports: [String]

    enum CodingKeys: String, CodingKey {
        case legacyForbiddenImportAllowlist = "legacy_forbidden_import_allowlist"
        case legacyPublicMLXLeakAllowlist = "legacy_public_mlx_leak_allowlist"
        case goalCoreTarget = "goal_core_target"
        case goalForbiddenImports = "goal_forbidden_imports"
    }
}

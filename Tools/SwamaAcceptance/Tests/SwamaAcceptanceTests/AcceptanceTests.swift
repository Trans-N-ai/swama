import Foundation
@testable import SwamaAcceptanceKit
import Testing

@Suite("Swama acceptance harness")
struct AcceptanceTests {
    @Test func reportSealRejectsEditedPayload() throws {
        var report: JSONObject = ["passed": true, "metric": 10]
        try sealReport(&report)
        try verifyReport(report)

        report["metric"] = 100
        #expect(throws: AcceptanceFailure.self) {
            try verifyReport(report)
        }
    }

    @Test func compareDetectsARealPerformanceRegression() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract)
        var baseline = try syntheticReport(paths: paths)
        var candidate = baseline
        var benchmarks = try candidate.object("benchmarks")
        var model = try benchmarks.object("model")
        var core = try model.object("core")
        var summary = try core.object("summary")
        summary["tokens_per_second_median"] = 1.0
        core["summary"] = summary
        model["core"] = core
        benchmarks["model"] = model
        candidate["benchmarks"] = benchmarks
        try sealReport(&baseline)
        try sealReport(&candidate)

        let result = try compareReports(
            baseline: baseline,
            candidate: candidate,
            contract: contract,
            contractURL: paths.contract,
            paths: paths
        )
        #expect(result["passed"] as? Bool == false)
        #expect((result["findings"] as? [JSONObject])?
            .contains { $0["metric"] as? String == "tokens_per_second" } == true
        )
    }

    @Test func compareRejectsPostHocInstrumentIdentityChange() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract)
        var baseline = try syntheticReport(paths: paths)
        var candidate = baseline
        var instrument = try baseline.object("instrument")
        instrument["harness_sources_sha256"] = String(repeating: "0", count: 64)
        baseline["instrument"] = instrument
        try sealReport(&baseline)
        try sealReport(&candidate)

        #expect(throws: AcceptanceFailure.self) {
            _ = try compareReports(
                baseline: baseline,
                candidate: candidate,
                contract: contract,
                contractURL: paths.contract,
                paths: paths
            )
        }
    }

    @Test func compareRejectsPostHocContractChange() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract)
        var baseline = try syntheticReport(paths: paths)
        var candidate = baseline
        try sealReport(&baseline)
        try sealReport(&candidate)
        let alteredContract = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-altered-contract-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: alteredContract) }
        var bytes = try Data(contentsOf: paths.contract)
        bytes.append(0x0A)
        try bytes.write(to: alteredContract)

        #expect(throws: AcceptanceFailure.self) {
            _ = try compareReports(
                baseline: baseline,
                candidate: candidate,
                contract: contract,
                contractURL: alteredContract,
                paths: paths
            )
        }
    }

    @Test func compareRejectsDifferentModelBytes() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract)
        var baseline = try syntheticReport(paths: paths)
        var candidate = baseline
        var provenance = try candidate.object("provenance")
        provenance["models"] = [["id": "test", "sha256": "different"]]
        candidate["provenance"] = provenance
        try sealReport(&baseline)
        try sealReport(&candidate)

        #expect(throws: AcceptanceFailure.self) {
            _ = try compareReports(
                baseline: baseline,
                candidate: candidate,
                contract: contract,
                contractURL: paths.contract,
                paths: paths
            )
        }
    }

    @Test func truncatedReportIsUnknown() throws {
        let truncated = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-truncated-report-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: truncated) }
        try Data(#"{"passed":true"#.utf8).write(to: truncated)
        #expect(throws: AcceptanceFailure.self) {
            _ = try loadJSONObject(truncated)
        }
    }

    @Test func cancellationAndDisconnectGatesCarryWeight() throws {
        let contract = ReliabilityContract(
            cancelMaxTokens: 512,
            cancelObservedTokenCeiling: 8,
            cancellationLatencyMillisecondsMax: 1000,
            concurrentRequests: 2,
            concurrentMaxTokens: 48,
            repeatRounds: 12,
            repeatMaxTokens: 32,
            switchCycles: 2,
            switchMaxTokens: 16,
            releaseSettleMilliseconds: 1000,
            releaseRSSRatioMax: 1.25,
            releaseRSSAbsoluteSlackBytes: 268_435_456,
            disconnectRecoveryTTFTMillisecondsMax: 1000,
            timeoutSeconds: 300
        )
        var report = reliabilityReport()
        var gates = try reliabilityGateStatus(report, contract: contract)
        #expect(!gates.values.contains(false))

        var cancel = try report.object("cancel_then_recover")
        cancel["observedTokensBeforeCancel"] = 99
        report["cancel_then_recover"] = cancel
        gates = try reliabilityGateStatus(report, contract: contract)
        #expect(gates["cancel_then_recover"] == false)

        report = reliabilityReport()
        var disconnect = try report.object("http_disconnect_then_recover")
        var followup = try disconnect.object("followup")
        followup["ttft_ms"] = 5000
        disconnect["followup"] = followup
        report["http_disconnect_then_recover"] = disconnect
        gates = try reliabilityGateStatus(report, contract: contract)
        #expect(gates["http_disconnect_then_recover"] == false)
    }

    @Test func architectureGateRejectsInternalShellReverseImport() throws {
        let temporary = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-architecture-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }
        for directory in [
            "swama/Sources/SwamaKit",
            "Tools/SwamaAcceptance",
            "Tests/AcceptanceFixture"
        ] {
            try FileManager.default.createDirectory(
                at: temporary.appendingPathComponent(directory),
                withIntermediateDirectories: true
            )
        }
        for file in [
            "swama/Package.swift",
            "Tools/SwamaAcceptance/Package.swift",
            "Tests/AcceptanceFixture/Package.swift"
        ] {
            try Data().write(to: temporary.appendingPathComponent(file))
        }
        try Data("import SwamaServer\n".utf8).write(
            to: temporary.appendingPathComponent("swama/Sources/SwamaKit/Bad.swift")
        )

        let paths = try WorkspacePaths.discover(explicit: temporary.path)
        let contract = ArchitectureContract(
            legacyForbiddenImportAllowlist: [],
            legacyPublicMLXLeakAllowlist: [],
            goalCoreTarget: "SwamaCore",
            goalForbiddenImports: ["SwamaServer", "SwamaAppSupport"]
        )
        let report = try architectureReport(contract: contract, stage: .legacyRatchet, paths: paths)
        #expect(report["passed"] as? Bool == false)
        #expect((report["new_forbidden_imports"] as? [String]) == ["swama/Sources/SwamaKit/Bad.swift:SwamaServer"])
    }

    @Test func medianUsesAllMeasuredSamples() {
        #expect(median([511, 568, 580, 574, 572]) == 572)
    }

    @Test func provenanceUsesResolvableAbsoluteSystemTools() {
        for tool in [SystemTool.systemProfiler, SystemTool.uname, SystemTool.swVers] {
            #expect(tool.hasPrefix("/"))
            #expect(FileManager.default.isExecutableFile(atPath: tool))
        }
    }

    @Test func cleanWorktreeCanBeBoundToHead() throws {
        let repository = try temporaryRepository()
        defer { try? FileManager.default.removeItem(at: repository) }
        let paths = try WorkspacePaths.discover(explicit: repository.path)
        try requireCleanWorktree(paths: paths)
    }

    @Test func trackedDirtyWorktreeIsUnknown() throws {
        let repository = try temporaryRepository()
        defer { try? FileManager.default.removeItem(at: repository) }
        try Data("changed\n".utf8).write(to: repository.appendingPathComponent("swama/Package.swift"))
        let paths = try WorkspacePaths.discover(explicit: repository.path)
        #expect(throws: AcceptanceFailure.self) {
            try requireCleanWorktree(paths: paths)
        }
    }

    @Test func untrackedDirtyWorktreeIsUnknown() throws {
        let repository = try temporaryRepository()
        defer { try? FileManager.default.removeItem(at: repository) }
        try Data("untracked\n".utf8).write(to: repository.appendingPathComponent("untracked.txt"))
        let paths = try WorkspacePaths.discover(explicit: repository.path)
        #expect(throws: AcceptanceFailure.self) {
            try requireCleanWorktree(paths: paths)
        }
    }

    @Test func buildFailureSummaryKeepsErrorLines() {
        let result = CommandResult(
            command: ["swift", "build"],
            returnCode: 1,
            stdout: String(repeating: "warning: setup\n", count: 500) + "error: root cause\n",
            stderr: "warning: final warning\n",
            durationMilliseconds: 1,
            peakResidentBytes: 0
        )
        let summary = commandFailureSummary(result, maximumCharacters: 80)
        #expect(summary.contains("error: root cause"))
        #expect(summary.contains("output tail:"))
    }

    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func syntheticReport(paths: WorkspacePaths) throws -> JSONObject {
        let instrument = try currentInstrumentIdentity(paths: paths)
        let summary: JSONObject = [
            "ttft_median_ms": 10.0,
            "tokens_per_second_median": 100.0,
            "peak_resident_bytes": 1000
        ]
        let route: JSONObject = ["summary": summary]
        return try [
            "contract_sha256": sha256File(paths.contract),
            "instrument": instrument,
            "provenance": [
                "platform": ["machine": "test"],
                "toolchain": ["swift": "test"],
                "models": [["id": "test"]]
            ],
            "build": [
                "metal_build": [
                    "metal_version": "test",
                    "metallib_version": "test",
                    "metal_executable_sha256": "test",
                    "metallib_executable_sha256": "test",
                    "mlx_swift_revision": "test",
                    "sha256": "test",
                    "sources": [["path": "a.metal", "sha256": "test"]]
                ]
            ],
            "benchmarks": ["model": ["core": route, "http": route]],
            "passed": true
        ]
    }

    private func reliabilityReport() -> JSONObject {
        [
            "cancel_then_recover": [
                "observedTokensBeforeCancel": 2,
                "cancellationMilliseconds": 10.0,
                "followupOutput": "ok"
            ],
            "same_model_concurrency": ["outputs": ["a", "b"]],
            "repeat_generation": ["outputs_nonempty": true],
            "model_switch_and_release": ["release_gate_passed": true],
            "http_disconnect_then_recover": [
                "followup": ["output": "ok", "ttft_ms": 10.0]
            ]
        ]
    }

    private func temporaryRepository() throws -> URL {
        let root = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-clean-tree-test-\(UUID().uuidString)")
        for directory in [
            "swama",
            "Tools/SwamaAcceptance",
            "Tests/AcceptanceFixture"
        ] {
            try FileManager.default.createDirectory(
                at: root.appendingPathComponent(directory),
                withIntermediateDirectories: true
            )
        }
        for file in [
            "swama/Package.swift",
            "Tools/SwamaAcceptance/Package.swift",
            "Tests/AcceptanceFixture/Package.swift"
        ] {
            try Data("fixture\n".utf8).write(to: root.appendingPathComponent(file))
        }
        for command in [
            ["git", "init"],
            ["git", "config", "user.email", "acceptance@example.invalid"],
            ["git", "config", "user.name", "Acceptance Test"],
            ["git", "add", "."],
            ["git", "commit", "-m", "fixture"]
        ] {
            let result = try runCommand(
                command,
                currentDirectory: root,
                timeout: 30,
                sampleMemory: false
            )
            guard result.returnCode == 0 else {
                throw AcceptanceFailure.unknown("cannot prepare clean repository: \(commandFailureSummary(result))")
            }
        }
        return root
    }
}

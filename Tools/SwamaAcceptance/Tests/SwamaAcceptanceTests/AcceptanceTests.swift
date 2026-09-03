import Darwin
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
        let coverage = try result.object("comparison_coverage")
        #expect(try coverage.integer("benchmark_models") == 1)
        #expect(try coverage.integer("benchmark_routes") == 2)
    }

    @Test func compareRejectsAnEmptyBaselineBenchmarkSet() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract)
        var baseline = try syntheticReport(paths: paths)
        var candidate = baseline
        baseline["benchmarks"] = JSONObject()
        candidate["benchmarks"] = JSONObject()
        try sealReport(&baseline)
        try sealReport(&candidate)

        do {
            _ = try compareReports(
                baseline: baseline,
                candidate: candidate,
                contract: contract,
                contractURL: paths.contract,
                paths: paths
            )
            Issue.record("an empty baseline benchmark set must not compare as passing")
        }
        catch let error as AcceptanceFailure {
            if case .failed = error.kind {
                Issue.record("an empty benchmark set is invalid evidence, not a product failure")
            }
            #expect(error.message == "baseline contains no benchmark models")
        }
    }

    @Test func compareRejectsNonStringMetalIdentityValues() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract)
        var baseline = try syntheticReport(paths: paths)
        var candidate = baseline
        var baselineBuild = try baseline.object("build")
        var baselineMetal = try baselineBuild.object("metal_build")
        baselineMetal["metal_version"] = 1
        baselineBuild["metal_build"] = baselineMetal
        baseline["build"] = baselineBuild
        var candidateBuild = try candidate.object("build")
        var candidateMetal = try candidateBuild.object("metal_build")
        candidateMetal["metal_version"] = 1
        candidateBuild["metal_build"] = candidateMetal
        candidate["build"] = candidateBuild
        try sealReport(&baseline)
        try sealReport(&candidate)

        do {
            _ = try compareReports(
                baseline: baseline,
                candidate: candidate,
                contract: contract,
                contractURL: paths.contract,
                paths: paths
            )
            Issue.record("non-string metal identity values must not compare as equal")
        }
        catch let error as AcceptanceFailure {
            if case .failed = error.kind {
                Issue.record("metal identity schema drift is invalid evidence, not a product failure")
            }
            #expect(error.message == "missing JSON string: metal_version")
        }
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

    @Test func architectureGateRejectsAttributedShellReverseImports() throws {
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
        try Data(
            """
            @_exported import SwamaServer
            internal import AppKit
            @testable import ArgumentParser
            @preconcurrency import MLXLMCommon

            """.utf8
        ).write(
            to: temporary.appendingPathComponent("swama/Sources/SwamaKit/Bad.swift")
        )

        let paths = try WorkspacePaths.discover(explicit: temporary.path)
        let contract = ArchitectureContract(
            legacyForbiddenImportAllowlist: [],
            legacyPublicMLXLeakAllowlist: [],
            goalCoreTarget: "SwamaCore",
            goalForbiddenImports: ["AppKit", "ArgumentParser", "SwamaServer", "SwamaAppSupport"]
        )
        let report = try architectureReport(contract: contract, stage: .legacyRatchet, paths: paths)
        #expect(report["passed"] as? Bool == false)
        #expect((report["new_forbidden_imports"] as? [String]) == [
            "swama/Sources/SwamaKit/Bad.swift:AppKit",
            "swama/Sources/SwamaKit/Bad.swift:ArgumentParser",
            "swama/Sources/SwamaKit/Bad.swift:SwamaServer"
        ])
    }

    @Test func architectureImportParserHandlesAttributesAccessAndScopedImports() throws {
        let expression = try NSRegularExpression(pattern: swiftImportDeclarationPattern)
        let imports = [
            "import SwamaServer": "SwamaServer",
            "@_exported import NIO": "NIO",
            "@preconcurrency import AppKit": "AppKit",
            "@testable import SwamaServer": "SwamaServer",
            "@_implementationOnly import ArgumentParser": "ArgumentParser",
            "@_spi(Testing) import NIOHTTP1": "NIOHTTP1",
            "internal import SwamaAppSupport": "SwamaAppSupport",
            "package import struct NIOCore.ByteBuffer": "NIOCore",
            "@preconcurrency import MLXLMCommon": "MLXLMCommon"
        ]

        for (line, module) in imports {
            #expect(swiftImportedModule(in: line, matching: expression) == module)
        }
        #expect(swiftImportedModule(in: "// @_exported import NIO", matching: expression) == nil)
        #expect(swiftImportedModule(in: "let example = \"import NIO\"", matching: expression) == nil)
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

    @Test func developerEnvironmentBindsSDKToSelectedXcode() throws {
        let temporary = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-developer-environment-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }
        let xcodebuild = temporary.appendingPathComponent("usr/bin/xcodebuild")
        let sdk = temporary.appendingPathComponent(
            "Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
        )
        try FileManager.default.createDirectory(
            at: xcodebuild.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data().write(to: xcodebuild)
        try FileManager.default.createDirectory(at: sdk, withIntermediateDirectories: true)

        let environment = try developerEnvironment(temporary)
        #expect(environment["DEVELOPER_DIR"] == temporary.path)
        #expect(environment["SDKROOT"] == sdk.path)
    }

    @Test func buildScratchDropsStaleIdentityAndResumesMatchingIdentity() throws {
        let temporary = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-build-scratch-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)

        let stale = try prepareAcceptanceBuildScratch(root: temporary, identity: "old")
        let staleArtifact = stale.package.appendingPathComponent("stale-object")
        try FileManager.default.createDirectory(at: stale.package, withIntermediateDirectories: true)
        try Data("stale".utf8).write(to: staleArtifact)

        let current = try prepareAcceptanceBuildScratch(root: temporary, identity: "current")
        #expect(!FileManager.default.fileExists(atPath: staleArtifact.path))
        let partialArtifact = current.fixture.appendingPathComponent("partial-object")
        try FileManager.default.createDirectory(at: current.fixture, withIntermediateDirectories: true)
        try Data("partial".utf8).write(to: partialArtifact)

        let resumed = try prepareAcceptanceBuildScratch(root: temporary, identity: "current")
        #expect(resumed.identity == "current")
        #expect(FileManager.default.fileExists(atPath: partialArtifact.path))
    }

    @Test func buildTimeoutIsUnknownInsteadOfProductFailure() throws {
        do {
            _ = try runCommand(
                ["/bin/sleep", "1"],
                currentDirectory: FileManager.default.temporaryDirectory,
                timeout: 0.01,
                sampleMemory: false,
                timeoutFailureKind: .unknown,
                timeoutContext: "external fixture build"
            )
            Issue.record("the command must time out")
        }
        catch let error as AcceptanceFailure {
            if case .failed = error.kind {
                Issue.record("a build timeout is tool UNKNOWN, not product FAIL")
            }
            #expect(error.message.contains("external fixture build timeout after 0.01s"))
        }
    }

    @Test func externalBuildTimeoutResumesWithoutManualIntervention() throws {
        let temporary = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-build-retry-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        let marker = temporary.appendingPathComponent("first-attempt")

        let result = try runCommand(
            [
                "/bin/sh",
                "-c",
                "if [ ! -f \"$1\" ]; then : > \"$1\"; sleep 2; fi",
                "resumable-build",
                marker.path
            ],
            currentDirectory: temporary,
            timeout: 0.2,
            sampleMemory: false,
            timeoutFailureKind: .unknown,
            timeoutContext: "external fixture build",
            timeoutRetryLimit: 1
        )
        #expect(result.returnCode == 0)
        #expect(result.attemptCount == 2)
        #expect(result.timeoutCount == 1)
        #expect(result.durationMilliseconds >= 200)
    }

    @Test func commandTimeoutTerminatesDescendantProcesses() throws {
        let temporary = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-process-tree-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        let childPIDFile = temporary.appendingPathComponent("child-pid")

        #expect(throws: AcceptanceFailure.self) {
            _ = try runCommand(
                [
                    "/bin/sh",
                    "-c",
                    "sleep 5 & echo $! > \"$1\"; wait",
                    "spawn-child",
                    childPIDFile.path
                ],
                currentDirectory: temporary,
                timeout: 0.2,
                sampleMemory: false,
                timeoutFailureKind: .unknown
            )
        }
        let childPID = try #require(
            pid_t(String(contentsOf: childPIDFile, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines))
        )
        #expect(kill(childPID, 0) == -1)
    }

    @Test func contractCarriesSeparateProductAndFixtureBuildBudgets() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract)
        #expect(contract.schemaVersion == 3)
        #expect(contract.build.productTimeoutSeconds == 1200)
        #expect(contract.build.externalFixtureTimeoutSeconds == 1200)
        #expect(contract.build.externalFixtureTimeoutRetryLimit == 1)
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
            peakResidentBytes: 0,
            attemptCount: 1,
            timeoutCount: 0
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

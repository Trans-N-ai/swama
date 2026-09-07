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

    @Test func jsonAccessorsDistinguishMissingKeysFromWrongTypes() throws {
        let key = "value"
        let accessors: [(expectedType: String, wrongValue: Any, read: (JSONObject) throws -> Void)] = [
            ("object", [Any](), { _ = try $0.object(key) }),
            ("array", JSONObject(), { _ = try $0.array(key) }),
            ("string", NSNumber(value: true), { _ = try $0.string(key) }),
            ("integer", NSNumber(value: 1.5), { _ = try $0.integer(key) }),
            ("number", NSNumber(value: true), { _ = try $0.double(key) }),
            ("boolean", NSNumber(value: 1), { _ = try $0.boolean(key) })
        ]

        for accessor in accessors {
            let missing = try #require(
                acceptanceFailure(from: [:], read: accessor.read)
            )
            if case .failed = missing.kind {
                Issue.record("a missing JSON key must be UNKNOWN")
            }
            #expect(missing.message.contains("missing JSON key"))
            #expect(missing.message.contains(key))
            #expect(missing.message.contains(accessor.expectedType))

            guard let wrongType = acceptanceFailure(
                from: [key: accessor.wrongValue],
                read: accessor.read
            )
            else {
                Issue.record("\(accessor.expectedType) accessor accepted a wrong JSON type")
                continue
            }

            if case .failed = wrongType.kind {
                Issue.record("a wrong JSON type must be UNKNOWN")
            }
            #expect(wrongType.message.contains("wrong JSON type"))
            #expect(wrongType.message.contains(key))
            #expect(wrongType.message.contains(accessor.expectedType))
            #expect(wrongType.message != missing.message)
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
            #expect(error.message.contains("wrong JSON type"))
            #expect(error.message.contains("metal_version"))
            #expect(error.message.contains("string"))
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
        let temporary = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-import-parser-\(UUID().uuidString).swift")
        defer { try? FileManager.default.removeItem(at: temporary) }
        let source = ####"""
        let explanation = "ignore; import FakeString"
        // ignore; import FakeComment
        let raw = #"literal \#"#; import FakeRaw"#
        let multiline = #"""
        literal \#"""#; import FakeMultiline
        """#
        let extended = #/foo\/#; import FakeExtended/#
        let bare = /foo; import FakeBare/
        prefix operator /
        prefix func / (value: Int) -> Int { value }
        let customPrefix = /1
        let afterPlus = 1 + /foo; import FakeAfterPlus/
        let quotient = 8 / 2
        #if(os(macOS))
        import MLXLMCommon
        #elseif os(iOS)
        import UIKit
        #else
        import Foundation
        #endif
        import `MLX`
        @_exported import NIO
        internal import AppKit
        package import struct NIOCore.ByteBuffer
        import SwamaCore; import ArgumentParser
        """####
        try Data(source.utf8).write(to: temporary)

        let declarations = try compilerImportedModules(
            in: temporary,
            developerDirectory: URL(fileURLWithPath: "/Applications/Xcode.app/Contents/Developer")
        )
        #expect(declarations.map(\.module) == [
            "MLXLMCommon",
            "UIKit",
            "Foundation",
            "MLX",
            "NIO",
            "AppKit",
            "NIOCore",
            "SwamaCore",
            "ArgumentParser"
        ])
        #expect(declarations.allSatisfy { $0.line > 0 })
        if declarations.count == 9 {
            #expect(declarations[7].line == declarations[8].line)
        }
        #expect(throws: AcceptanceFailure.self) {
            _ = try parsedSwiftImports(source: "import", file: temporary)
        }
        #expect(throws: AcceptanceFailure.self) {
            _ = try compilerImportedModules(
                in: temporary,
                developerDirectory: URL(fileURLWithPath: "/nonexistent/swama-xcode")
            )
        }
    }

    @Test func compilerImportParserAcceptsTheRealDiagnosticsFile() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let file = paths.package
            .appendingPathComponent("Sources/SwamaKit/Diagnostics/SwamaDiagnostics.swift")
        let declarations = try compilerImportedModules(
            in: file,
            developerDirectory: URL(fileURLWithPath: "/Applications/Xcode.app/Contents/Developer")
        )
        #expect(declarations.map(\.module) == ["CryptoKit", "Darwin", "Foundation"])
        #expect(declarations.map(\.line) == [1, 2, 3])
    }

    @Test func compilerPublicAPIGateRejectsUpstreamTypesAndConformances() throws {
        let graph: JSONObject = [
            "module": ["name": "SwamaCore"],
            "symbols": [
                symbolGraphSymbol(
                    precise: "s:9SwamaCore6EngineC5model11MLXLMCommon14ModelContainerCvp",
                    path: ["Engine", "model"],
                    declaration: [
                        typeFragment("Engine", precise: "s:9SwamaCore6EngineC"),
                        typeFragment("ModelContainer", precise: "s:11MLXLMCommon14ModelContainerC")
                    ]
                ),
                symbolGraphSymbol(
                    precise: "s:9SwamaCore6EngineC5cacheSay11MLXLMCommon7KVCache_pGvp",
                    path: ["Engine", "cache"],
                    declaration: [
                        typeFragment("KVCache", precise: "s:11MLXLMCommon7KVCacheP")
                    ]
                ),
                symbolGraphSymbol(
                    precise: "s:9SwamaCore7PayloadV4data10Foundation4DataVvp",
                    path: ["Payload", "data"],
                    declaration: [
                        typeFragment("Data", precise: "s:10Foundation4DataV")
                    ]
                ),
                symbolGraphSymbol(
                    precise: "s:9SwamaCore7PayloadV",
                    path: ["Payload"],
                    declaration: [
                        typeFragment("Payload", precise: "s:9SwamaCore7PayloadV")
                    ]
                ),
                symbolGraphSymbol(
                    precise: "s:9SwamaCore18externalMemberTestyyF",
                    path: ["External", "member"],
                    declaration: [["kind": "identifier", "spelling": "member"]]
                )
            ],
            "relationships": [
                [
                    "kind": "conformsTo",
                    "source": "s:9SwamaCore7PayloadV",
                    "target": "s:12ForeignTypes15ForeignProtocolP",
                    "targetFallback": "ForeignTypes.ForeignProtocol"
                ],
                [
                    "kind": "memberOf",
                    "source": "s:9SwamaCore18externalMemberTestyyF",
                    "target": "s:8Upstream8ExternalV",
                    "targetFallback": "Upstream.External"
                ]
            ]
        ]

        let report = try analyzePublicAPISymbolGraphs(
            [graph],
            target: "SwamaCore",
            allowedModules: ["Swift", "Foundation", "SwamaCore"]
        )
        #expect(report["passed"] as? Bool == false)
        let violations = (report["violations"] as? [JSONObject]) ?? []
        let modules = Set(violations.compactMap { $0["module"] as? String })
        #expect(modules == ["ForeignTypes", "MLXLMCommon", "Upstream"])
        #expect(Set(violations.compactMap { ($0["path"] as? [String])?.joined(separator: ".") }) == [
            "Engine.cache",
            "Engine.model",
            "External.member",
            "Payload"
        ])
        #expect((report["manifest_sha256"] as? String)?.count == 64)

        let contradictoryPrecise = "s:11MLXLMCommon14ModelContainerC"
        let contradictoryReport = try analyzePublicAPISymbolGraphs(
            [[
                "module": ["name": "SwamaCore"],
                "symbols": [
                    symbolGraphSymbol(
                        precise: contradictoryPrecise,
                        path: ["ContradictoryContainer"],
                        declaration: [
                            typeFragment("ModelContainer", precise: contradictoryPrecise)
                        ]
                    )
                ],
                "relationships": []
            ]],
            target: "SwamaCore",
            allowedModules: ["Swift", "Foundation", "SwamaCore"]
        )
        #expect(try contradictoryReport.boolean("passed") == false)
        #expect(try contradictoryReport.array("violations").contains { value in
            (value as? JSONObject)?["module"] as? String == "MLXLMCommon"
        })
    }

    @Test func compilerSymbolGraphIncludesExtensionBlocks() {
        let swift = URL(fileURLWithPath: "/toolchain/swift")
        let build = symbolGraphBuildCommand(
            package: URL(fileURLWithPath: "/package"),
            scratch: URL(fileURLWithPath: "/scratch"),
            target: "SwamaCore",
            swift: swift
        )
        #expect(build.contains("--target"))
        #expect(build.contains("SwamaCore"))
        #expect(!build.contains("dump-symbol-graph"))

        let extract = symbolGraphExtractCommand(
            extractor: URL(fileURLWithPath: "/toolchain/swift-symbolgraph-extract"),
            target: "SwamaCore",
            targetTriple: "arm64-apple-macosx15.4",
            sdk: URL(fileURLWithPath: "/sdk"),
            modules: URL(fileURLWithPath: "/modules"),
            output: URL(fileURLWithPath: "/output")
        )
        #expect(extract.contains("-emit-extension-block-symbols"))
        #expect(extract.contains("-skip-synthesized-members"))
        #expect(extract.contains("public"))
        #expect(extract.contains("SwamaCore"))

        let validTarget =
            #"{"target":{"triple":"arm64-apple-macosx15.4","unversionedTriple":"arm64-apple-macosx","platform":"macosx","arch":"arm64"}}"#
        #expect((try? swiftTargetTriple(validTarget))
            == "arm64-apple-macosx15.4"
        )
        for malformed in [
            "not-json",
            #"{"target":{}}"#,
            #"{"target":{"triple":"","unversionedTriple":"arm64-apple-macosx","platform":"macosx","arch":"arm64"}}"#,
            #"{"target":{"triple":" arm64-apple-macosx15.4 ","unversionedTriple":"arm64-apple-macosx","platform":"macosx","arch":"arm64"}}"#,
            #"{"target":{"triple":"arm64-apple15.4","unversionedTriple":"arm64-apple","platform":"macosx","arch":"arm64"}}"#,
            #"{"target":{"triple":"arm64-apple-macosx15.4","unversionedTriple":"x86_64-apple-macosx","platform":"macosx","arch":"arm64"}}"#,
            #"{"target":{"triple":"arm64-apple-ios15.4","unversionedTriple":"arm64-apple-ios","platform":"ios","arch":"arm64"}}"#,
            #"{"target":{"triple":"mips-apple-macosx15.4","unversionedTriple":"mips-apple-macosx","platform":"macosx","arch":"mips"}}"#,
            #"{"target":{"triple":42,"unversionedTriple":"arm64-apple-macosx","platform":"macosx","arch":"arm64"}}"#
        ] {
            #expect(throws: AcceptanceFailure.self) {
                _ = try swiftTargetTriple(malformed)
            }
        }
    }

    @Test func compilerSymbolGraphBuildIsScopedToTheCoreTarget() throws {
        let temporary = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-target-symbol-graph-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }
        let package = temporary.appendingPathComponent("swama")
        let foreignPackage = temporary.appendingPathComponent("ForeignKit")
        try FileManager.default.createDirectory(
            at: package.appendingPathComponent("Sources/SwamaCore"),
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: package.appendingPathComponent("Tests/BrokenTests"),
            withIntermediateDirectories: true
        )
        try FileManager.default.createDirectory(
            at: foreignPackage.appendingPathComponent("Sources/ForeignKit"),
            withIntermediateDirectories: true
        )
        try Data("""
        // swift-tools-version: 6.2
        import PackageDescription

        let package = Package(
            name: "ForeignKit",
            platforms: [.macOS("15.4")],
            products: [.library(name: "ForeignKit", targets: ["ForeignKit"])],
            targets: [.target(name: "ForeignKit")]
        )
        """.utf8).write(to: foreignPackage.appendingPathComponent("Package.swift"))
        try Data("public struct ExternalType {}\n".utf8).write(
            to: foreignPackage.appendingPathComponent("Sources/ForeignKit/ForeignKit.swift")
        )
        try Data("""
        // swift-tools-version: 6.2
        import PackageDescription

        let package = Package(
            name: "TargetScopedGraph",
            platforms: [.macOS("15.4")],
            products: [.library(name: "SwamaCore", targets: ["SwamaCore"])],
            dependencies: [.package(path: "../ForeignKit")],
            targets: [
                .target(
                    name: "SwamaCore",
                    dependencies: [.product(name: "ForeignKit", package: "ForeignKit")]
                ),
                .testTarget(name: "BrokenTests", dependencies: ["SwamaCore"])
            ]
        )
        """.utf8).write(to: package.appendingPathComponent("Package.swift"))
        try Data("""
        import ForeignKit

        extension ExternalType {
            public func leaked() {}
        }
        """.utf8).write(
            to: package.appendingPathComponent("Sources/SwamaCore/SwamaCore.swift")
        )
        try Data("let broken = MissingType()\n".utf8).write(
            to: package.appendingPathComponent("Tests/BrokenTests/BrokenTests.swift")
        )

        let contract = try AcceptanceContract.load(
            from: repositoryRoot.appendingPathComponent("Tools/SwamaAcceptance/contract.json")
        )
        let report = try compilerPublicAPIReport(
            target: "SwamaCore",
            paths: WorkspacePaths(repository: temporary),
            developerDirectory: URL(fileURLWithPath: "/Applications/Xcode.app/Contents/Developer"),
            contract: contract.coreGuards
        )
        #expect(try report.boolean("passed") == false)
        #expect(try report.integer("graph_count") == 2)
        #expect(try report.array("violations").contains { value in
            guard let violation = value as? JSONObject else {
                return false
            }

            return violation["module"] as? String == "ForeignKit"
                && violation["source"] as? String == "extensionTo"
        })

        let developerDirectory = URL(fileURLWithPath: "/Applications/Xcode.app/Contents/Developer")
        let swift = developerDirectory.appendingPathComponent(
            "Toolchains/XcodeDefault.xctoolchain/usr/bin/swift"
        )
        let brokenTestBuild = try runCommand(
            [swift.path, "test", "--package-path", package.path],
            currentDirectory: package,
            environment: developerEnvironment(developerDirectory),
            timeout: 60,
            sampleMemory: false
        )
        #expect(brokenTestBuild.returnCode != 0)
    }

    @Test func compilerPublicAPIGateRejectsMalformedSymbolGraphEntries() throws {
        let validSymbol = symbolGraphSymbol(
            precise: "s:9SwamaCore7PayloadV",
            path: ["Payload"],
            declaration: [typeFragment("Payload", precise: "s:9SwamaCore7PayloadV")]
        )
        var malformedFragmentSymbol = validSymbol
        malformedFragmentSymbol["declarationFragments"] = [42]
        var unqualifiedPreciseSymbol = validSymbol
        unqualifiedPreciseSymbol["declarationFragments"] = [[
            "kind": "typeIdentifier",
            "spelling": "ForeignType",
            "preciseIdentifier": "ForeignType"
        ]]
        var truncatedPreciseSymbol = validSymbol
        truncatedPreciseSymbol["declarationFragments"] = [[
            "kind": "typeIdentifier",
            "spelling": "Foo",
            "preciseIdentifier": "s:99Foo"
        ]]
        var emptySwiftPreciseSymbol = validSymbol
        emptySwiftPreciseSymbol["declarationFragments"] = [[
            "kind": "typeIdentifier",
            "spelling": "Empty",
            "preciseIdentifier": "s:"
        ]]
        var zeroLengthSwiftPreciseSymbol = validSymbol
        zeroLengthSwiftPreciseSymbol["declarationFragments"] = [[
            "kind": "typeIdentifier",
            "spelling": "Empty",
            "preciseIdentifier": "s:0"
        ]]
        var invalidSwiftPreciseSymbol = validSymbol
        invalidSwiftPreciseSymbol["declarationFragments"] = [[
            "kind": "typeIdentifier",
            "spelling": "NotReal",
            "preciseIdentifier": "s:not-real"
        ]]
        var invalidNumericPreciseSymbol = validSymbol
        invalidNumericPreciseSymbol["declarationFragments"] = [[
            "kind": "typeIdentifier",
            "spelling": "Swift",
            "preciseIdentifier": "s:5Swift?"
        ]]
        var concatenatedPreciseSymbol = validSymbol
        concatenatedPreciseSymbol["declarationFragments"] = [[
            "kind": "typeIdentifier",
            "spelling": "Int",
            "preciseIdentifier": "s:Si5Other4TypeV"
        ]]
        var unknownFragmentKindSymbol = validSymbol
        unknownFragmentKindSymbol["declarationFragments"] = [[
            "kind": "futureTypeReference",
            "spelling": "ModelContainer",
            "preciseIdentifier": "s:11MLXLMCommon14ModelContainerC"
        ]]
        var unknownAccessSymbol = validSymbol
        unknownAccessSymbol["accessLevel"] = "futurePublic"
        var extensionWithoutTargetSymbol = validSymbol
        extensionWithoutTargetSymbol["identifier"] = [
            "precise": "s:e:s:10ForeignKit12ExternalTypeV9SwamaCoreE",
            "interfaceLanguage": "swift"
        ]
        extensionWithoutTargetSymbol["kind"] = [
            "identifier": "swift.extension",
            "displayName": "Extension"
        ]
        let extensionIdentifier = "s:e:s:10ForeignKit12ExternalTypeV9SwamaCoreE"
        var publicExtensionSymbol = extensionWithoutTargetSymbol
        publicExtensionSymbol["identifier"] = [
            "precise": extensionIdentifier,
            "interfaceLanguage": "swift"
        ]
        publicExtensionSymbol["declarationFragments"] = [
            ["kind": "keyword", "spelling": "extension"],
            ["kind": "text", "spelling": " ExternalType"]
        ]
        var internalExtensionSymbol = publicExtensionSymbol
        internalExtensionSymbol["accessLevel"] = "internal"
        var opaqueFragmentSymbol = validSymbol
        opaqueFragmentSymbol["declarationFragments"] = [[
            "kind": "typeIdentifier",
            "spelling": "ExternalType",
            "preciseIdentifier": extensionIdentifier
        ]]
        let publicMember = symbolGraphSymbol(
            precise: "s:9SwamaCore11publicMemberyyF",
            path: ["publicMember"],
            declaration: [["kind": "identifier", "spelling": "publicMember"]]
        )
        let validExtensionTarget: JSONObject = [
            "kind": "extensionTo",
            "source": extensionIdentifier,
            "target": "s:10ForeignKit12ExternalTypeV",
            "targetFallback": "ForeignKit.ExternalType"
        ]
        let malformedGraphs: [JSONObject] = [
            ["module": ["name": "SwamaCore"], "symbols": [42], "relationships": []],
            ["module": ["name": "SwamaCore"], "symbols": [validSymbol], "relationships": [42]],
            [
                "module": ["name": "SwamaCore"],
                "symbols": [malformedFragmentSymbol],
                "relationships": []
            ],
            ["module": ["name": "SwamaCore"], "symbols": [unqualifiedPreciseSymbol], "relationships": []],
            ["module": ["name": "SwamaCore"], "symbols": [truncatedPreciseSymbol], "relationships": []],
            ["module": ["name": "SwamaCore"], "symbols": [emptySwiftPreciseSymbol], "relationships": []],
            ["module": ["name": "SwamaCore"], "symbols": [zeroLengthSwiftPreciseSymbol], "relationships": []],
            ["module": ["name": "SwamaCore"], "symbols": [invalidSwiftPreciseSymbol], "relationships": []],
            ["module": ["name": "SwamaCore"], "symbols": [invalidNumericPreciseSymbol], "relationships": []],
            ["module": ["name": "SwamaCore"], "symbols": [concatenatedPreciseSymbol], "relationships": []],
            ["module": ["name": "SwamaCore"], "symbols": [unknownFragmentKindSymbol], "relationships": []],
            ["module": ["name": "SwamaCore"], "symbols": [unknownAccessSymbol], "relationships": []],
            ["module": ["name": "SwamaCore"], "symbols": [extensionWithoutTargetSymbol], "relationships": []],
            [
                "module": ["name": "SwamaCore"],
                "symbols": [publicExtensionSymbol],
                "relationships": [[
                    "kind": "extensionTo",
                    "source": extensionIdentifier,
                    "target": extensionIdentifier
                ]]
            ],
            [
                "module": ["name": "SwamaCore"],
                "symbols": [publicExtensionSymbol, opaqueFragmentSymbol],
                "relationships": [validExtensionTarget]
            ],
            [
                "module": ["name": "SwamaCore"],
                "symbols": [publicExtensionSymbol, validSymbol],
                "relationships": [
                    validExtensionTarget,
                    [
                        "kind": "conformsTo",
                        "source": "s:9SwamaCore7PayloadV",
                        "target": extensionIdentifier
                    ]
                ]
            ],
            [
                "module": ["name": "SwamaCore"],
                "symbols": [internalExtensionSymbol, publicMember],
                "relationships": [
                    validExtensionTarget,
                    [
                        "kind": "memberOf",
                        "source": "s:9SwamaCore11publicMemberyyF",
                        "target": extensionIdentifier
                    ]
                ]
            ],
            [
                "module": ["name": "SwamaCore"],
                "symbols": [validSymbol],
                "relationships": [[
                    "kind": "conformsTo",
                    "source": "s:9SwamaCore7PayloadV",
                    "target": "ForeignProtocol"
                ]]
            ],
            [
                "module": ["name": "SwamaCore"],
                "symbols": [validSymbol],
                "relationships": [[
                    "kind": "conformsTo",
                    "source": "s:9SwamaCore7PayloadV",
                    "target": "s:not-real",
                    "targetFallback": "Swift.Encodable"
                ]]
            ],
            [
                "module": ["name": "SwamaCore"],
                "symbols": [validSymbol],
                "relationships": [[
                    "kind": "futureConformance",
                    "source": "s:9SwamaCore7PayloadV",
                    "target": "s:11MLXLMCommon14ModelContainerC"
                ]]
            ]
        ]

        for graph in malformedGraphs {
            #expect(throws: AcceptanceFailure.self) {
                _ = try analyzePublicAPISymbolGraphs(
                    [graph],
                    target: "SwamaCore",
                    allowedModules: ["Swift", "Foundation", "SwamaCore"]
                )
            }
        }

        for (spelling, precise) in [
            ("Int", "s:Si"),
            ("Error", "s:s5ErrorP"),
            ("Hashable", "s:SH"),
            ("FloatingPoint", "s:SF"),
            ("BidirectionalCollection", "s:SK"),
            ("Comparable", "s:SL"),
            ("MutableCollection", "s:SM"),
            ("ClosedRange", "s:SN"),
            ("Sequence", "s:ST"),
            ("Numeric", "s:Sj"),
            ("RandomAccessCollection", "s:Sk"),
            ("Collection", "s:Sl"),
            ("RangeReplaceableCollection", "s:Sm"),
            ("IteratorProtocol", "s:St"),
            ("Strideable", "s:Sx"),
            ("BinaryInteger", "s:Sz"),
            ("Range", "s:Sn"),
            ("Set", "s:Sh"),
            ("Void", "s:s4Voida"),
            ("TimeInterval", "c:@T@NSTimeInterval")
        ] {
            var validSwiftSymbol = validSymbol
            validSwiftSymbol["declarationFragments"] = [[
                "kind": "typeIdentifier",
                "spelling": spelling,
                "preciseIdentifier": precise
            ]]
            let report = try analyzePublicAPISymbolGraphs(
                [["module": ["name": "SwamaCore"], "symbols": [validSwiftSymbol], "relationships": []]],
                target: "SwamaCore",
                allowedModules: ["Swift", "Foundation", "SwamaCore"]
            )
            #expect(try report.boolean("passed"))
            #expect(try report.array("violations").isEmpty)
        }

        var validAttributeSymbol = validSymbol
        validAttributeSymbol["declarationFragments"] = [
            ["kind": "attribute", "spelling": "MainActor", "preciseIdentifier": "s:ScM"],
            typeFragment("Payload", precise: "s:9SwamaCore7PayloadV")
        ]
        let validAttributeReport = try analyzePublicAPISymbolGraphs(
            [["module": ["name": "SwamaCore"], "symbols": [validAttributeSymbol], "relationships": []]],
            target: "SwamaCore",
            allowedModules: ["Swift", "Foundation", "SwamaCore"]
        )
        #expect(try validAttributeReport.boolean("passed"))
        #expect(try validAttributeReport.array("violations").isEmpty)

        var externalAttributeSymbol = validSymbol
        externalAttributeSymbol["declarationFragments"] = [
            [
                "kind": "attribute",
                "spelling": "ExternalWrapper",
                "preciseIdentifier": "s:8Upstream15ExternalWrapperV"
            ],
            typeFragment("Payload", precise: "s:9SwamaCore7PayloadV")
        ]
        let externalAttributeReport = try analyzePublicAPISymbolGraphs(
            [["module": ["name": "SwamaCore"], "symbols": [externalAttributeSymbol], "relationships": []]],
            target: "SwamaCore",
            allowedModules: ["Swift", "Foundation", "SwamaCore"]
        )
        #expect(try externalAttributeReport.boolean("passed") == false)
        #expect(try externalAttributeReport.array("violations").contains { value in
            (value as? JSONObject)?["module"] as? String == "Upstream"
        })

        for (precise, fallback) in [
            ("s:SE", "Swift.Encodable"),
            ("s:SQ", "Swift.Equatable"),
            ("s:SY", "Swift.RawRepresentable"),
            ("s:ScA", "_Concurrency.Actor"),
            ("s:Se", "Swift.Decodable")
        ] {
            let report = try analyzePublicAPISymbolGraphs(
                [[
                    "module": ["name": "SwamaCore"],
                    "symbols": [validSymbol],
                    "relationships": [[
                        "kind": "conformsTo",
                        "source": "s:9SwamaCore7PayloadV",
                        "target": precise,
                        "targetFallback": fallback
                    ]]
                ]],
                target: "SwamaCore",
                allowedModules: ["Swift", "Foundation", "SwamaCore"]
            )
            #expect(try report.boolean("passed"))
            #expect(try report.array("violations").isEmpty)
        }

        let preciseIdentifiers = ["s:SN", "s:5Other4TypeV"]
        let mangledNames = ["$sSN", "$s5Other4TypeV"]
        let validDemanglerOutput = """
        Demangling for $sSN
        kind=Global
          kind=Structure
            kind=Module, text="Swift"
            kind=Identifier, text="ClosedRange"

        Demangling for $s5Other4TypeV
        kind=Global
          kind=Structure
            kind=Module, text="Other"
            kind=Identifier, text="Type"


        """
        let parsedModules = try parseBatchDemanglerOutput(
            validDemanglerOutput,
            preciseIdentifiers: preciseIdentifiers,
            mangledNames: mangledNames
        )
        #expect(parsedModules[0] == "Swift")
        #expect(parsedModules[1] == "Other")
        for malformed in [
            validDemanglerOutput.replacingOccurrences(
                of: "Demangling for $sSN",
                with: "Demangling for $s5Other4TypeV"
            ),
            "\n" + validDemanglerOutput,
            validDemanglerOutput.trimmingCharacters(in: .newlines),
            validDemanglerOutput + "noise\n"
        ] {
            #expect(throws: AcceptanceFailure.self) {
                _ = try parseBatchDemanglerOutput(
                    malformed,
                    preciseIdentifiers: preciseIdentifiers,
                    mangledNames: mangledNames
                )
            }
        }

        let invalidModules = try parseBatchDemanglerOutput(
            "Demangling for $s5Swift?\n<<NULL>>\nDemangling for $sXX\n<<NULL>>\n",
            preciseIdentifiers: ["s:5Swift?", "s:XX"],
            mangledNames: ["$s5Swift?", "$sXX"]
        )
        #expect(invalidModules == [nil, nil])

        let concatenatedModules = try parseBatchDemanglerOutput(
            """
            Demangling for $sSi5Other4TypeV
            kind=Global
              kind=Structure
                kind=Module, text="Swift"
                kind=Identifier, text="Int"
              kind=Structure
                kind=Module, text="Other"
                kind=Identifier, text="Type"


            """,
            preciseIdentifiers: ["s:Si5Other4TypeV"],
            mangledNames: ["$sSi5Other4TypeV"]
        )
        #expect(concatenatedModules == [nil])
    }

    @Test func coreBoundaryTurnsGreenWhenTheDependencyFreeTargetExists() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract)
        let report = try architectureReport(
            contract: contract.architecture,
            coreGuards: contract.coreGuards,
            stage: .coreBoundary,
            paths: paths
        )

        #expect(try report.boolean("passed"))
        #expect(try report.boolean("core_target_present"))
        let compiler = try report.object("compiler_public_api")
        #expect(try compiler.string("status") == "ready")
        #expect(try compiler.boolean("passed"))
        #expect(try compiler.integer("symbol_count") == 0)
        #expect(try compiler.array("symbols").isEmpty)
        #expect(try compiler.array("violations").isEmpty)
        #expect(try compiler.string("manifest_sha256")
            == "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        )
        let dependencies = try report.object("core_target_dependencies")
        #expect(try dependencies.string("status") == "ready")
        #expect(try dependencies.boolean("passed"))

        let consumerReport = try architectureReport(
            contract: contract.architecture,
            coreGuards: contract.coreGuards,
            stage: .consumerBoundary,
            paths: paths
        )
        #expect(try consumerReport.boolean("passed") == false)
        #expect(try consumerReport.object("external_consumer_boundary").boolean("passed") == false)
    }

    @Test func consumerBoundaryNamesEveryCurrentMLXDependencyAndImport() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract)
        let report = try externalConsumerBoundaryReport(
            fixture: paths.fixture,
            expectedPackage: paths.package,
            contract: contract.coreGuards
        )

        #expect(try report.boolean("passed") == false)
        #expect(try report.array("unexpected_products").contains { ($0 as? String) == "MLXLMCommon" })
        #expect(try report.array("unexpected_imports").contains { ($0 as? String) == "MLXLMCommon" })
    }

    @Test func consumerBoundaryRejectsExtraPathPackagesAndByNameTargets() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract).coreGuards
        let description: JSONObject = [
            "dependencies": [
                ["fileSystem": [["identity": "swama", "path": "/repo/swama"]]],
                ["fileSystem": [["identity": "helper", "path": "/repo/helper"]]]
            ],
            "targets": [[
                "name": "SwamaAcceptanceProbe",
                "dependencies": [
                    ["product": ["SwamaCore", "swama", NSNull(), NSNull()]],
                    ["byName": ["Helper", NSNull()]]
                ]
            ]]
        ]

        let report = try analyzeExternalConsumerPackage(
            description,
            imports: ["Foundation", "SwamaCore"],
            expectedPackagePath: "/repo/swama",
            contract: contract
        )
        #expect(try report.boolean("passed") == false)
        #expect(try report.array("unexpected_package_dependencies").contains { value in
            (value as? JSONObject)?["identity"] as? String == "helper"
        })
        #expect(try report.array("unexpected_target_dependencies").contains { value in
            (value as? JSONObject)?["kind"] as? String == "byName"
        })

        let wrongPath: JSONObject = [
            "dependencies": [[
                "fileSystem": [["identity": "swama", "path": "/tmp/unreviewed/swama"]]
            ]],
            "targets": [[
                "name": "SwamaAcceptanceProbe",
                "dependencies": [["product": ["SwamaCore", "swama", NSNull(), NSNull()]]]
            ]]
        ]
        let wrongPathReport = try analyzeExternalConsumerPackage(
            wrongPath,
            imports: ["Foundation", "SwamaCore"],
            expectedPackagePath: "/repo/swama",
            contract: contract
        )
        #expect(try wrongPathReport.boolean("passed") == false)
        #expect(try wrongPathReport.array("unexpected_package_dependencies").contains { value in
            (value as? JSONObject)?["path"] as? String == "/tmp/unreviewed/swama"
        })
    }

    @Test func consumerBoundaryRejectsMalformedManifestEntries() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract).coreGuards
        let validDependency: JSONObject = [
            "fileSystem": [["identity": "swama", "path": "/repo/swama"]]
        ]
        let validTarget: JSONObject = [
            "name": "SwamaAcceptanceProbe",
            "dependencies": [["product": ["SwamaCore", "swama", NSNull(), NSNull()]]]
        ]
        let malformed: [JSONObject] = [
            ["dependencies": [42], "targets": [validTarget]],
            ["dependencies": [["fileSystem": [42]]], "targets": [validTarget]],
            ["dependencies": [validDependency], "targets": [42]],
            [
                "dependencies": [validDependency],
                "targets": [["name": "SwamaAcceptanceProbe", "dependencies": [42]]]
            ]
        ]

        for description in malformed {
            #expect(throws: AcceptanceFailure.self) {
                _ = try analyzeExternalConsumerPackage(
                    description,
                    imports: ["Foundation", "SwamaCore"],
                    expectedPackagePath: "/repo/swama",
                    contract: contract
                )
            }
        }
    }

    @Test func targetDependencyGraphRejectsTransitiveAudioProducts() throws {
        let manifests: [String: JSONObject] = [
            "swama": [
                "dependencies": [["fileSystem": [["identity": "wrapper"]]]],
                "products": [[
                    "name": "SwamaCore",
                    "targets": ["SwamaCore"],
                    "type": ["library": ["automatic"]]
                ]],
                "targets": [
                    [
                        "name": "SwamaCore",
                        "dependencies": [["target": ["Implementation", NSNull()]]]
                    ],
                    [
                        "name": "Implementation",
                        "dependencies": [["product": ["Wrapper", "wrapper", NSNull(), NSNull()]]]
                    ]
                ]
            ],
            "wrapper": [
                "dependencies": [["fileSystem": [["identity": "audio"]]]],
                "products": [[
                    "name": "Wrapper",
                    "targets": ["Wrapper"],
                    "type": ["library": ["automatic"]]
                ]],
                "targets": [[
                    "name": "Wrapper",
                    "dependencies": [["product": ["MLXAudioCore", "audio", NSNull(), NSNull()]]]
                ]]
            ],
            "audio": [
                "dependencies": [],
                "products": [[
                    "name": "MLXAudioCore",
                    "targets": ["MLXAudioCore"],
                    "type": ["library": ["automatic"]]
                ]],
                "targets": [["name": "MLXAudioCore", "dependencies": []]]
            ]
        ]
        let report = try analyzeResolvedTargetDependencyGraph(
            rootIdentity: "swama",
            manifests: manifests,
            target: "SwamaCore",
            forbiddenProducts: ["MLXAudioCore", "MLXAudioSTT", "MLXAudioTTS"]
        )

        #expect(try report.boolean("passed") == false)
        #expect(try report.array("forbidden_products") as? [String] == ["MLXAudioCore"])
        #expect(try Set(report.array("target_dependencies").compactMap { $0 as? String }) == [
            "audio:MLXAudioCore",
            "swama:Implementation",
            "swama:SwamaCore",
            "wrapper:Wrapper"
        ])
        #expect(try Set(report.array("product_dependencies").compactMap { $0 as? String }) == [
            "audio:MLXAudioCore",
            "wrapper:Wrapper"
        ])
        #expect(try report.boolean("library_product_present"))
        #expect(try report.array("unresolved_dependencies").isEmpty)
    }

    @Test func resolvedTargetDependencyOracleReadsTheRealSwiftPMPackageGraph() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract).coreGuards
        let report = try coreTargetDependencyReport(
            target: "SwamaKit",
            paths: paths,
            developerDirectory: URL(fileURLWithPath: "/Applications/Xcode.app/Contents/Developer"),
            contract: contract
        )

        #expect(try report.boolean("passed") == false)
        #expect(try Set(report.array("forbidden_products").compactMap { $0 as? String }) == [
            "MLXAudioCore",
            "MLXAudioSTT",
            "MLXAudioTTS"
        ])
        #expect(try report.object("manifest_sha256").keys.contains("mlx-audio-swift"))
        #expect(try report.object("resolved_package_traits").array("swama") as? [String] == ["default"])
    }

    @Test func targetDependencyGraphRejectsAMissingLibraryProduct() throws {
        let missingProduct: [String: JSONObject] = [
            "swama": [
                "dependencies": [],
                "products": [],
                "targets": [["name": "SwamaCore", "dependencies": []]]
            ]
        ]
        let missingReport = try analyzeResolvedTargetDependencyGraph(
            rootIdentity: "swama",
            manifests: missingProduct,
            target: "SwamaCore",
            forbiddenProducts: ["MLXAudioCore", "MLXAudioSTT", "MLXAudioTTS"]
        )

        #expect(try missingReport.boolean("passed") == false)
        #expect(try missingReport.boolean("library_product_present") == false)
        #expect(try missingReport.string("status") == "unmet")
        #expect(try missingReport.array("forbidden_products").isEmpty)

        let miswiredProduct: [String: JSONObject] = [
            "swama": [
                "dependencies": [],
                "products": [[
                    "name": "SwamaCore",
                    "targets": ["Implementation"],
                    "type": ["library": ["automatic"]]
                ]],
                "targets": [["name": "SwamaCore", "dependencies": []]]
            ]
        ]
        let miswiredReport = try analyzeResolvedTargetDependencyGraph(
            rootIdentity: "swama",
            manifests: miswiredProduct,
            target: "SwamaCore",
            forbiddenProducts: ["MLXAudioCore", "MLXAudioSTT", "MLXAudioTTS"]
        )

        #expect(try miswiredReport.boolean("passed") == false)
        #expect(try miswiredReport.boolean("library_product_present") == false)
        #expect(try miswiredReport.array("library_product_targets") as? [String] == ["Implementation"])
    }

    @Test func targetDependencyGraphRejectsAnUnknownExplicitPackageAlias() throws {
        let manifests: [String: JSONObject] = [
            "swama": [
                "dependencies": [],
                "products": [[
                    "name": "SwamaCore",
                    "targets": ["SwamaCore"],
                    "type": ["library": ["automatic"]]
                ]],
                "targets": [[
                    "name": "SwamaCore",
                    "dependencies": [[
                        "product": ["HiddenProduct", "undeclared-package", NSNull(), NSNull()]
                    ]]
                ]]
            ]
        ]
        let report = try analyzeResolvedTargetDependencyGraph(
            rootIdentity: "swama",
            manifests: manifests,
            target: "SwamaCore",
            forbiddenProducts: ["MLXAudioCore", "MLXAudioSTT", "MLXAudioTTS"]
        )

        #expect(try report.boolean("passed") == false)
        #expect(try report.array("unresolved_dependencies") as? [String] == [
            "unresolved product swama:HiddenProduct"
        ])
    }

    @Test func targetDependencyGraphRejectsMalformedNestedEvidence() throws {
        let malformed: [[String: JSONObject]] = [
            [
                "swama": [
                    "dependencies": [],
                    "products": [[
                        "name": "SwamaCore",
                        "targets": ["SwamaCore", NSNull()],
                        "type": ["library": ["automatic"]]
                    ]],
                    "targets": [["name": "SwamaCore", "dependencies": []]]
                ]
            ],
            [
                "swama": [
                    "dependencies": [],
                    "products": [[
                        "name": "SwamaCore",
                        "targets": ["SwamaCore"],
                        "type": ["library": ["automatic"]]
                    ]],
                    "targets": [[
                        "name": "SwamaCore",
                        "dependencies": [["product": ["Hidden", 42, NSNull(), NSNull()]]]
                    ]]
                ]
            ],
            [
                "swama": [
                    "dependencies": [],
                    "products": [[
                        "name": "SwamaCore",
                        "targets": ["SwamaCore"],
                        "type": ["library": ["automatic"]]
                    ]],
                    "targets": [[
                        "name": "SwamaCore",
                        "dependencies": [["product": ["Hidden", NSNull(), NSNull(), "active"]]]
                    ]]
                ]
            ],
            [
                "swama": [
                    "dependencies": [["fileSystem": [[
                        "identity": "dependency",
                        "nameForTargetDependencyResolutionOnly": 42
                    ]]]],
                    "products": [[
                        "name": "SwamaCore",
                        "targets": ["SwamaCore"],
                        "type": ["library": ["automatic"]]
                    ]],
                    "targets": [["name": "SwamaCore", "dependencies": []]]
                ]
            ],
            [
                "swama": [
                    "dependencies": [],
                    "products": [[
                        "name": "SwamaCore",
                        "targets": ["SwamaCore"],
                        "type": ["library": []]
                    ]],
                    "targets": [["name": "SwamaCore", "dependencies": []]]
                ]
            ],
            [
                "swama": [
                    "dependencies": [],
                    "products": [[
                        "name": "SwamaCore",
                        "targets": ["SwamaCore"],
                        "type": ["executable": ["unexpected"]]
                    ]],
                    "targets": [["name": "SwamaCore", "dependencies": []]]
                ]
            ],
            [
                "swama": [
                    "dependencies": [["fileSystem": [["identity": "dependency"]]]],
                    "products": [[
                        "name": "SwamaCore",
                        "targets": ["SwamaCore"],
                        "type": ["library": ["automatic"]]
                    ]],
                    "targets": [[
                        "name": "SwamaCore",
                        "dependencies": [["product": ["Hidden", "", NSNull(), NSNull()]]]
                    ]]
                ],
                "dependency": [
                    "dependencies": [],
                    "products": [[
                        "name": "Hidden",
                        "targets": ["Hidden"],
                        "type": ["library": ["automatic"]]
                    ]],
                    "targets": [["name": "Hidden", "dependencies": []]]
                ]
            ]
        ]

        for manifests in malformed {
            #expect(throws: AcceptanceFailure.self) {
                _ = try analyzeResolvedTargetDependencyGraph(
                    rootIdentity: "swama",
                    manifests: manifests,
                    target: "SwamaCore",
                    forbiddenProducts: ["MLXAudioCore", "MLXAudioSTT", "MLXAudioTTS"]
                )
            }
        }
    }

    @Test func targetDependencyGraphResolvesCustomPackageAliases() throws {
        let manifests: [String: JSONObject] = [
            "swama": [
                "dependencies": [["fileSystem": [[
                    "identity": "dependency",
                    "nameForTargetDependencyResolutionOnly": "ChosenAlias"
                ]]]],
                "products": [[
                    "name": "SwamaCore",
                    "targets": ["SwamaCore"],
                    "type": ["library": ["automatic"]]
                ]],
                "targets": [[
                    "name": "SwamaCore",
                    "dependencies": [[
                        "product": ["HiddenProduct", "ChosenAlias", NSNull(), NSNull()]
                    ]]
                ]]
            ],
            "dependency": [
                "dependencies": [["fileSystem": [["identity": "audio"]]]],
                "products": [[
                    "name": "HiddenProduct",
                    "targets": ["HiddenTarget"],
                    "type": ["library": ["automatic"]]
                ]],
                "targets": [[
                    "name": "HiddenTarget",
                    "dependencies": [[
                        "product": ["MLXAudioCore", "audio", NSNull(), NSNull()]
                    ]]
                ]]
            ],
            "audio": [
                "dependencies": [],
                "products": [[
                    "name": "MLXAudioCore",
                    "targets": ["MLXAudioCore"],
                    "type": ["library": ["automatic"]]
                ]],
                "targets": [["name": "MLXAudioCore", "dependencies": []]]
            ]
        ]
        let report = try analyzeResolvedTargetDependencyGraph(
            rootIdentity: "swama",
            manifests: manifests,
            target: "SwamaCore",
            forbiddenProducts: ["MLXAudioCore", "MLXAudioSTT", "MLXAudioTTS"]
        )

        #expect(try report.boolean("passed") == false)
        #expect(try report.array("forbidden_products") as? [String] == ["MLXAudioCore"])
        #expect(try report.array("unresolved_dependencies").isEmpty)
        #expect(try report.array("product_dependencies").contains { value in
            (value as? String) == "dependency:HiddenProduct"
        })
    }

    @Test func inlinableAndUsableFromInlineCannotOpenUncheckedReachability() throws {
        let temporary = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-inline-boundary-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        try Data(
            """
            @inlinable public func leak() {}
            @available(macOS 15.4, *) @usableFromInline internal let hidden = 1
            // @inlinable public func commentOnly() {}
            let example = "@usableFromInline"

            """.utf8
        ).write(to: temporary.appendingPathComponent("Leak.swift"))

        let hits = try publicReachabilityAttributeHits(in: temporary, repository: temporary)
        #expect(hits.count == 2)
        #expect(Set(hits.compactMap { $0["attribute"] as? String }) == ["@inlinable", "@usableFromInline"])
    }

    @Test func paritySchemaRejectsMissingRouteAndUnknownEvents() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract).coreGuards.parity
        let valid: JSONObject = [
            "schema_version": contract.schemaVersion,
            "case_id": "fixed-case",
            "route": "core",
            "events": [["sequence": 0, "type": "text_delta", "text": "ok"]],
            "terminal": [
                "kind": "response",
                "output": "ok",
                "tool_calls": [],
                "finish_reason": "completed",
                "usage": ["prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2]
            ]
        ]
        #expect(throws: Never.self) { _ = try validateParityRecord(valid, contract: contract) }

        var missingRoute = valid
        missingRoute.removeValue(forKey: "route")
        #expect(throws: AcceptanceFailure.self) {
            _ = try validateParityRecord(missingRoute, contract: contract)
        }

        var unknownEvent = valid
        unknownEvent["events"] = [["sequence": 0, "type": "mystery"]]
        #expect(throws: AcceptanceFailure.self) {
            _ = try validateParityRecord(unknownEvent, contract: contract)
        }

        var validToolCalls = valid
        validToolCalls["events"] = [[
            "sequence": 0,
            "type": "tool_call",
            "name": "lookup",
            "arguments": ["query": "hello"]
        ]]
        validToolCalls["terminal"] = [
            "kind": "response",
            "output": "",
            "tool_calls": [["name": "lookup", "arguments": ["query": "hello"]]],
            "finish_reason": "tool_call",
            "usage": ["prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2]
        ]
        #expect(throws: Never.self) {
            _ = try validateParityRecord(validToolCalls, contract: contract)
        }

        var nullEventArguments = validToolCalls
        nullEventArguments["events"] = [[
            "sequence": 0,
            "type": "tool_call",
            "name": "lookup",
            "arguments": NSNull()
        ]]
        #expect(throws: AcceptanceFailure.self) {
            _ = try validateParityRecord(nullEventArguments, contract: contract)
        }

        var scalarTerminalArguments = validToolCalls
        scalarTerminalArguments["terminal"] = [
            "kind": "response",
            "output": "",
            "tool_calls": [["name": "lookup", "arguments": 42]],
            "finish_reason": "tool_call",
            "usage": ["prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2]
        ]
        #expect(throws: AcceptanceFailure.self) {
            _ = try validateParityRecord(scalarTerminalArguments, contract: contract)
        }
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

    @Test func fixtureBuildConsumesContractRetryAndKeepsTimeoutUnknown() throws {
        let temporary = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-fixture-stage-retry-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        let attemptFile = temporary.appendingPathComponent("attempt-count")
        let scratch = try prepareAcceptanceBuildScratch(
            root: temporary.appendingPathComponent("scratch"),
            identity: "fixture-stage-test"
        )
        let contract = BuildContract(
            productTimeoutSeconds: 1,
            externalFixtureTimeoutSeconds: 0.05,
            externalFixtureTimeoutRetryLimit: 1
        )

        do {
            _ = try buildExternalFixture(
                contract: contract,
                paths: .init(repository: temporary),
                scratch: scratch,
                environment: ProcessInfo.processInfo.environment,
                command: [
                    "/bin/sh",
                    "-c",
                    "count=$(cat \"$1\" 2>/dev/null || echo 0); echo $((count + 1)) > \"$1\"; sleep 1",
                    "fixture-build",
                    attemptFile.path
                ]
            )
            Issue.record("the fixture build must exhaust its timeout retry")
        }
        catch let error as AcceptanceFailure {
            if case .failed = error.kind {
                Issue.record("a fixture build timeout is tool UNKNOWN, not product FAIL")
            }
            #expect(error.message.contains("external fixture build timeout after 2 attempt(s) at 0.05s each"))
        }

        let attempts = try Int(
            String(contentsOf: attemptFile, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        )
        #expect(attempts == 2)
    }

    @Test func commandTimeoutTerminatesDescendantProcesses() throws {
        let temporary = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-process-tree-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        let childPIDFile = temporary.appendingPathComponent("child-pid")
        let grandchildPIDFile = temporary.appendingPathComponent("grandchild-pid")
        let childScript = temporary.appendingPathComponent("spawn-grandchild.sh")
        let rootScript = temporary.appendingPathComponent("spawn-child.sh")
        try Data("""
        #!/bin/sh
        sleep 5 &
        echo $! > "$1"
        wait
        """.utf8).write(to: childScript)
        try Data("""
        #!/bin/sh
        /bin/sh "$2" "$3" &
        echo $! > "$1"
        wait
        """.utf8).write(to: rootScript)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: rootScript.path
        )
        var childPID: pid_t?
        var grandchildPID: pid_t?
        defer {
            if let childPID {
                _ = kill(childPID, SIGKILL)
            }
            if let grandchildPID {
                _ = kill(grandchildPID, SIGKILL)
            }
            try? FileManager.default.removeItem(at: temporary)
        }

        #expect(throws: AcceptanceFailure.self) {
            _ = try runCommand(
                [
                    "/bin/sh",
                    rootScript.path,
                    childPIDFile.path,
                    childScript.path,
                    grandchildPIDFile.path
                ],
                currentDirectory: temporary,
                timeout: 0.5,
                sampleMemory: false,
                timeoutFailureKind: .unknown
            )
        }
        let childPIDValue = try #require(
            Int(String(contentsOf: childPIDFile, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            )
        )
        let grandchildPIDValue = try #require(
            Int(String(contentsOf: grandchildPIDFile, encoding: .utf8)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            )
        )
        childPID = pid_t(childPIDValue)
        grandchildPID = pid_t(grandchildPIDValue)
        #expect(kill(pid_t(childPIDValue), 0) == -1)
        #expect(kill(pid_t(grandchildPIDValue), 0) == -1)
    }

    @Test func contractCarriesSeparateProductAndFixtureBuildBudgets() throws {
        let paths = try WorkspacePaths.discover(explicit: repositoryRoot.path)
        let contract = try AcceptanceContract.load(from: paths.contract)
        #expect(contract.schemaVersion == 4)
        #expect(contract.build.productTimeoutSeconds == 1200)
        #expect(contract.build.externalFixtureTimeoutSeconds == 1200)
        #expect(contract.build.externalFixtureTimeoutRetryLimit == 1)
        #expect(contract.coreGuards.schemaVersion == 1)
        #expect(contract.coreGuards.parity.requiredRoutes == ["core", "cli", "http"])
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

    private func acceptanceFailure(
        from object: JSONObject,
        read: (JSONObject) throws -> Void
    ) -> AcceptanceFailure? {
        do {
            try read(object)
            return nil
        }
        catch let error as AcceptanceFailure {
            return error
        }
        catch {
            Issue.record("unexpected error: \(error)")
            return nil
        }
    }

    private func symbolGraphSymbol(
        precise: String,
        path: [String],
        declaration: [JSONObject]
    ) -> JSONObject {
        [
            "identifier": ["precise": precise, "interfaceLanguage": "swift"],
            "kind": ["identifier": "swift.property", "displayName": "Instance Property"],
            "pathComponents": path,
            "declarationFragments": declaration,
            "accessLevel": "public"
        ]
    }

    private func typeFragment(_ spelling: String, precise: String) -> JSONObject {
        ["kind": "typeIdentifier", "spelling": spelling, "preciseIdentifier": precise]
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

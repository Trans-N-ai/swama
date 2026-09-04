import Foundation

// MARK: - BuildProducts

struct BuildProducts {
    let swama: URL
    let probe: URL
    let report: JSONObject
}

// MARK: - AcceptanceBuildScratch

struct AcceptanceBuildScratch {
    let identity: String
    let root: URL
    let package: URL
    let fixture: URL
    let moduleCache: URL
}

func prepareAcceptanceBuildScratch(root: URL, identity: String) throws -> AcceptanceBuildScratch {
    let identityFile = root.appendingPathComponent("identity")
    let existingIdentity = try? String(contentsOf: identityFile, encoding: .utf8)
        .trimmingCharacters(in: .whitespacesAndNewlines)

    if existingIdentity != identity {
        if FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.removeItem(at: root)
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("\(identity)\n".utf8).write(to: identityFile, options: .atomic)
    }

    return .init(
        identity: identity,
        root: root,
        package: root.appendingPathComponent("package"),
        fixture: root.appendingPathComponent("fixture"),
        moduleCache: root.appendingPathComponent("module-cache")
    )
}

func buildProducts(
    contract: BuildContract,
    developerDirectory: URL,
    metalDeveloperDirectory: URL,
    paths: WorkspacePaths
) throws -> BuildProducts {
    let metalToolchain = try resolveMetalToolchain(
        metalDeveloperDirectory: metalDeveloperDirectory,
        paths: paths
    )
    let scratch = try acceptanceBuildScratch(
        developerDirectory: developerDirectory,
        paths: paths
    )
    var environment = try developerEnvironment(developerDirectory)
    environment["SWIFTPM_MODULECACHE_OVERRIDE"] = scratch.moduleCache.path
    let releaseBuild = try runCommand(
        [
            "xcrun",
            "swift",
            "build",
            "--package-path",
            paths.package.path,
            "--scratch-path",
            scratch.package.path,
            "--force-resolved-versions",
            "-c",
            "release",
            "-j",
            "4"
        ],
        currentDirectory: paths.repository,
        environment: environment,
        timeout: contract.productTimeoutSeconds,
        sampleMemory: false,
        timeoutFailureKind: .unknown,
        timeoutContext: "Swama release build"
    )
    guard releaseBuild.returnCode == 0 else {
        throw AcceptanceFailure.failed("Swama release build failed:\n\(commandFailureSummary(releaseBuild))")
    }

    let fixtureBuild = try buildExternalFixture(
        contract: contract,
        paths: paths,
        scratch: scratch,
        environment: environment
    )
    guard fixtureBuild.returnCode == 0 else {
        throw AcceptanceFailure
            .failed("external consumer fixture build failed:\n\(commandFailureSummary(fixtureBuild))")
    }

    let testBuild = try runCommand(
        [
            "xcrun",
            "swift",
            "build",
            "--package-path",
            paths.package.path,
            "--scratch-path",
            scratch.package.path,
            "--force-resolved-versions",
            "--build-tests",
            "-j",
            "4"
        ],
        currentDirectory: paths.repository,
        environment: environment,
        timeout: contract.productTimeoutSeconds,
        sampleMemory: false,
        timeoutFailureKind: .unknown,
        timeoutContext: "Swama test build"
    )
    guard testBuild.returnCode == 0 else {
        throw AcceptanceFailure.failed("Swama test build failed:\n\(commandFailureSummary(testBuild))")
    }

    let releaseBin = try URL(fileURLWithPath: commandOutput(
        [
            "xcrun",
            "swift",
            "build",
            "--package-path",
            paths.package.path,
            "--scratch-path",
            scratch.package.path,
            "--force-resolved-versions",
            "-c",
            "release",
            "--show-bin-path"
        ],
        currentDirectory: paths.repository,
        environment: environment
    ))
    let fixtureBin = try URL(fileURLWithPath: commandOutput(
        [
            "xcrun",
            "swift",
            "build",
            "--package-path",
            paths.fixture.path,
            "--scratch-path",
            scratch.fixture.path,
            "--force-resolved-versions",
            "-c",
            "release",
            "--show-bin-path"
        ],
        currentDirectory: paths.repository,
        environment: environment
    ))
    let debugBin = try URL(fileURLWithPath: commandOutput(
        [
            "xcrun",
            "swift",
            "build",
            "--package-path",
            paths.package.path,
            "--scratch-path",
            scratch.package.path,
            "--force-resolved-versions",
            "-c",
            "debug",
            "--show-bin-path"
        ],
        currentDirectory: paths.repository,
        environment: environment
    ))
    let swama = releaseBin.appendingPathComponent("swama")
    let probe = fixtureBin.appendingPathComponent("SwamaAcceptanceProbe")
    let testExecutable = debugBin
        .appendingPathComponent("swamaPackageTests.xctest/Contents/MacOS/swamaPackageTests")
    for required in [swama, probe, testExecutable] where !FileManager.default.fileExists(atPath: required.path) {
        throw AcceptanceFailure.unknown("expected build product is missing: \(required.path)")
    }

    let metal = try buildMetallib(
        toolchain: metalToolchain,
        packageScratchDirectory: scratch.package,
        paths: paths
    )
    defer { metal.remove() }
    let swamaMetal = try installMetallib(metal.library, nextTo: swama)
    let probeMetal = try installMetallib(metal.library, nextTo: probe)
    let testMetal = try installMetallib(metal.library, nextTo: testExecutable)

    let tests = try runCommand(
        [
            "xcrun",
            "swift",
            "test",
            "--package-path",
            paths.package.path,
            "--scratch-path",
            scratch.package.path,
            "--force-resolved-versions",
            "--skip-build",
            "-j",
            "4"
        ],
        currentDirectory: paths.repository,
        environment: environment,
        timeout: 600,
        sampleMemory: false
    )
    guard tests.returnCode == 0 else {
        throw AcceptanceFailure.failed(
            "Swift tests failed:\n\(commandFailureSummary(tests))"
        )
    }

    let testCount = firstCapture(#"Test run with (\d+) tests"#, in: tests.stdout).flatMap(Int.init) ?? -1

    return .init(
        swama: swama,
        probe: probe,
        report: try [
            "duration_ms": releaseBuild.durationMilliseconds
                + fixtureBuild.durationMilliseconds
                + testBuild.durationMilliseconds,
            "swama": swama.path,
            "probe": probe.path,
            "artifacts": [
                "swama_sha256": sha256File(swama),
                "probe_sha256": sha256File(probe),
                "test_executable_sha256": sha256File(testExecutable)
            ],
            "cache": [
                "identity": scratch.identity,
                "policy": "single-identity resumable scratch",
                "root": scratch.root.path,
                "module_cache": scratch.moduleCache.path
            ],
            "phases": [
                "swama_release": buildPhaseReport(
                    releaseBuild,
                    timeoutSeconds: contract.productTimeoutSeconds
                ),
                "external_fixture_prebuild": buildPhaseReport(
                    fixtureBuild,
                    timeoutSeconds: contract.externalFixtureTimeoutSeconds,
                    timeoutRetryLimit: contract.externalFixtureTimeoutRetryLimit
                ),
                "swama_tests": buildPhaseReport(
                    testBuild,
                    timeoutSeconds: contract.productTimeoutSeconds
                )
            ],
            "metal_build": metal.provenance,
            "metallib": swamaMetal,
            "probe_metallib": probeMetal,
            "swift_tests": [
                "count": testCount,
                "duration_ms": tests.durationMilliseconds,
                "metallib": testMetal
            ]
        ]
    )
}

func buildExternalFixture(
    contract: BuildContract,
    paths: WorkspacePaths,
    scratch: AcceptanceBuildScratch,
    environment: [String: String],
    command overrideCommand: [String]? = nil
) throws -> CommandResult {
    let command = overrideCommand ?? [
        "xcrun",
        "swift",
        "build",
        "--package-path",
        paths.fixture.path,
        "--scratch-path",
        scratch.fixture.path,
        "--force-resolved-versions",
        "-c",
        "release",
        "-j",
        "4"
    ]
    return try runCommand(
        command,
        currentDirectory: paths.repository,
        environment: environment,
        timeout: contract.externalFixtureTimeoutSeconds,
        sampleMemory: false,
        timeoutFailureKind: .unknown,
        timeoutContext: "external fixture build",
        timeoutRetryLimit: contract.externalFixtureTimeoutRetryLimit
    )
}

private func acceptanceBuildScratch(
    developerDirectory: URL,
    paths: WorkspacePaths
) throws -> AcceptanceBuildScratch {
    let environment = try developerEnvironment(developerDirectory)
    let head = try commandOutput(
        ["git", "rev-parse", "HEAD"],
        currentDirectory: paths.repository
    )
    let tree = try commandOutput(
        ["git", "rev-parse", "HEAD^{tree}"],
        currentDirectory: paths.repository
    )
    let swiftVersion = try commandOutput(
        ["xcrun", "swift", "--version"],
        currentDirectory: paths.repository,
        environment: environment
    )
    let xcodeVersion = try commandOutput(
        ["xcodebuild", "-version"],
        currentDirectory: paths.repository,
        environment: environment
    )
    let packageResolved = try sha256File(paths.package.appendingPathComponent("Package.resolved"))
    let fixtureResolved = try sha256File(paths.fixture.appendingPathComponent("Package.resolved"))
    let components = [
        "head=\(head)",
        "tree=\(tree)",
        "developer_dir=\(developerDirectory.standardizedFileURL.path)",
        "swift=\(swiftVersion)",
        "xcode=\(xcodeVersion)",
        "package_resolved=\(packageResolved)",
        "fixture_resolved=\(fixtureResolved)"
    ]
    let identity = sha256(Data(components.joined(separator: "\u{0}").utf8))
    let root = paths.repository.appendingPathComponent(".build/swama-acceptance")
    return try prepareAcceptanceBuildScratch(root: root, identity: identity)
}

private func buildPhaseReport(
    _ result: CommandResult,
    timeoutSeconds: Double,
    timeoutRetryLimit: Int = 0
) -> JSONObject {
    [
        "command": result.command,
        "duration_ms": result.durationMilliseconds,
        "timeout_seconds_per_attempt": timeoutSeconds,
        "timeout_retry_limit": timeoutRetryLimit,
        "attempt_count": result.attemptCount,
        "timeout_count": result.timeoutCount
    ]
}

func commandFailureSummary(_ result: CommandResult, maximumCharacters: Int = 4000) -> String {
    let combined = result.stdout + "\n" + result.stderr
    let errorLines = combined.split(separator: "\n")
        .filter { $0.localizedCaseInsensitiveContains("error:") }
        .suffix(20)
        .joined(separator: "\n")
    let tail = String(combined.suffix(maximumCharacters))
    guard !errorLines.isEmpty else {
        return tail
    }

    return "error lines:\n\(errorLines)\noutput tail:\n\(tail)"
}

func firstCapture(_ pattern: String, in text: String) -> String? {
    guard let expression = try? NSRegularExpression(pattern: pattern) else {
        return nil
    }

    let range = NSRange(text.startIndex ..< text.endIndex, in: text)
    guard let match = expression.firstMatch(in: text, range: range),
          match.numberOfRanges > 1,
          let capture = Range(match.range(at: 1), in: text)
    else {
        return nil
    }

    return String(text[capture])
}

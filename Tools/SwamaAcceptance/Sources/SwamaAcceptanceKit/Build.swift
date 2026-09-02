import Foundation

// MARK: - BuildProducts

struct BuildProducts {
    let swama: URL
    let probe: URL
    let report: JSONObject
}

func buildProducts(
    developerDirectory: URL,
    metalDeveloperDirectory: URL,
    paths: WorkspacePaths
) throws -> BuildProducts {
    let environment = try developerEnvironment(developerDirectory)
    let releaseBuild = try runCommand(
        [
            "xcrun",
            "swift",
            "build",
            "--package-path",
            paths.package.path,
            "--force-resolved-versions",
            "-c",
            "release",
            "-j",
            "4"
        ],
        currentDirectory: paths.repository,
        environment: environment,
        timeout: 1200,
        sampleMemory: false
    )
    guard releaseBuild.returnCode == 0 else {
        throw AcceptanceFailure.failed("Swama release build failed:\n\(commandFailureSummary(releaseBuild))")
    }

    let fixtureBuild = try runCommand(
        [
            "xcrun",
            "swift",
            "build",
            "--package-path",
            paths.fixture.path,
            "--force-resolved-versions",
            "-c",
            "release",
            "-j",
            "4"
        ],
        currentDirectory: paths.repository,
        environment: environment,
        timeout: 1200,
        sampleMemory: false
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
            "--force-resolved-versions",
            "--build-tests",
            "-j",
            "4"
        ],
        currentDirectory: paths.repository,
        environment: environment,
        timeout: 1200,
        sampleMemory: false
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

    let metal = try buildMetallib(metalDeveloperDirectory: metalDeveloperDirectory, paths: paths)
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
        report: [
            "duration_ms": releaseBuild.durationMilliseconds + fixtureBuild.durationMilliseconds,
            "swama": swama.path,
            "probe": probe.path,
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

import Foundation

// MARK: - SystemTool

enum SystemTool {
    static let systemProfiler = "/usr/sbin/system_profiler"
    static let uname = "/usr/bin/uname"
    static let swVers = "/usr/bin/sw_vers"
}

func captureProvenance(
    developerDirectory: URL,
    models: [String],
    paths: WorkspacePaths
) throws -> JSONObject {
    let environment = try developerEnvironment(developerDirectory)
    let hardwareText = try commandOutput(
        [SystemTool.systemProfiler, "SPHardwareDataType"],
        currentDirectory: paths.repository
    )
    var hardware: JSONObject = [:]
    for key in ["Model Identifier", "Chip", "Total Number of Cores", "Memory"] {
        hardware[key] = value(after: "\(key):", in: hardwareText) ?? "UNKNOWN"
    }
    let sourceDiff = try commandOutput(
        ["git", "diff", "--name-only", "origin/main", "--", "swama/Sources"],
        currentDirectory: paths.repository
    )

    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    return try [
        "captured_at_utc": formatter.string(from: Date()),
        "git": [
            "head": commandOutput(["git", "rev-parse", "HEAD"], currentDirectory: paths.repository),
            "tree": commandOutput(["git", "rev-parse", "HEAD^{tree}"], currentDirectory: paths.repository),
            "origin_main": commandOutput(["git", "rev-parse", "origin/main"], currentDirectory: paths.repository),
            "production_source_diff": sourceDiff.split(separator: "\n").map(String.init)
        ],
        "platform": [
            "system": "Darwin",
            "machine": commandOutput([SystemTool.uname, "-m"], currentDirectory: paths.repository),
            "macos": commandOutput([SystemTool.swVers, "-productVersion"], currentDirectory: paths.repository),
            "hardware": hardware
        ],
        "toolchain": [
            "developer_dir": developerDirectory.path,
            "swift": commandOutput(
                ["xcrun", "swift", "--version"],
                currentDirectory: paths.repository,
                environment: environment
            ),
            "xcode": commandOutput(
                ["xcodebuild", "-version"],
                currentDirectory: paths.repository,
                environment: environment
            ),
            "package_resolved_sha256": sha256File(paths.package.appendingPathComponent("Package.resolved"))
        ],
        "models": models.map(modelIdentity)
    ]
}

func currentInstrumentIdentity(paths: WorkspacePaths) throws -> JSONObject {
    let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL
    let harnessSources = try regularFiles(
        in: paths.harness.appendingPathComponent("Sources"),
        extensions: ["swift"]
    ) + [paths.harness.appendingPathComponent("Package.swift")]
    let harnessTests = try regularFiles(
        in: paths.harness.appendingPathComponent("Tests"),
        extensions: ["swift"]
    )
    let probeSources = try regularFiles(
        in: paths.fixture.appendingPathComponent("Sources"),
        extensions: ["swift"]
    )
    return try [
        "harness_binary_sha256": sha256File(executable),
        "harness_sources_sha256": sha256Tree(harnessSources, relativeTo: paths.repository),
        "harness_tests_sha256": sha256Tree(harnessTests, relativeTo: paths.repository),
        "core_probe_sha256": sha256Tree(probeSources, relativeTo: paths.repository),
        "fixture_manifest_sha256": sha256File(paths.fixture.appendingPathComponent("Package.swift")),
        "package_manifest_sha256": sha256File(paths.package.appendingPathComponent("Package.swift"))
    ]
}

private func modelIdentity(_ modelID: String) throws -> JSONObject {
    let directory = FileManager.default
        .homeDirectoryForCurrentUser
        .appendingPathComponent(".swama/models")
        .appendingPathComponent(modelID)
    let metadataURL = directory.appendingPathComponent(".swama-meta.json")
    guard FileManager.default.fileExists(atPath: metadataURL.path) else {
        throw AcceptanceFailure.unknown("local Swama model is absent or unowned: \(modelID)")
    }
    guard let enumerator = FileManager.default.enumerator(
        at: directory,
        includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]
    )
    else {
        throw AcceptanceFailure.unknown("cannot enumerate model directory: \(directory.path)")
    }

    let files = try enumerator.compactMap { item -> JSONObject? in
        guard let url = item as? URL, url.lastPathComponent != ".DS_Store" else { return nil }

        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true else {
            return nil
        }

        return try [
            "path": url.path.replacingOccurrences(of: directory.path + "/", with: ""),
            "bytes": values.fileSize ?? 0,
            "sha256": sha256File(url)
        ]
    }
    .sorted {
        ($0["path"] as? String ?? "") < ($1["path"] as? String ?? "")
    }

    return try [
        "id": modelID,
        "metadata": loadJSONObject(metadataURL),
        "directory": directory.path,
        "files": files
    ]
}

private func value(after prefix: String, in text: String) -> String? {
    text.split(separator: "\n")
        .map { $0.trimmingCharacters(in: .whitespaces) }
        .first { $0.hasPrefix(prefix) }?
        .dropFirst(prefix.count)
        .trimmingCharacters(in: .whitespaces)
}

import Foundation

// MARK: - MetalBuildArtifact

struct MetalBuildArtifact {
    let library: URL
    let provenance: JSONObject
    let temporaryDirectory: URL

    func remove() {
        try? FileManager.default.removeItem(at: temporaryDirectory)
    }
}

// MARK: - MetalToolchain

struct MetalToolchain {
    let developerDirectory: URL
    let environment: [String: String]
    let metalVersion: String
    let metallibVersion: String
    let metalExecutable: URL
    let metallibExecutable: URL
}

func resolveMetalToolchain(
    metalDeveloperDirectory: URL,
    paths: WorkspacePaths
) throws -> MetalToolchain {
    let environment = try developerEnvironment(metalDeveloperDirectory)
    let metalVersion = try commandOutput(
        ["xcrun", "-sdk", "macosx", "metal", "--version"],
        currentDirectory: paths.repository,
        environment: environment
    )
    let metallibVersion = try commandOutput(
        ["xcrun", "-sdk", "macosx", "metallib", "--version"],
        currentDirectory: paths.repository,
        environment: environment
    )
    let metalExecutable = try URL(fileURLWithPath: commandOutput(
        ["xcrun", "-sdk", "macosx", "-f", "metal"],
        currentDirectory: paths.repository,
        environment: environment
    )).standardizedFileURL
    let metallibExecutable = try URL(fileURLWithPath: commandOutput(
        ["xcrun", "-sdk", "macosx", "-f", "metallib"],
        currentDirectory: paths.repository,
        environment: environment
    )).standardizedFileURL
    guard FileManager.default.fileExists(atPath: metalExecutable.path),
          FileManager.default.fileExists(atPath: metallibExecutable.path)
    else {
        throw AcceptanceFailure.unknown("resolved Metal compiler executable is missing")
    }

    return .init(
        developerDirectory: metalDeveloperDirectory,
        environment: environment,
        metalVersion: metalVersion,
        metallibVersion: metallibVersion,
        metalExecutable: metalExecutable,
        metallibExecutable: metallibExecutable
    )
}

func buildMetallib(
    toolchain: MetalToolchain,
    packageScratchDirectory: URL,
    paths: WorkspacePaths
) throws -> MetalBuildArtifact {
    let environment = toolchain.environment

    let checkout = packageScratchDirectory.appendingPathComponent("checkouts/mlx-swift")
    let sourceRoot = checkout.appendingPathComponent("Source/Cmlx/mlx-generated/metal")
    let sources = try regularFiles(in: sourceRoot, extensions: ["metal"]).sorted { $0.path < $1.path }
    guard !sources.isEmpty else {
        throw AcceptanceFailure.unknown("current mlx-swift checkout has no generated Metal sources")
    }

    let checkoutRevision = try commandOutput(["git", "rev-parse", "HEAD"], currentDirectory: checkout)

    let temporary = FileManager.default
        .temporaryDirectory
        .appendingPathComponent("swama-acceptance-metal-\(UUID().uuidString)")
    let airDirectory = temporary.appendingPathComponent("air")
    try FileManager.default.createDirectory(at: airDirectory, withIntermediateDirectories: true)
    do {
        var commands: [[String]] = []
        var airFiles: [URL] = []
        var sourceManifest: [JSONObject] = []
        for (index, source) in sources.enumerated() {
            let air = airDirectory.appendingPathComponent(String(
                format: "%02d-%@.air",
                index,
                source.deletingPathExtension().lastPathComponent
            ))
            let command = [
                "xcrun", "-sdk", "macosx", "metal", "-c",
                "-target", "air64-apple-macos14.0",
                "-fmetal-math-mode=fast",
                "-fmetal-math-fp32-functions=fast",
                source.path,
                "-o", air.path
            ]
            let result = try runCommand(
                command,
                currentDirectory: paths.repository,
                environment: environment,
                timeout: 300,
                sampleMemory: false,
                timeoutFailureKind: .unknown,
                timeoutContext: "Metal source compilation"
            )
            guard result.returnCode == 0, FileManager.default.fileExists(atPath: air.path) else {
                throw AcceptanceFailure
                    .unknown("Metal source compilation failed for \(source.lastPathComponent): \(result.stderr)")
            }

            commands.append(command)
            airFiles.append(air)
            try sourceManifest.append([
                "path": source.path.replacingOccurrences(of: checkout.path + "/", with: ""),
                "sha256": sha256File(source),
                "air_sha256": sha256File(air)
            ])
        }

        let output = temporary.appendingPathComponent("default.metallib")
        let linkCommand = ["xcrun", "-sdk", "macosx", "metallib"]
            + airFiles.map(\.path)
            + ["-o", output.path]
        let linkResult = try runCommand(
            linkCommand,
            currentDirectory: paths.repository,
            environment: environment,
            timeout: 300,
            sampleMemory: false,
            timeoutFailureKind: .unknown,
            timeoutContext: "metallib link"
        )
        guard linkResult.returnCode == 0, FileManager.default.fileExists(atPath: output.path) else {
            throw AcceptanceFailure.unknown("metallib link failed: \(linkResult.stderr)")
        }

        let normalizedMetalVersion = toolchain.metalVersion.split(separator: "\n")
            .filter { !$0.hasPrefix("InstalledDir:") }
            .joined(separator: "\n")
        return try .init(
            library: output,
            provenance: [
                "developer_dir": toolchain.developerDirectory.path,
                "metal_version": normalizedMetalVersion,
                "metallib_version": toolchain.metallibVersion,
                "metal_executable": toolchain.metalExecutable.path,
                "metal_executable_sha256": sha256File(toolchain.metalExecutable),
                "metallib_executable": toolchain.metallibExecutable.path,
                "metallib_executable_sha256": sha256File(toolchain.metallibExecutable),
                "mlx_swift_revision": checkoutRevision,
                "sources": sourceManifest,
                "compile_commands": commands,
                "link_command": linkCommand,
                "sha256": sha256File(output)
            ],
            temporaryDirectory: temporary
        )
    }
    catch {
        try? FileManager.default.removeItem(at: temporary)
        throw error
    }
}

func installMetallib(_ source: URL, nextTo binary: URL) throws -> JSONObject {
    let destination = binary.deletingLastPathComponent().appendingPathComponent("mlx.metallib")
    if FileManager.default.fileExists(atPath: destination.path) {
        try FileManager.default.removeItem(at: destination)
    }
    try FileManager.default.copyItem(at: source, to: destination)
    return try [
        "path": destination.path,
        "sha256": sha256File(destination),
        "copied": "true",
        "source": source.path
    ]
}

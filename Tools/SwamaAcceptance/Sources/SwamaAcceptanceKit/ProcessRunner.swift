import Darwin
import Foundation

// MARK: - CommandResult

struct CommandResult: Sendable {
    let command: [String]
    let returnCode: Int32
    let stdout: String
    let stderr: String
    let durationMilliseconds: Double
    let peakResidentBytes: UInt64
}

func developerEnvironment(_ developerDirectory: URL) throws -> [String: String] {
    guard FileManager.default.fileExists(
        atPath: developerDirectory.appendingPathComponent("usr/bin/xcodebuild").path
    )
    else {
        throw AcceptanceFailure.unknown("missing Xcode developer directory: \(developerDirectory.path)")
    }

    var environment = ProcessInfo.processInfo.environment
    environment["DEVELOPER_DIR"] = developerDirectory.path
    return environment
}

func runCommand(
    _ command: [String],
    currentDirectory: URL,
    environment: [String: String]? = nil,
    timeout: TimeInterval = 300,
    sampleMemory: Bool = true
) throws -> CommandResult {
    guard let executable = command.first else {
        throw AcceptanceFailure.unknown("empty command")
    }

    let temporary = FileManager.default
        .temporaryDirectory
        .appendingPathComponent("swama-acceptance-command-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }

    let stdoutURL = temporary.appendingPathComponent("stdout")
    let stderrURL = temporary.appendingPathComponent("stderr")
    FileManager.default.createFile(atPath: stdoutURL.path, contents: nil)
    FileManager.default.createFile(atPath: stderrURL.path, contents: nil)
    let stdoutHandle = try FileHandle(forWritingTo: stdoutURL)
    let stderrHandle = try FileHandle(forWritingTo: stderrURL)
    defer {
        try? stdoutHandle.close()
        try? stderrHandle.close()
    }

    let process = Process()
    if executable.contains("/") {
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = Array(command.dropFirst())
    }
    else {
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = command
    }
    process.currentDirectoryURL = currentDirectory
    process.environment = environment
    process.standardOutput = stdoutHandle
    process.standardError = stderrHandle

    let start = ContinuousClock.now
    do {
        try process.run()
    }
    catch {
        throw AcceptanceFailure.unknown("cannot launch \(command.joined(separator: " ")): \(error)")
    }

    let deadline = Date().addingTimeInterval(timeout)
    var peak: UInt64 = 0
    while process.isRunning {
        if sampleMemory {
            peak = max(peak, residentBytes(pid: process.processIdentifier))
        }
        if Date() >= deadline {
            kill(process.processIdentifier, SIGKILL)
            process.waitUntilExit()
            throw AcceptanceFailure.failed("timeout after \(timeout)s: \(command.joined(separator: " "))")
        }
        Thread.sleep(forTimeInterval: 0.02)
    }
    process.waitUntilExit()
    peak = max(peak, residentBytes(pid: process.processIdentifier))

    try stdoutHandle.synchronize()
    try stderrHandle.synchronize()
    let stdout = try String(decoding: Data(contentsOf: stdoutURL), as: UTF8.self)
    let stderr = try String(decoding: Data(contentsOf: stderrURL), as: UTF8.self)
    let duration = start.duration(to: .now)
    let milliseconds = Double(duration.components.seconds) * 1000
        + Double(duration.components.attoseconds) / 1_000_000_000_000_000
    let returnCode = process.terminationReason == .uncaughtSignal
        ? -process.terminationStatus
        : process.terminationStatus

    return .init(
        command: command,
        returnCode: returnCode,
        stdout: stdout,
        stderr: stderr,
        durationMilliseconds: milliseconds,
        peakResidentBytes: peak
    )
}

func commandOutput(
    _ command: [String],
    currentDirectory: URL,
    environment: [String: String]? = nil
) throws -> String {
    let result = try runCommand(
        command,
        currentDirectory: currentDirectory,
        environment: environment,
        sampleMemory: false
    )
    guard result.returnCode == 0 else {
        throw AcceptanceFailure.unknown(
            "command failed (\(result.returnCode)): \(command.joined(separator: " "))\n\(result.stderr)"
        )
    }

    return result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
}

func residentBytes(pid: pid_t) -> UInt64 {
    let process = Process()
    let pipe = Pipe()
    process.executableURL = URL(fileURLWithPath: "/bin/ps")
    process.arguments = ["-o", "rss=", "-p", String(pid)]
    process.standardOutput = pipe
    process.standardError = FileHandle.nullDevice
    do {
        try process.run()
        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard process.terminationStatus == 0,
              let text = String(data: data, encoding: .utf8),
              let kibibytes = UInt64(text.trimmingCharacters(in: .whitespacesAndNewlines))
        else {
            return 0
        }

        return kibibytes * 1024
    }
    catch {
        return 0
    }
}

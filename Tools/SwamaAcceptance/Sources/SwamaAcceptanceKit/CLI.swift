import Foundation

// MARK: - AcceptanceCLI

public enum AcceptanceCLI {
    public static func run(_ rawArguments: [String]) async -> Int32 {
        do {
            let arguments = try Arguments.parse(rawArguments)
            let paths = try WorkspacePaths.discover(explicit: arguments.repoRoot)
            let contractURL = arguments.contract.map { URL(fileURLWithPath: $0) } ?? paths.contract
            let contract = try AcceptanceContract.load(from: contractURL)

            switch arguments.command {
            case .architecture:
                let report = try architectureReport(
                    contract: contract.architecture,
                    stage: arguments.stage,
                    paths: paths
                )
                try writeStandardOutput(report)
                return report["passed"] as? Bool == true ? 0 : 1

            case .baseline:
                guard let output = arguments.output else {
                    throw AcceptanceFailure.unknown("baseline requires --output")
                }

                var report = try await runBaseline(
                    contract: contract,
                    contractURL: contractURL,
                    developerDirectory: URL(fileURLWithPath: arguments.developerDirectory),
                    metalDeveloperDirectory: URL(fileURLWithPath: arguments.metalDeveloperDirectory),
                    architectureStage: arguments.stage,
                    paths: paths
                )
                try sealReport(&report)
                try prettyJSONData(report).write(to: URL(fileURLWithPath: output), options: .atomic)
                try writeStandardOutput(report)
                return report["passed"] as? Bool == true ? 0 : 1

            case .compare:
                guard let baseline = arguments.baseline, let candidate = arguments.candidate else {
                    throw AcceptanceFailure.unknown("compare requires --baseline and --candidate")
                }

                var report = try compareReports(
                    baseline: loadJSONObject(URL(fileURLWithPath: baseline)),
                    candidate: loadJSONObject(URL(fileURLWithPath: candidate)),
                    contract: contract,
                    contractURL: contractURL,
                    paths: paths
                )
                if let output = arguments.output {
                    try sealReport(&report)
                    try prettyJSONData(report).write(to: URL(fileURLWithPath: output), options: .atomic)
                }
                try writeStandardOutput(report)
                return report["passed"] as? Bool == true ? 0 : 1
            }
        }
        catch let error as AcceptanceFailure {
            let status = error.kind == .unknown ? "UNKNOWN" : "FAIL"
            try? writeStandardError(["status": status, "error": error.message])
            return error.kind == .unknown ? 2 : 1
        }
        catch {
            try? writeStandardError(["status": "UNKNOWN", "error": String(describing: error)])
            return 2
        }
    }
}

// MARK: - Command

private enum Command: String {
    case architecture
    case baseline
    case compare
}

// MARK: - Arguments

private struct Arguments {
    let command: Command
    let repoRoot: String?
    let contract: String?
    let developerDirectory: String
    let metalDeveloperDirectory: String
    let stage: ArchitectureStage
    let output: String?
    let baseline: String?
    let candidate: String?

    static func parse(_ raw: [String]) throws -> Self {
        var values: [String: String] = [:]
        var positionals: [String] = []
        var index = 0
        while index < raw.count {
            let argument = raw[index]
            if argument.hasPrefix("--") {
                guard index + 1 < raw.count else {
                    throw AcceptanceFailure.unknown("missing value for \(argument)")
                }

                values[argument] = raw[index + 1]
                index += 2
            }
            else {
                positionals.append(argument)
                index += 1
            }
        }
        guard positionals.count == 1, let command = Command(rawValue: positionals[0]) else {
            throw AcceptanceFailure.unknown("usage: swama-acceptance [options] architecture|baseline|compare")
        }

        let stageText = values["--stage"] ?? values["--architecture-stage"] ?? "legacy-ratchet"
        guard let stage = ArchitectureStage(rawValue: stageText) else {
            throw AcceptanceFailure.unknown("unknown architecture stage: \(stageText)")
        }

        return .init(
            command: command,
            repoRoot: values["--repo-root"],
            contract: values["--contract"],
            developerDirectory: values["--developer-dir"] ?? "/Applications/Xcode.app/Contents/Developer",
            metalDeveloperDirectory: values["--metal-developer-dir"] ??
                "/Applications/Xcode-beta.app/Contents/Developer",
            stage: stage,
            output: values["--output"],
            baseline: values["--baseline"],
            candidate: values["--candidate"]
        )
    }
}

private func writeStandardOutput(_ object: Any) throws {
    try FileHandle.standardOutput.write(prettyJSONData(object))
}

private func writeStandardError(_ object: Any) throws {
    try FileHandle.standardError.write(prettyJSONData(object))
}

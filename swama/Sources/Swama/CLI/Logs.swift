import ArgumentParser
import Foundation
import SwamaKit

// MARK: - LogsError

private enum LogsError: LocalizedError {
    case malformed
    case invalidLineCount

    var errorDescription: String? {
        switch self {
        case .malformed:
            "UNKNOWN: diagnostics JSONL is malformed or truncated"
        case .invalidLineCount:
            "--lines must be zero or greater"
        }
    }
}

// MARK: - Logs

struct Logs: AsyncParsableCommand {
    static let configuration: CommandConfiguration = .init(
        abstract: "Read Swama's agent-friendly JSONL diagnostics"
    )

    @Option(name: .long, help: "Number of existing lines to print before following (default: 200)")
    var lines: Int = 200

    @Flag(name: .long, help: "Continue printing new diagnostic events")
    var follow = false

    func run() async throws {
        guard lines >= 0 else {
            throw LogsError.invalidLineCount
        }

        let url = SwamaDiagnostics.logFileURL
        guard FileManager.default.fileExists(atPath: url.path)
            || FileManager.default.fileExists(atPath: url.appendingPathExtension("1").path)
        else {
            return
        }

        let snapshot = try SwamaDiagnostics.readLogSnapshot()
        guard SwamaDiagnostics.isValidSnapshot(snapshot) else {
            throw LogsError.malformed
        }

        let existingLines = snapshot.split(separator: 0x0A, omittingEmptySubsequences: true)
        var seen = Set(existingLines.compactMap(eventKey))
        let selectedLines = existingLines.suffix(lines)
        if !selectedLines.isEmpty {
            let output = selectedLines.map { Data($0) }.reduce(into: Data()) { result, line in
                result.append(line)
                result.append(0x0A)
            }
            try FileHandle.standardOutput.write(contentsOf: output)
        }

        guard follow else {
            return
        }

        while !Task.isCancelled {
            try await Task.sleep(for: .seconds(1))
            let nextSnapshot = try SwamaDiagnostics.readLogSnapshot()
            guard SwamaDiagnostics.isValidSnapshot(nextSnapshot) else {
                throw LogsError.malformed
            }

            let nextLines = nextSnapshot.split(separator: 0x0A, omittingEmptySubsequences: true)
            let nextKeys = Set(nextLines.compactMap(eventKey))
            for line in nextLines {
                guard let key = eventKey(line), !seen.contains(key) else {
                    continue
                }

                try FileHandle.standardOutput.write(contentsOf: Data(line))
                try FileHandle.standardOutput.write(contentsOf: Data([0x0A]))
            }
            seen = nextKeys
        }
    }

    private func eventKey(_ line: Data.SubSequence) -> String? {
        guard let object = try? JSONSerialization.jsonObject(with: Data(line)) as? [String: Any],
              let session = object["session"] as? String,
              let sequence = object["seq"] as? NSNumber
        else {
            return nil
        }

        return "\(session):\(sequence.uint64Value)"
    }
}

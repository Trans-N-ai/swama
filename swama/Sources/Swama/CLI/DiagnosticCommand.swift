import ArgumentParser
import SwamaKit

// MARK: - SwamaLoggedCommand

protocol SwamaLoggedCommand: AsyncParsableCommand {
    static var diagnosticMode: SwamaDiagnosticMode { get }
    func runLogged() async throws
}

extension SwamaLoggedCommand {
    static var diagnosticMode: SwamaDiagnosticMode { .cli }

    func run() async throws {
        try await SwamaDiagnostics.withSession(mode: Self.diagnosticMode) {
            try await runLogged()
        }
    }
}

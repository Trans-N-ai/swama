import ArgumentParser
import NIO
import NIOHTTP1
import SwamaKit

struct Serve: AsyncParsableCommand {
    static let configuration: CommandConfiguration = .init(
        abstract: "Start the Swama API service (NIO)"
    )

    @Option(name: .long, help: "Host to bind to (default: 0.0.0.0)")
    var host: String = "0.0.0.0"

    @Option(name: .long, help: "Port to bind to (default: 28100)")
    var port: Int = 28100

    func run() async throws {
        // Use the ServerManager from SwamaKit to run the server in CLI mode.
        // Create a new instance of ServerManager for the CLI.
        // The ServerManager's properties (group, channel) are not used by runForCLI directly,
        // as runForCLI manages its own NIO resources.
        let serverManager = SwamaKit.ServerManager()
        print("CLI Serve: Initialized SwamaKit.ServerManager for CLI operation.")

        // The runForCLI method now encapsulates the NIO server setup and lifecycle for the CLI.
        try await serverManager.runForCLI(host: host, port: port)

        print("CLI Serve: Server (via SwamaKit.ServerManager.runForCLI) has shut down.")
    }
}

//
//  ServerManager.swift
//  SwamaKit
//

import Foundation
import NIO
import NIOHTTP1

// MARK: - ServerError

public enum ServerError: Error {
    case setupFailed(String)
}

// MARK: - ServerManager

public class ServerManager {
    // MARK: Lifecycle

    /// Initializer for when ServerManager is used for a background server (e.g., by TestApp)
    public init(host: String = "0.0.0.0", port: Int? = nil) {
        self.hostForBackground = host
        self.portForBackground = port ?? Self.defaultPort()
        NSLog("SwamaKit.ServerManager: Initialized for background server at \(host):\(self.portForBackground)")
    }

    /// Convenience initializer for CLI or other uses where background host/port are not pre-defined
    public convenience init() {
        self.init(host: "0.0.0.0", port: nil) // Uses environment variable or default
        NSLog("SwamaKit.ServerManager: Initialized (using environment configuration)")
    }

    deinit {
        NSLog("SwamaKit.ServerManager: Deinit called.")
        // This deinit is for the ServerManager instance, which primarily manages the background server.
        // If group or channel for background server are still active, attempt cleanup.
        if channel?.isActive == true || group != nil {
            let localGroup = self.group
            let localChannel = self.channel
            // Clear them on self to prevent re-entry if deinit is somehow called again or concurrently
            self.group = nil
            self.channel = nil

            Task.detached {
                NSLog("SwamaKit.ServerManager: Deinit task starting shutdown for background server resources.")
                if let ch = localChannel, ch.isActive {
                    do {
                        try await ch.close(mode: .all).get()
                        NSLog("SwamaKit.ServerManager Deinit: Closed channel.")
                    }
                    catch {
                        NSLog("SwamaKit.ServerManager Deinit Error: Failed to close channel - \(error)")
                    }
                }
                if let grp = localGroup {
                    do {
                        try await grp.shutdownGracefully()
                        NSLog("SwamaKit.ServerManager Deinit: Shutdown group.")
                    }
                    catch {
                        NSLog("SwamaKit.ServerManager Deinit Error: Failed to shutdown group - \(error)")
                    }
                }
                NSLog("SwamaKit.ServerManager: Deinit task finished shutdown attempts for background server resources.")
            }
        }
    }

    // MARK: Public

    /// Returns the default port, reading from SWAMA_PORT environment variable if available
    public static func defaultPort() -> Int {
        let defaultPort = 28100

        guard let portString = ProcessInfo.processInfo.environment["SWAMA_PORT"] else {
            NSLog("SwamaKit.ServerManager: Using default port \(defaultPort) (SWAMA_PORT not set)")
            return defaultPort
        }
        guard let port = Int(portString), port > 0, port <= 65535 else {
            NSLog("SwamaKit.ServerManager: Invalid SWAMA_PORT value '\(portString)', using default port \(defaultPort)")
            return defaultPort
        }

        NSLog("SwamaKit.ServerManager: Using port \(port) from SWAMA_PORT environment variable")
        return port
    }

    @MainActor // Ensures UI-related calls or state updates are safe if any; primarily for Task setup
    public func startInBackground() throws { // Renamed from startServer, uses host/port from init
        NSLog(
            "SwamaKit.ServerManager: Attempting to start server in background on \(hostForBackground):\(portForBackground)..."
        )

        // Synchronous part of setup that can throw
        guard self.group == nil else {
            NSLog("SwamaKit.ServerManager Error: Background server already started or starting (group exists).")
            throw ServerError.setupFailed("Background server already started or group exists.")
        }

        self.group = MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)
        guard self.group != nil else {
            NSLog("SwamaKit.ServerManager Error: Failed to create EventLoopGroup for background server.")
            // self.group will remain nil
            throw ServerError.setupFailed("Failed to create EventLoopGroup for background server.")
        }

        Task { // Launch server operations on a background thread using a Task
            // Capture self weakly to avoid retain cycles if ServerManager instance is deallocated before Task completes
            // However, the group is owned by self, so self must live as long as the group.
            // The Task itself will retain self until it completes.
            guard let group = self.group else { // Re-check group, though it should be set
                NSLog("SwamaKit.ServerManager Error: EventLoopGroup is nil at start of Task.")
                return
            }

            let bootstrap = ServerBootstrap(group: group)
                .serverChannelOption(ChannelOptions.backlog, value: 256)
                .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
                .childChannelInitializer { channelInOut in
                    channelInOut.pipeline.configureHTTPServerPipeline().flatMap {
                        // HTTPHandler() must be accessible from SwamaKit
                        channelInOut.pipeline.addHandler(HTTPHandler())
                    }
                }
                .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
                .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 16)
                .childChannelOption(ChannelOptions.recvAllocator, value: AdaptiveRecvByteBufferAllocator())

            do {
                // Use hostForBackground and portForBackground from the instance
                let boundChannel = try await bootstrap.bind(host: self.hostForBackground, port: self.portForBackground)
                    .get()
                self.channel = boundChannel // Store the channel for the background server
                NSLog(
                    "🚀 SwamaKit.ServerManager: Swama background server started on \(self.hostForBackground):\(self.portForBackground)"
                )

                try await boundChannel.closeFuture.get()
                NSLog("SwamaKit.ServerManager: Background server channel closed.")
            }
            catch {
                NSLog("SwamaKit.ServerManager Error (Background Task): Failed to start or run server: \(error)")
            }
            // This part will be reached after the server channel is closed or if an error occurred in the Task.
            // Ensure shutdown is called for the background server's group.
            await self.shutdownServerGracefully(isCalledFromDeinit: false, fromError: true)
        }
    }

    /// Renamed from stopServer
    public func stop() async {
        NSLog("SwamaKit.ServerManager: Explicitly stopping background server (stop() called)...")
        // This will trigger the shutdown of the channel and group managed by startInBackground's Task.
        await self.shutdownServerGracefully(isCalledFromDeinit: false, fromError: false)
    }

    /// This method is primarily for the background server instance's resources.
    public func shutdownServerGracefully(isCalledFromDeinit: Bool = false, fromError: Bool = false) async {
        let contextMessage = isCalledFromDeinit ? "Deinit" : (fromError ? "Error/Completion" : "Explicit Stop")
        NSLog("SwamaKit.ServerManager (\(contextMessage)): Initiating graceful shutdown for background server...")

        if let ch = self.channel, ch.isActive {
            NSLog("SwamaKit.ServerManager (\(contextMessage)): Closing background server channel.")
            do {
                try await ch.close(mode: .all).get() // Ensure it's awaited if called from async context
                NSLog("SwamaKit.ServerManager (\(contextMessage)): Background server channel closed.")
            }
            catch {
                NSLog(
                    "SwamaKit.ServerManager Error (\(contextMessage)): Failed to close background server channel - \(error)"
                )
            }
        }
        self.channel = nil

        if let grp = self.group {
            NSLog("SwamaKit.ServerManager (\(contextMessage)): Shutting down background EventLoopGroup.")
            do {
                try await grp.shutdownGracefully()
                NSLog("SwamaKit.ServerManager (\(contextMessage)): Background EventLoopGroup shut down.")
            }
            catch {
                NSLog(
                    "SwamaKit.ServerManager Error (\(contextMessage)): Failed to shutdown background EventLoopGroup - \(error)"
                )
            }
        }
        self.group = nil
        NSLog("SwamaKit.ServerManager (\(contextMessage)): Graceful shutdown for background server completed.")
    }

    /// New method for CLI: Manages its own resources.
    public func runForCLI(host cliHost: String, port cliPort: Int) async throws {
        NSLog("SwamaKit.ServerManager: Running for CLI on \(cliHost):\(cliPort)...")
        // CLI manages its own group and channel, separate from the instance's background server properties
        let cliGroup = MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)
        var cliChannelLocal: Channel? // Local variable for the CLI's channel

        let bootstrap = ServerBootstrap(group: cliGroup)
            .serverChannelOption(ChannelOptions.backlog, value: 256)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channelInOut in
                channelInOut.pipeline.configureHTTPServerPipeline().flatMap {
                    channelInOut.pipeline.addHandler(HTTPHandler()) // Uses SwamaKit.HTTPHandler
                }
            }
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 16)
            .childChannelOption(ChannelOptions.recvAllocator, value: AdaptiveRecvByteBufferAllocator())

        do {
            let serverChannel = try await bootstrap.bind(host: cliHost, port: cliPort).get()
            cliChannelLocal = serverChannel
            NSLog(
                "🚀 SwamaKit.ServerManager (CLI): Server running on \(serverChannel.localAddress!). Press CTRL+C to stop."
            )

            // Wait for the server channel to close (e.g., due to SIGINT/Ctrl+C handled by ArgumentParser or OS)
            try await serverChannel.closeFuture.get()
            NSLog("SwamaKit.ServerManager (CLI): Server channel closed.")
        }
        catch {
            NSLog("SwamaKit.ServerManager (CLI) Error: \(error)")
            // Ensure channel is closed if it was opened before the error
            if let ch = cliChannelLocal, ch.isActive {
                do {
                    try await ch.close().get()
                }
                catch {
                    NSLog("SwamaKit.ServerManager (CLI) Error: Failed to close channel during error handling - \(error)"
                    )
                }
            }
            throw error // Re-throw error to be handled by the CLI caller
        }
        // Code previously in 'finally' block is moved here
        NSLog("SwamaKit.ServerManager (CLI): Shutting down EventLoopGroup...")
        do {
            try await cliGroup.shutdownGracefully()
            NSLog("SwamaKit.ServerManager (CLI): EventLoopGroup shut down.")
        }
        catch {
            NSLog("SwamaKit.ServerManager (CLI) Error: Failed to shutdown EventLoopGroup - \(error)")
            // Optionally rethrow or handle this specific shutdown error if critical
        }
    }

    // MARK: Private

    private var group: MultiThreadedEventLoopGroup?
    private var channel: Channel? // For the background server instance

    private let hostForBackground: String
    private let portForBackground: Int
}

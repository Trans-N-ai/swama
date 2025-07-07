import AppKit // Required for NSApplication
import ArgumentParser
import SwamaKit

// MARK: - Swama

@main
@available(macOS 13.3, *)
struct Swama: AsyncParsableCommand {
    static let configuration: CommandConfiguration = .init(
        commandName: "swama",
        abstract: "Swama - The Swift-native LLM runtime for macOS",
        version: "1.4.0",
        subcommands: [Serve.self, Pull.self, Run.self, MenuBar.self, List.self, Transcribe.self], // Added Transcribe
        defaultSubcommand: Serve.self
    )
}

// MARK: - MenuBar

/// 添加新的子命令用于启动菜单栏应用
struct MenuBar: AsyncParsableCommand {
    static let configuration: CommandConfiguration = .init(
        commandName: "menubar",
        abstract: "Run Swama as a menu bar application." // Updated abstract
    )

    // @OptionGroup()
    // var commonOptions: CommonRunOptions // Kept for future use if needed

    @MainActor // NSApplication operations must be on the main actor
    func run() async throws { // Changed to async throws
        print("Swama CLI: Starting in menu bar application mode...")

        // Re-implementing the core logic of the old bootstrapMenuBarApp here
        let app = NSApplication.shared

        // Create and assign the AppDelegate from SwamaKit
        // SwamaKit.AppDelegate's applicationDidFinishLaunching will handle icon and menu setup.
        let delegate = SwamaKit.AppDelegate()
        app.delegate = delegate

        // Set activation policy for a menu bar app (no Dock icon, no main window usually)
        if app.activationPolicy() != .accessory {
            app.setActivationPolicy(.accessory)
            print("Swama CLI: Set activation policy to .accessory")
        }

        // Run the application's main event loop.
        // This is a blocking call and will not return until the app terminates.
        app.run()

        // This line will only be reached if app.run() somehow returns,
        // which typically only happens on termination.
        print("Swama CLI: Menu bar application has exited.")
    }
}

// MARK: - CommonRunOptions

/// 如果有通用选项，定义在这里，例如：
struct CommonRunOptions: ParsableArguments {
    // @Flag(help: "Enable verbose output.")
    // var verbose: Bool = false
}

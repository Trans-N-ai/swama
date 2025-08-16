import AppKit
import Foundation

// MARK: - AppDelegate

@MainActor
public class AppDelegate: NSObject, NSApplicationDelegate {
    // MARK: Lifecycle

    // MARK: ‑ Init

    override public init() {
        super.init()
        NSLog("SwamaKit.AppDelegate: init() called.")

        self.serverManager = ServerManager(host: "0.0.0.0", port: 28100)
    }

    // MARK: Public

    // MARK: ‑ NSApplicationDelegate

    public func applicationDidFinishLaunching(_: Notification) {
        // turn app into a menu‑bar‑only application
        if NSApp.activationPolicy() != .accessory {
            NSApp.setActivationPolicy(.accessory)
        }

        // start backend server
        Task {
            do { try serverManager?.startInBackground() }
            catch { NSLog("SwamaKit.AppDelegate: server failed to start → \(error)") }
        }
    }

    public func applicationWillTerminate(_: Notification) {
        Task { await serverManager?.stop() }
    }

    // MARK: Private

    private var serverManager: ServerManager?
}

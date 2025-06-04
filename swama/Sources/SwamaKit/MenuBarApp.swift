import AppKit
import Foundation

/// AppDelegate for Swama menu‑bar application
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

        // create status‑bar item
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = statusItem.button {
            button.image = NSImage(named: "MenuBarIcon")
            button.image?.accessibilityDescription = "Swama Control"
            button.image?.isTemplate = true
        }
        buildMenu()

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

    private var statusItem: NSStatusItem!
    private var serverManager: ServerManager?

    // MARK: ‑ Menu

    private func buildMenu() {
        let menu = NSMenu()
        menu.addItem(withTitle: "Install Command Line Tool…", action: #selector(installCLITool), keyEquivalent: "")
        menu.addItem(NSMenuItem.separator())
        menu.addItem(withTitle: "Quit Swama", action: #selector(quitApplication), keyEquivalent: "q")
        statusItem.menu = menu
    }

    // MARK: ‑ Helper utils

    private func shellEscape(_ p: String) -> String { "'" + p.replacingOccurrences(of: "'", with: "'\\''") + "'" }

    private func alert(_ title: String, _ msg: String, info: Bool = false) {
        let a = NSAlert(); a.messageText = title; a.informativeText = msg
        a.alertStyle = info ? .informational : .warning; a.addButton(withTitle: "OK"); a.runModal()
    }

    private func manualMsg(tmp: String, dest: String, why: String) {
        let cmd =
            "sudo mkdir -p /usr/local/bin && sudo cp \(shellEscape(tmp)) \(shellEscape(dest)) && sudo chmod 755 \(shellEscape(dest))"
        alert("Manual Step Required", "\(why)\n\nRun in Terminal:\n\n\(cmd)", info: true)
    }

    // MARK: ‑ Install CLI wrapper

    /// Installs a small wrapper script to `/usr/local/bin/swama`.
    /// The real Mach‑O binary is named **swama‑bin** inside `Contents/Helpers` to avoid name collision.
    @objc private func installCLITool() {
        let cliBinName = "swama-bin" // real executable
        let wrapperPath = "/usr/local/bin/swama" // destination wrapper
        let binDir = "/usr/local/bin"

        // locate Helpers directory & binary
        let helpersDir = "\(Bundle.main.bundlePath)/Contents/Helpers"
        let binPath = "\(helpersDir)/\(cliBinName)"
        guard FileManager.default.isExecutableFile(atPath: binPath) else {
            alert("Installation Failed", "Mach‑O binary not found: \(binPath)")
            return
        }

        // generate wrapper script (string literal)
        let script = """
        #!/usr/bin/env bash
        [ -n "$SWAMA_WRAPPER_DEBUG" ] && set -x
        set -euo pipefail
        PREFIX=\"\(helpersDir)\"
        export SWIFTPM_BUNDLE=\"$PREFIX\"
        export DYLD_FRAMEWORK_PATH=\"$PREFIX${DYLD_FRAMEWORK_PATH:+:$DYLD_FRAMEWORK_PATH}\"
        exec \"$PREFIX/\(cliBinName)\" \"$@\"
        """

        // write to temporary file
        let tmpURL = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("swama-wrapper-\(UUID().uuidString).sh")
        do {
            try script.write(to: tmpURL, atomically: true, encoding: .utf8)
        }
        catch {
            alert("Installation Failed", "Unable to write temporary wrapper script: \(error.localizedDescription)")
            return
        }

        // assemble AppleScript: ensure dir, copy, chmod
        let cpCmd =
            "/bin/mkdir -p \(binDir) && /bin/cp \(shellEscape(tmpURL.path)) \(shellEscape(wrapperPath)) && /bin/chmod 755 \(shellEscape(wrapperPath))"
        let osaSrc = "do shell script \"\(cpCmd)\" with administrator privileges"

        var errDict: NSDictionary?
        if let osa = NSAppleScript(source: osaSrc) {
            // The result of executeAndReturnError is an NSAppleEventDescriptor, not a Bool.
            // We check if errDict is nil for success, and if errDict is populated on failure.
            let _ = osa.executeAndReturnError(&errDict)
            if errDict == nil {
                alert("Installation Successful", "'swama' CLI wrapper installed to \(wrapperPath)", info: true)
            }
            else {
                let why = errDict?["NSAppleScriptErrorMessage"] as? String ?? "AppleScript failed"
                manualMsg(tmp: tmpURL.path, dest: wrapperPath, why: why)
            }
        }
        else {
            // This case would mean NSAppleScript(source: osaSrc) failed, which is unlikely for a valid script string.
            // However, to be safe, handle it as a failure.
            let why = "Failed to initialize NSAppleScript object."
            manualMsg(tmp: tmpURL.path, dest: wrapperPath, why: why)
        }
    }

    // MARK: ‑ Quit

    @objc private func quitApplication() { NSApp.terminate(nil) }
}

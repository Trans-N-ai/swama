import AppKit
import Foundation

/// AppDelegate for Swama menu‑bar application
public enum CLIToolStatus {
    case notInstalled
    case needsUpdate
    case upToDate
}

private struct CLIToolPaths {
    let cliBinName = "swama-bin" // real executable
    let wrapperPath = "/usr/local/bin/swama" // destination wrapper
    let binDir = "/usr/local/bin"
    let helpersDir = "\(Bundle.main.bundlePath)/Contents/Helpers"
    
    var binPath: String {
        "\(helpersDir)/\(cliBinName)"
    }
}


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

    // MARK: ‑ CLI Tool Paths

    private var cliToolPaths: CLIToolPaths {
        CLIToolPaths()
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

    /// Generate the wrapper script content
    private func generateWrapperScript() -> String {
        let paths = cliToolPaths
        
        return """
        #!/usr/bin/env bash
        [ -n "$SWAMA_WRAPPER_DEBUG" ] && set -x
        set -euo pipefail
        PREFIX=\"\(paths.helpersDir)\"
        export SWIFTPM_BUNDLE=\"$PREFIX\"
        export DYLD_FRAMEWORK_PATH=\"$PREFIX${DYLD_FRAMEWORK_PATH:+:$DYLD_FRAMEWORK_PATH}\"
        exec \"$PREFIX/\(paths.cliBinName)\" \"$@\"
        """
    }

    /// Installs a small wrapper script to `/usr/local/bin/swama`.
    /// The real Mach‑O binary is named **swama‑bin** inside `Contents/Helpers` to avoid name collision.
    public func installCLITool() {
        let paths = cliToolPaths

        // locate Helpers directory & binary
        guard FileManager.default.isExecutableFile(atPath: paths.binPath) else {
            alert("Installation Failed", "Mach‑O binary not found: \(paths.binPath)")
            return
        }

        // generate wrapper script
        let script = generateWrapperScript()

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
            "/bin/mkdir -p \(paths.binDir) && /bin/cp \(shellEscape(tmpURL.path)) \(shellEscape(paths.wrapperPath)) && /bin/chmod 755 \(shellEscape(paths.wrapperPath))"
        let osaSrc = "do shell script \"\(cpCmd)\" with administrator privileges"

        var errDict: NSDictionary?
        if let osa = NSAppleScript(source: osaSrc) {
            let _ = osa.executeAndReturnError(&errDict)
            if errDict == nil {
                alert("Installation Successful", "'swama' CLI wrapper installed to \(paths.wrapperPath)", info: true)
            }
            else {
                let why = errDict?["NSAppleScriptErrorMessage"] as? String ?? "AppleScript failed"
                manualMsg(tmp: tmpURL.path, dest: paths.wrapperPath, why: why)
            }
        }
        else {
            let why = "Failed to initialize NSAppleScript object."
            manualMsg(tmp: tmpURL.path, dest: paths.wrapperPath, why: why)
        }
    }

    public func checkCLIToolStatus() -> CLIToolStatus {
        let paths = cliToolPaths
        
        // Check if wrapper exists
        guard FileManager.default.fileExists(atPath: paths.wrapperPath) else {
            return .notInstalled
        }
        
        // Read current wrapper content
        guard let currentContent = try? String(contentsOfFile: paths.wrapperPath, encoding: .utf8) else {
            return .notInstalled
        }
        
        // Generate expected wrapper script
        let expectedScript = generateWrapperScript()
        
        // Compare content (trim whitespace for comparison)
        let currentTrimmed = currentContent.trimmingCharacters(in: .whitespacesAndNewlines)
        let expectedTrimmed = expectedScript.trimmingCharacters(in: .whitespacesAndNewlines)
        
        if currentTrimmed == expectedTrimmed {
            return .upToDate
        } else {
            return .needsUpdate
        }
    }
}
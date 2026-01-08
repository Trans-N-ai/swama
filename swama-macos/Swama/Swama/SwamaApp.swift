//
//  SwamaApp.swift
//  SwamaApp
//
//  Created by Xingyue on 2025/05/08.
//

import SwamaKit
import SwiftUI

@main
struct SwamaApp: App {
    /// Use NSApplicationDelegateAdaptor to integrate the existing AppDelegate
    /// from SwamaKit for managing the application lifecycle and menu bar.
    @NSApplicationDelegateAdaptor(SwamaKit.AppDelegate.self) var appDelegate

    @State private var cliToolStatus: CLIToolStatus = .notInstalled
    @State private var contextLimit: Int = ContextLimitConfig.Constants.defaultLimit
    @State private var pendingContextLimit: Int?

    init() {
        NSLog("SwamaApp: init() called. The application is starting.")
    }

    var body: some Scene {
        MenuBarExtra("Swama", image: "MenuBarIcon") {
            VStack {
                // Only show CLI tool button if not up to date
                if cliToolStatus != .upToDate {
                    Button(cliToolButtonTitle) {
                        appDelegate.installCLITool()
                        // Refresh status after installation
                        cliToolStatus = appDelegate.checkCLIToolStatus()
                    }
                    .keyboardShortcut("I", modifiers: [.command])
                    .onAppear {
                        // Check CLI tool status when menu appears
                        cliToolStatus = appDelegate.checkCLIToolStatus()
                    }

                    Button(restartButtonTitle) {
                        restartServer()
                    }

                    Divider()
                }
                else {
                    // Still need to check status when menu appears, even when button is hidden
                    Color.clear
                        .frame(height: 0)
                        .onAppear {}
                }

                Menu("Context Limit") {
                    Button(limitLabel("1k", 1024)) { setPendingContextLimit(1024) }
                    Button(limitLabel("4k", 4096)) { setPendingContextLimit(4096) }
                    Button(limitLabel("8k", 8192)) { setPendingContextLimit(8192) }
                    Button(limitLabel("16k", 16384)) { setPendingContextLimit(16384) }
                    Button(limitLabel("32k", 32768)) { setPendingContextLimit(32768) }
                    Button(limitLabel("64k", 65536)) { setPendingContextLimit(65536) }
                    Button(limitLabel("128k", 131_072)) { setPendingContextLimit(131_072) }
                    Button(limitLabel("256k", 262_144)) { setPendingContextLimit(262_144) }
                    Button(limitLabel("1M", 1_048_576)) { setPendingContextLimit(1_048_576) }
                }

                if cliToolStatus == .upToDate {
                    Button(restartButtonTitle) {
                        restartServer()
                    }
                }

                if let pending = pendingContextLimit {
                    Text("Pending: \(pendingLabel(pending)) (Restart to apply)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .padding(.top, 4)
                }

                Button("Quit Swama") {
                    NSApplication.shared.terminate(nil)
                }
                .keyboardShortcut("Q", modifiers: [.command])
            }
            .onAppear {
                cliToolStatus = appDelegate.checkCLIToolStatus()
                Task {
                    let current = await appDelegate.currentContextLimit()
                    await MainActor.run { contextLimit = current }
                }
            }
        }
    }

    private var cliToolButtonTitle: String {
        switch cliToolStatus {
        case .notInstalled:
            "Install Command Line Tool…"
        case .needsUpdate:
            "Update Command Line Tool…"
        case .upToDate:
            ""
        }
    }

    private var restartButtonTitle: String {
        pendingContextLimit == nil ? "Restart Server" : "Restart Server (Apply \(pendingLabel(pendingContextLimit)))"
    }

    private func limitLabel(_ label: String, _ value: Int) -> String {
        if value == contextLimit {
            return "\(label)  ✓"
        }
        if value == pendingContextLimit {
            return "\(label)  →"
        }
        return label
    }

    private func setPendingContextLimit(_ value: Int) {
        pendingContextLimit = value == contextLimit ? nil : value
    }

    private func restartServer() {
        Task {
            if let pending = pendingContextLimit {
                await appDelegate.applyContextLimit(pending)
                await MainActor.run {
                    contextLimit = pending
                    pendingContextLimit = nil
                }
            }
            await appDelegate.restartServer()
        }
    }

    private func pendingLabel(_ value: Int?) -> String {
        guard let value else {
            return ""
        }

        if value >= 1_048_576 {
            return "\(value / 1_048_576)M"
        }
        return "\(value / 1024)k"
    }
}

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
                    
                    Divider()
                } else {
                    // Still need to check status when menu appears, even when button is hidden
                    Color.clear
                        .frame(height: 0)
                        .onAppear {
                            cliToolStatus = appDelegate.checkCLIToolStatus()
                        }
                }

                Button("Quit Swama") {
                    NSApplication.shared.terminate(nil)
                }
                .keyboardShortcut("Q", modifiers: [.command])
            }
        }
    }

    private var cliToolButtonTitle: String {
        switch cliToolStatus {
        case .notInstalled:
            return "Install Command Line Tool…"
        case .needsUpdate:
            return "Update Command Line Tool…"
        case .upToDate:
            return ""
        }
    }
}

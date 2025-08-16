//
//  SwamaApp.swift
//  SwamaApp
//
//  Created by Xingyue on 2025/05/08.
//

import SwamaKit
import SwiftUI
import Combine

@main
struct SwamaApp: App {
    /// Use NSApplicationDelegateAdaptor to integrate the existing AppDelegate
    /// from SwamaKit for managing the application lifecycle and menu bar.
    @NSApplicationDelegateAdaptor(SwamaKit.AppDelegate.self) var appDelegate
    var menuDelegate: MenuDelegate { .shared }

    init() {
        NSLog("SwamaApp: init() called. The application is starting.")
    }

    var body: some Scene {
        MenuBarExtra("Swama", image: "MenuBarIcon") {
            VStack {
                ToolStatusView()
                ModelCacheView()
                Button("Quit Swama") {
                    NSApplication.shared.terminate(nil)
                }
                .keyboardShortcut("Q", modifiers: [.command])
            }
        }
    }
    
    struct ModelCacheView: View {
        var menuDelegate: MenuDelegate { .shared }
        @State private var modelCache: [String] = []
        @State private var cancellable: AnyCancellable?
        
        var body: some View {
            VStack {
                if modelCache.isEmpty {
                    Text("No Models Loaded")
                } else {
                    Text("Loaded Models")
                    ForEach(modelCache, id: \.self) { model in
                        Text(model).foregroundStyle(.gray)
                    }
                    Button("Unload All Models") {
                        Task { await ModelPool.shared.clearCache() }
                    }
                    Divider()
                }
            }
            .onAppear {
                Task {
                    self.cancellable = await ModelPool.shared.modelCachePublisher
                        .receive(on: DispatchQueue.main)
                        .sink { models in
                            self.modelCache = models
                        }
                }
            }
            .onDisappear {
                cancellable?.cancel()
            }
        }
    }
    
    struct ToolStatusView: View {
        var menuDelegate: MenuDelegate { .shared }
        @State private var cliToolStatus: CLIToolStatus = .notInstalled

        var body: some View {
            // Only show CLI tool button if not up to date
            if cliToolStatus != .upToDate {
                Button(cliToolButtonTitle) {
                    menuDelegate.installCLITool()
                    // Refresh status after installation
                    cliToolStatus = menuDelegate.checkCLIToolStatus()
                }
                .keyboardShortcut("I", modifiers: [.command])
                .onAppear {
                    // Check CLI tool status when menu appears
                    cliToolStatus = menuDelegate.checkCLIToolStatus()
                }
                Divider()
            } else {
                // Still need to check status when menu appears, even when button is hidden
                Color.clear
                    .frame(height: 0)
                    .onAppear {
                        cliToolStatus = menuDelegate.checkCLIToolStatus()
                    }
            }
        }

        private var cliToolButtonTitle: String {
            switch cliToolStatus {
            case .notInstalled:
                return "Install CLI…"
            case .needsUpdate:
                return "Update CLI…"
            case .upToDate:
                return ""
            }
        }
    }
}

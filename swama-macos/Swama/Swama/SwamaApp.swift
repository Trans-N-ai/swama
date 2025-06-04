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

    init() {
        NSLog("SwamaApp: init() called. The application is starting.")
    }

    var body: some Scene {
        // This application is primarily a menu bar app that runs a background server.
        // A traditional WindowGroup for a main window is not needed.
        // Using a Settings scene ensures the app stays running without a visible main window.
        // An EmptyView can be used if no settings UI is currently required.
        Settings {
            // Optionally, a simple view could be provided here for status or basic settings.
            EmptyView()
        }
    }
}

// The default ContentView is not used in this menu-bar-focused application.
// It can be removed or commented out if no main window UI is intended.
// struct ContentView: View {
//     var body: some View {
//         VStack {
//             Image(systemName: "globe")
//                 .imageScale(.large)
//                 .foregroundStyle(.tint)
//             Text("Hello, world!")
//         }
//         .padding()
//     }
// }

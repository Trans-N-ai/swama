import Foundation
import Testing

@Suite("Swama target boundaries")
struct TargetBoundaryTests {
    @Test func coreManifestHasNoShellDependencies() throws {
        let manifest = try String(
            contentsOf: packageRoot.appendingPathComponent("Package.swift"),
            encoding: .utf8
        )
        let core = try targetDeclaration(named: "SwamaKit", in: manifest)
        let server = try targetDeclaration(named: "SwamaServer", in: manifest)
        let appSupport = try targetDeclaration(named: "SwamaAppSupport", in: manifest)

        for forbiddenDependency in ["NIO", "NIOHTTP1", "ArgumentParser", "SwamaServer", "SwamaAppSupport"] {
            if core.contains("name: \"\(forbiddenDependency)\"") {
                Issue.record("SwamaKit must not depend on shell target or product \(forbiddenDependency)")
            }
        }

        #expect(server.contains(".target(name: \"SwamaKit\")"))
        #expect(server.contains("name: \"NIO\""))
        #expect(server.contains("name: \"NIOHTTP1\""))
        #expect(appSupport.contains(".target(name: \"SwamaKit\")"))
        #expect(appSupport.contains(".target(name: \"SwamaServer\")"))
    }

    @Test func coreSourcesDoNotImportShellFrameworks() throws {
        let coreRoot = packageRoot.appendingPathComponent("Sources/SwamaKit")
        let forbiddenModules = [
            "AppKit",
            "ArgumentParser",
            "NIO",
            "NIOCore",
            "NIOHTTP1",
            "SwamaServer",
            "SwamaAppSupport"
        ]

        let sourceURLs = try #require(
            FileManager.default
                .enumerator(
                    at: coreRoot,
                    includingPropertiesForKeys: nil
                )?.allObjects
                .compactMap { $0 as? URL }
                .filter { $0.pathExtension == "swift" }
        )

        for sourceURL in sourceURLs {
            let source = try String(contentsOf: sourceURL, encoding: .utf8)
            let imports = source.split(separator: "\n").map {
                $0.trimmingCharacters(in: .whitespaces)
            }

            for module in forbiddenModules {
                if imports.contains("import \(module)") {
                    Issue.record("SwamaKit must not import shell module \(module): \(sourceURL.lastPathComponent)")
                }
            }
        }
    }

    @Test func shellSourcesLiveOutsideTheCoreTarget() {
        let sources = packageRoot.appendingPathComponent("Sources")

        #expect(!FileManager.default.fileExists(atPath: sources.appendingPathComponent("SwamaKit/Server").path))
        #expect(!FileManager.default
            .fileExists(atPath: sources.appendingPathComponent("SwamaKit/MenuBarApp.swift").path)
        )
        #expect(FileManager.default
            .fileExists(atPath: sources.appendingPathComponent("SwamaServer/ServerManager.swift").path)
        )
        #expect(FileManager.default
            .fileExists(atPath: sources.appendingPathComponent("SwamaAppSupport/MenuBarApp.swift").path)
        )
    }

    private var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func targetDeclaration(named name: String, in manifest: String) throws -> Substring {
        let marker = ".target(\n            name: \"\(name)\""
        let start = try #require(manifest.range(of: marker)?.lowerBound)
        let end = try #require(
            manifest.range(
                of: "\n        ),",
                range: start ..< manifest.endIndex
            )?.upperBound
        )
        return manifest[start ..< end]
    }
}

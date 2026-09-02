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
        let forbiddenModules: Set = [
            "AppKit",
            "ArgumentParser",
            "NIO",
            "NIOCore",
            "NIOHTTP1",
            "SwamaServer",
            "SwamaAppSupport"
        ]
        let importExpression = try NSRegularExpression(pattern: swiftImportDeclarationPattern)

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
            for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
                if let module = importedModule(in: String(line), matching: importExpression),
                   forbiddenModules.contains(module)
                {
                    Issue.record("SwamaKit must not import shell module \(module): \(sourceURL.lastPathComponent)")
                }
            }
        }
    }

    @Test func importParserHandlesAttributesAccessAndScopedImports() throws {
        let expression = try NSRegularExpression(pattern: swiftImportDeclarationPattern)
        let imports = [
            "import SwamaServer": "SwamaServer",
            "@_exported import NIO": "NIO",
            "@preconcurrency import AppKit": "AppKit",
            "@testable import SwamaServer": "SwamaServer",
            "@_implementationOnly import ArgumentParser": "ArgumentParser",
            "@_spi(Testing) import NIOHTTP1": "NIOHTTP1",
            "internal import SwamaAppSupport": "SwamaAppSupport",
            "package import struct NIOCore.ByteBuffer": "NIOCore",
            "@preconcurrency import MLXLMCommon": "MLXLMCommon"
        ]

        for (line, module) in imports {
            #expect(importedModule(in: line, matching: expression) == module)
        }
        #expect(importedModule(in: "// @_exported import NIO", matching: expression) == nil)
        #expect(importedModule(in: "let example = \"import NIO\"", matching: expression) == nil)
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

    private var swiftImportDeclarationPattern: String {
        // Keep this grammar aligned with the external acceptance scanner. This committed test
        // must still protect the boundary when that external tool is not being run.
        #"^\s*(?:@[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?\s+)*(?:(?:private|fileprivate|internal|package|public|open)\s+)?import\s+(?:(?:typealias|struct|class|enum|protocol|let|var|func|macro)\s+)?([A-Za-z_][A-Za-z0-9_]*)\b"#
    }

    private func importedModule(in line: String, matching expression: NSRegularExpression) -> String? {
        let range = NSRange(line.startIndex ..< line.endIndex, in: line)
        guard let match = expression.firstMatch(in: line, range: range),
              let moduleRange = Range(match.range(at: 1), in: line)
        else {
            return nil
        }

        return String(line[moduleRange])
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

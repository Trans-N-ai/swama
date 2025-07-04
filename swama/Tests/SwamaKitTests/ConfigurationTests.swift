import Foundation
@testable import SwamaKit
import Testing

@MainActor @Suite(.serialized)
final class ConfigurationTests {
    @Test func fileManagerOperations() {
        // Test basic file manager operations that the app uses
        let fileManager = FileManager.default
        let homeDirectory = fileManager.homeDirectoryForCurrentUser

        #expect(homeDirectory.isFileURL)
        #expect(fileManager.fileExists(atPath: homeDirectory.path))
    }

    @Test func uRLExtensions() {
        // Test URL extensions and operations used in the app
        let testURL = URL(fileURLWithPath: "/tmp/test")
        #expect(testURL.path == "/tmp/test")
        #expect(testURL.isFileURL)

        let documentsURL = testURL.appendingPathComponent("documents")
        #expect(documentsURL.lastPathComponent == "documents")
    }

    @Test func metadataSourceCases() {
        // Test all MetadataSource enum cases
        let metaFile = MetadataSource.metaFile
        let directoryScan = MetadataSource.directoryScan

        #expect(metaFile != directoryScan)

        // Test that we can use them in functions
        func getSourceName(_ source: MetadataSource) -> String {
            switch source {
            case .metaFile:
                "meta_file"
            case .directoryScan:
                "directory_scan"
            }
        }

        #expect(getSourceName(metaFile) == "meta_file")
        #expect(getSourceName(directoryScan) == "directory_scan")
    }

    @Test func foundationIntegration() {
        // Test Foundation framework integration
        let testData = "Hello, World!".data(using: .utf8)
        #expect(testData != nil)

        if let data = testData {
            let string = String(data: data, encoding: .utf8)
            #expect(string == "Hello, World!")
        }
    }
}

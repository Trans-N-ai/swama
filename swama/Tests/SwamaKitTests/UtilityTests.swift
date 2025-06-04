import Foundation
@testable import SwamaKit
import Testing

@MainActor @Suite(.serialized)
final class UtilityTests {
    @Test func currentTimestamp() {
        // Test that we can get current timestamp using Foundation
        let timestamp = Int(Date().timeIntervalSince1970)

        // Check that it's a valid Unix timestamp (should be positive and reasonable)
        #expect(timestamp > 1_000_000_000) // After year 2001
        #expect(timestamp < 2_000_000_000) // Before year 2033

        // Test that consecutive calls are reasonably close
        let timestamp2 = Int(Date().timeIntervalSince1970)
        #expect(abs(timestamp2 - timestamp) <= 1) // Within 1 second
    }

    @Test func filePathUtilities() {
        // Test URL path construction
        let testURL = URL(fileURLWithPath: "/tmp/test")
        #expect(testURL.path == "/tmp/test")
        #expect(testURL.isFileURL)
    }

    @Test func stringExtensions() {
        // Basic string tests (if any custom extensions exist)
        let testString = "Hello, World!"
        #expect(testString.count == 13)
        #expect(testString.contains("World"))
    }

    @Test func dataConversion() {
        // Test basic data conversion utilities
        let testString = "Hello, Swift!"
        let data = testString.data(using: .utf8)
        #expect(data != nil)

        if let data {
            let convertedString = String(data: data, encoding: .utf8)
            #expect(convertedString == testString)
        }
    }

    @Test func jSONSerialization() throws {
        // Test JSON serialization/deserialization
        let testDict: [String: Any] = [
            "name": "Test Model",
            "version": 1.0,
            "active": true
        ]

        do {
            let jsonData = try JSONSerialization.data(withJSONObject: testDict)
            #expect(!jsonData.isEmpty)

            let deserializedDict = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any]
            #expect(deserializedDict != nil)
            #expect(deserializedDict?["name"] as? String == "Test Model")
            #expect(deserializedDict?["version"] as? Double == 1.0)
            #expect(deserializedDict?["active"] as? Bool == true)
        }
        catch {
            Issue.record("JSON serialization failed: \(error)")
        }
    }
}

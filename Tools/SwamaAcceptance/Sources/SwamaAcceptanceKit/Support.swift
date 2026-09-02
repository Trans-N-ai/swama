import CryptoKit
import Foundation

typealias JSONObject = [String: Any]

// MARK: - AcceptanceKind

enum AcceptanceKind {
    case failed
    case unknown
}

// MARK: - AcceptanceFailure

struct AcceptanceFailure: Error, CustomStringConvertible {
    let kind: AcceptanceKind
    let message: String

    var description: String { message }

    static func failed(_ message: String) -> Self {
        .init(kind: .failed, message: message)
    }

    static func unknown(_ message: String) -> Self {
        .init(kind: .unknown, message: message)
    }
}

// MARK: - WorkspacePaths

struct WorkspacePaths: Sendable {
    let repository: URL

    var package: URL { repository.appendingPathComponent("swama") }
    var harness: URL { repository.appendingPathComponent("Tools/SwamaAcceptance") }
    var fixture: URL { repository.appendingPathComponent("Tests/AcceptanceFixture") }
    var contract: URL { harness.appendingPathComponent("contract.json") }

    static func discover(explicit: String?) throws -> Self {
        if let explicit {
            return try validate(URL(fileURLWithPath: explicit).standardizedFileURL.resolvingSymlinksInPath())
        }

        var cursor = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).standardizedFileURL
        while cursor.path != "/" {
            if let result = try? validate(cursor) {
                return result
            }
            cursor.deleteLastPathComponent()
        }
        throw AcceptanceFailure.unknown("cannot locate repository root; pass --repo-root")
    }

    private static func validate(_ root: URL) throws -> Self {
        let required = [
            root.appendingPathComponent("swama/Package.swift"),
            root.appendingPathComponent("Tools/SwamaAcceptance/Package.swift"),
            root.appendingPathComponent("Tests/AcceptanceFixture/Package.swift")
        ]
        guard required.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) else {
            throw AcceptanceFailure.unknown("not a Swama acceptance repository root: \(root.path)")
        }

        return .init(repository: root)
    }
}

func loadJSONObject(_ url: URL) throws -> JSONObject {
    do {
        let data = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: data) as? JSONObject else {
            throw AcceptanceFailure.unknown("JSON root is not an object: \(url.path)")
        }

        return object
    }
    catch let error as AcceptanceFailure {
        throw error
    }
    catch {
        throw AcceptanceFailure.unknown("cannot read JSON \(url.path): \(error)")
    }
}

func compactJSONData(_ object: Any) throws -> Data {
    guard JSONSerialization.isValidJSONObject(object) else {
        throw AcceptanceFailure.unknown("report contains a non-JSON value")
    }

    do {
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes])
    }
    catch {
        throw AcceptanceFailure.unknown("cannot serialize JSON: \(error)")
    }
}

func prettyJSONData(_ object: Any) throws -> Data {
    guard JSONSerialization.isValidJSONObject(object) else {
        throw AcceptanceFailure.unknown("report contains a non-JSON value")
    }

    do {
        var data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        data.append(0x0A)
        return data
    }
    catch {
        throw AcceptanceFailure.unknown("cannot serialize JSON: \(error)")
    }
}

func sealReport(_ report: inout JSONObject) throws {
    report.removeValue(forKey: "report_sha256")
    report["report_sha256"] = try sha256(compactJSONData(report))
}

func verifyReport(_ report: JSONObject) throws {
    guard let expected = report["report_sha256"] as? String else {
        throw AcceptanceFailure.unknown("report self-hash is missing")
    }

    var payload = report
    payload.removeValue(forKey: "report_sha256")
    let actual = try sha256(compactJSONData(payload))
    guard expected == actual else {
        throw AcceptanceFailure.unknown("report self-hash is invalid")
    }
}

func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

func sha256File(_ url: URL) throws -> String {
    guard let input = InputStream(url: url) else {
        throw AcceptanceFailure.unknown("cannot open file for hashing: \(url.path)")
    }

    input.open()
    defer { input.close() }

    var hasher = SHA256()
    let capacity = 8 * 1024 * 1024
    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: capacity)
    defer { buffer.deallocate() }

    while input.hasBytesAvailable {
        let count = input.read(buffer, maxLength: capacity)
        if count < 0 {
            throw AcceptanceFailure.unknown("failed while hashing file: \(url.path)")
        }
        if count == 0 {
            break
        }
        hasher.update(data: Data(bytesNoCopy: buffer, count: count, deallocator: .none))
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

func sha256Tree(_ urls: [URL], relativeTo root: URL) throws -> String {
    var hasher = SHA256()
    for url in urls.sorted(by: { $0.path < $1.path }) {
        let relative = url.path.replacingOccurrences(of: root.path + "/", with: "")
        hasher.update(data: Data(relative.utf8))
        hasher.update(data: Data([0]))
        let data = try Data(contentsOf: url)
        hasher.update(data: data)
        hasher.update(data: Data([0]))
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

func regularFiles(in root: URL, extensions: Set<String>? = nil) throws -> [URL] {
    guard let enumerator = FileManager.default.enumerator(
        at: root,
        includingPropertiesForKeys: [.isRegularFileKey],
        options: [.skipsHiddenFiles]
    )
    else {
        throw AcceptanceFailure.unknown("cannot enumerate directory: \(root.path)")
    }

    return try enumerator.compactMap { item in
        guard let url = item as? URL else {
            return nil
        }

        let values = try url.resourceValues(forKeys: [.isRegularFileKey])
        guard values.isRegularFile == true else {
            return nil
        }

        if let extensions, !extensions.contains(url.pathExtension) {
            return nil
        }
        return url
    }
}

func writeReport(_ report: JSONObject, to output: URL) throws {
    var sealed = report
    try sealReport(&sealed)
    try FileManager.default.createDirectory(
        at: output.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try prettyJSONData(sealed).write(to: output, options: .atomic)
}

func lastJSONValue(_ output: String) throws -> Any {
    for line in output.split(separator: "\n").reversed() {
        let candidate = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard candidate.first == "[" || candidate.first == "{" else { continue }

        if let data = candidate.data(using: .utf8),
           let value = try? JSONSerialization.jsonObject(with: data)
        {
            return value
        }
    }
    throw AcceptanceFailure.unknown("process produced no machine-readable JSON record")
}

extension [String: Any] {
    func object(_ key: String) throws -> JSONObject {
        guard let value = self[key] as? JSONObject else {
            throw AcceptanceFailure.unknown("missing JSON object: \(key)")
        }

        return value
    }

    func array(_ key: String) throws -> [Any] {
        guard let value = self[key] as? [Any] else {
            throw AcceptanceFailure.unknown("missing JSON array: \(key)")
        }

        return value
    }

    func string(_ key: String) throws -> String {
        guard let value = self[key] as? String else {
            throw AcceptanceFailure.unknown("missing JSON string: \(key)")
        }

        return value
    }

    func integer(_ key: String) throws -> Int {
        guard let value = self[key] as? NSNumber else {
            throw AcceptanceFailure.unknown("missing JSON integer: \(key)")
        }

        return value.intValue
    }

    func double(_ key: String) throws -> Double {
        guard let value = self[key] as? NSNumber else {
            throw AcceptanceFailure.unknown("missing JSON number: \(key)")
        }

        return value.doubleValue
    }

    func boolean(_ key: String) throws -> Bool {
        guard let value = self[key] as? Bool else {
            throw AcceptanceFailure.unknown("missing JSON boolean: \(key)")
        }

        return value
    }
}

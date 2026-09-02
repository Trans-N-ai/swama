import Foundation

// MARK: - ArchitectureStage

enum ArchitectureStage: String, Sendable {
    case legacyRatchet = "legacy-ratchet"
    case coreBoundary = "core-boundary"
}

func architectureReport(
    contract: ArchitectureContract,
    stage: ArchitectureStage,
    paths: WorkspacePaths
) throws -> JSONObject {
    let packageManifest = try String(contentsOf: paths.package.appendingPathComponent("Package.swift"), encoding: .utf8)
    let swamaKit = paths.package.appendingPathComponent("Sources/SwamaKit")
    let forbidden = Set(contract.goalForbiddenImports)
    let legacyImports = try imports(in: swamaKit, forbidden: forbidden, repository: paths.repository)
    let legacyLeaks = try publicMLXLeaks(in: swamaKit, repository: paths.repository)
    var report: JSONObject = [
        "stage": stage.rawValue,
        "legacy_forbidden_imports": legacyImports,
        "legacy_public_mlx_leaks": legacyLeaks
    ]

    switch stage {
    case .legacyRatchet:
        let actualImports = Set(legacyImports.compactMap { item -> String? in
            guard let file = item["file"] as? String, let module = item["module"] as? String else { return nil }

            return "\(file):\(module)"
        })
        let allowedImports = Set(contract.legacyForbiddenImportAllowlist)
        let actualLeaks = Set(legacyLeaks.compactMap { item -> String? in
            guard let file = item["file"] as? String, let text = item["text"] as? String else { return nil }

            return "\(file):\(text)"
        })
        let allowedLeaks = Set(contract.legacyPublicMLXLeakAllowlist)
        let newImports = Array(actualImports.subtracting(allowedImports)).sorted()
        let removedImports = Array(allowedImports.subtracting(actualImports)).sorted()
        let newLeaks = Array(actualLeaks.subtracting(allowedLeaks)).sorted()
        let removedLeaks = Array(allowedLeaks.subtracting(actualLeaks)).sorted()
        report["new_forbidden_imports"] = newImports
        report["removed_forbidden_imports"] = removedImports
        report["new_public_mlx_leaks"] = newLeaks
        report["removed_public_mlx_leaks"] = removedLeaks
        report["passed"] = newImports.isEmpty && newLeaks.isEmpty

    case .coreBoundary:
        let coreRoot = paths.package.appendingPathComponent("Sources/\(contract.goalCoreTarget)")
        let coreImports = try imports(in: coreRoot, forbidden: forbidden, repository: paths.repository)
        let coreLeaks = try publicMLXLeaks(in: coreRoot, repository: paths.repository)
        let targetPresent = packageManifest.contains("name: \"\(contract.goalCoreTarget)\"")
            && FileManager.default.fileExists(atPath: coreRoot.path)
        report["core_target_present"] = targetPresent
        report["core_forbidden_imports"] = coreImports
        report["core_public_mlx_leaks"] = coreLeaks
        report["passed"] = targetPresent && coreImports.isEmpty && coreLeaks.isEmpty
    }
    return report
}

private func imports(
    in root: URL,
    forbidden: Set<String>,
    repository: URL
) throws -> [JSONObject] {
    guard FileManager.default.fileExists(atPath: root.path) else {
        return []
    }

    let expression = try NSRegularExpression(pattern: #"^\s*(?:@preconcurrency\s+)?import\s+([A-Za-z0-9_]+)"#)
    var hits: [JSONObject] = []

    for file in try regularFiles(in: root, extensions: ["swift"]).sorted(by: { $0.path < $1.path }) {
        let text = try String(contentsOf: file, encoding: .utf8)
        for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            let value = String(line)
            let range = NSRange(value.startIndex ..< value.endIndex, in: value)
            guard let match = expression.firstMatch(in: value, range: range),
                  let moduleRange = Range(match.range(at: 1), in: value)
            else {
                continue
            }

            let module = String(value[moduleRange])
            if forbidden.contains(module) {
                hits.append([
                    "file": relativePath(file, to: repository),
                    "line": index + 1,
                    "module": module
                ])
            }
        }
    }
    return hits
}

private func publicMLXLeaks(in root: URL, repository: URL) throws -> [JSONObject] {
    guard FileManager.default.fileExists(atPath: root.path) else {
        return []
    }

    let declaration = try NSRegularExpression(
        pattern: #"\bpublic\s+(?:(?:nonisolated|static|final|class|mutating|nonmutating)\s+)*(func|init|let|var|subscript|typealias)\b"#
    )
    let mlxToken = try NSRegularExpression(
        pattern: #"\b(?:MLX[A-Za-z0-9_]*|ModelContainer|GenerateParameters|GenerateCompletionInfo|UserInput|ToolCall)\b"#
    )
    var hits: [JSONObject] = []

    for file in try regularFiles(in: root, extensions: ["swift"]).sorted(by: { $0.path < $1.path }) {
        let lines = try String(contentsOf: file, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var index = 0
        while index < lines.count {
            let line = lines[index]
            let lineRange = NSRange(line.startIndex ..< line.endIndex, in: line)
            guard let match = declaration.firstMatch(in: line, range: lineRange),
                  let kindRange = Range(match.range(at: 1), in: line)
            else {
                index += 1
                continue
            }

            let kind = String(line[kindRange])
            var block = [line.trimmingCharacters(in: .whitespaces)]
            var end = index
            if ["func", "init", "subscript"].contains(kind), !line.contains("{") {
                while end + 1 < lines.count, end - index < 30 {
                    end += 1
                    block.append(lines[end].trimmingCharacters(in: .whitespaces))
                    if lines[end].contains("{") { break }
                }
            }

            let normalized = block.filter { !$0.isEmpty }.joined(separator: " ")
            let normalizedRange = NSRange(normalized.startIndex ..< normalized.endIndex, in: normalized)
            if mlxToken.firstMatch(in: normalized, range: normalizedRange) != nil {
                hits.append([
                    "file": relativePath(file, to: repository),
                    "line": index + 1,
                    "text": normalized
                ])
            }
            index = end + 1
        }
    }
    return hits
}

private func relativePath(_ file: URL, to root: URL) -> String {
    file.resolvingSymlinksInPath().path.replacingOccurrences(
        of: root.resolvingSymlinksInPath().path + "/",
        with: ""
    )
}

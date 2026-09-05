import Foundation

/// Import attributes are open-ended (`@_exported`, `@_spi(...)`, etc.), so match their
/// grammar rather than enumerating spellings. Access modifiers and scoped-import kinds are
/// finite parts of the Swift import declaration grammar.
let swiftImportDeclarationPattern =
    #"^\s*(?:@[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?\s+)*(?:(?:private|fileprivate|internal|package|public|open)\s+)?import\s+(?:(?:typealias|struct|class|enum|protocol|let|var|func|macro)\s+)?([A-Za-z_][A-Za-z0-9_]*)\b"#

func swiftImportedModule(in line: String, matching expression: NSRegularExpression) -> String? {
    let range = NSRange(line.startIndex ..< line.endIndex, in: line)
    guard let match = expression.firstMatch(in: line, range: range),
          let moduleRange = Range(match.range(at: 1), in: line)
    else {
        return nil
    }

    return String(line[moduleRange])
}

func swiftImportedModules(
    in source: String,
    matching expression: NSRegularExpression
) -> [(module: String, line: Int)] {
    enum LexicalState {
        case normal
        case string(rawHashCount: Int, quoteCount: Int)
        case regex(rawHashCount: Int)
        case lineComment
        case blockComment(depth: Int)
    }

    let characters = Array(source)
    var state = LexicalState.normal
    var statement = ""
    var statementLine = 1
    var line = 1
    var index = 0
    var declarations: [(module: String, line: Int)] = []

    func appendStatement() {
        if let module = swiftImportedModule(in: statement, matching: expression) {
            declarations.append((module, statementLine))
        }
        statement = ""
        statementLine = line
    }

    func stringOpening(at start: Int) -> (rawHashCount: Int, quoteCount: Int, length: Int)? {
        var cursor = start
        var rawHashCount = 0
        while characters.indices.contains(cursor), characters[cursor] == "#" {
            rawHashCount += 1
            cursor += 1
        }
        guard characters.indices.contains(cursor), characters[cursor] == "\"" else {
            return nil
        }

        let hasTripleQuote = characters.indices.contains(cursor + 2)
            && characters[cursor + 1] == "\""
            && characters[cursor + 2] == "\""
        let quoteCount = hasTripleQuote ? 3 : 1
        return (rawHashCount, quoteCount, rawHashCount + quoteCount)
    }

    func stringClosingLength(at start: Int, rawHashCount: Int, quoteCount: Int) -> Int? {
        if rawHashCount > 0, start > rawHashCount {
            let hashStart = start - rawHashCount
            let hasEscapeHashes = characters[hashStart ..< start].allSatisfy { $0 == "#" }
            if hasEscapeHashes, characters[hashStart - 1] == "\\" {
                return nil
            }
        }
        for offset in 0 ..< quoteCount
            where !characters.indices.contains(start + offset) || characters[start + offset] != "\""
        {
            return nil
        }
        let hashStart = start + quoteCount
        for offset in 0 ..< rawHashCount
            where !characters.indices.contains(hashStart + offset) || characters[hashStart + offset] != "#"
        {
            return nil
        }
        return quoteCount + rawHashCount
    }

    func regexOpening(at start: Int) -> (rawHashCount: Int, length: Int)? {
        var cursor = start
        var rawHashCount = 0
        while characters.indices.contains(cursor), characters[cursor] == "#" {
            rawHashCount += 1
            cursor += 1
        }
        guard characters.indices.contains(cursor),
              characters[cursor] == "/"
        else {
            return nil
        }

        if rawHashCount == 0 {
            let trimmed = statement.trimmingCharacters(in: .whitespaces)
            let expressionPrefixes = "=([{,:;!&|?"
            let expressionKeywords: Set<String> = [
                "await", "case", "consume", "copy", "discard", "in", "return", "throw", "try", "yield"
            ]
            let lastWord = trimmed.split { !$0.isLetter && !$0.isNumber && $0 != "_" }.last.map(String.init)
            guard trimmed.isEmpty
                || trimmed.last.map(expressionPrefixes.contains) == true
                || lastWord.map(expressionKeywords.contains) == true
            else {
                return nil
            }
        }

        return (rawHashCount, rawHashCount + 1)
    }

    func regexClosingLength(at start: Int, rawHashCount: Int) -> Int? {
        var cursor = start
        var precedingBackslashes = 0
        while cursor > 0, characters[cursor - 1] == "\\" {
            precedingBackslashes += 1
            cursor -= 1
        }
        guard precedingBackslashes.isMultiple(of: 2) else {
            return nil
        }
        guard characters.indices.contains(start), characters[start] == "/" else {
            return nil
        }

        for offset in 0 ..< rawHashCount
            where !characters.indices.contains(start + 1 + offset) || characters[start + 1 + offset] != "#"
        {
            return nil
        }
        return rawHashCount + 1
    }

    func appendCharacters(from start: Int, count: Int) {
        for offset in 0 ..< count {
            statement.append(characters[start + offset])
        }
    }

    while index < characters.count {
        let character = characters[index]
        let next = characters.indices.contains(index + 1) ? characters[index + 1] : nil
        switch state {
        case .normal:
            if character == "/", next == "/" {
                state = .lineComment
                index += 1
            }
            else if character == "/", next == "*" {
                statement.append(" ")
                state = .blockComment(depth: 1)
                index += 1
            }
            else if let opening = stringOpening(at: index) {
                appendCharacters(from: index, count: opening.length)
                state = .string(
                    rawHashCount: opening.rawHashCount,
                    quoteCount: opening.quoteCount
                )
                index += opening.length - 1
            }
            else if let opening = regexOpening(at: index) {
                appendCharacters(from: index, count: opening.length)
                state = .regex(rawHashCount: opening.rawHashCount)
                index += opening.length - 1
            }
            else if character == ";" {
                appendStatement()
            }
            else if character == "\n" {
                appendStatement()
                line += 1
                statementLine = line
            }
            else {
                statement.append(character)
            }

        case let .string(rawHashCount, quoteCount):
            if rawHashCount == 0, character == "\\", next != nil {
                statement.append(character)
                index += 1
                statement.append(characters[index])
            }
            else if let closingLength = stringClosingLength(
                at: index,
                rawHashCount: rawHashCount,
                quoteCount: quoteCount
            ) {
                appendCharacters(from: index, count: closingLength)
                state = .normal
                index += closingLength - 1
            }
            else {
                statement.append(character)
            }
            if character == "\n" {
                line += 1
            }

        case let .regex(rawHashCount):
            if let closingLength = regexClosingLength(at: index, rawHashCount: rawHashCount) {
                appendCharacters(from: index, count: closingLength)
                state = .normal
                index += closingLength - 1
            }
            else {
                statement.append(character)
            }
            if character == "\n" {
                line += 1
            }

        case .lineComment:
            if character == "\n" {
                appendStatement()
                line += 1
                statementLine = line
                state = .normal
            }

        case let .blockComment(depth):
            if character == "/", next == "*" {
                state = .blockComment(depth: depth + 1)
                index += 1
            }
            else if character == "*", next == "/" {
                state = depth == 1 ? .normal : .blockComment(depth: depth - 1)
                index += 1
            }
            else if character == "\n" {
                line += 1
            }
        }
        index += 1
    }
    appendStatement()
    return declarations
}

// MARK: - ArchitectureStage

enum ArchitectureStage: String, Sendable {
    case legacyRatchet = "legacy-ratchet"
    case coreBoundary = "core-boundary"
    case consumerBoundary = "consumer-boundary"
}

func architectureReport(
    contract: ArchitectureContract,
    coreGuards: CoreGuardContract? = nil,
    stage: ArchitectureStage,
    paths: WorkspacePaths,
    developerDirectory: URL = URL(fileURLWithPath: "/Applications/Xcode.app/Contents/Developer")
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
    if let coreGuards {
        report["external_consumer_boundary"] = try externalConsumerBoundaryReport(
            fixture: paths.fixture,
            expectedPackage: paths.package,
            contract: coreGuards,
            developerDirectory: developerDirectory
        )
        report["semantic_parity_schema"] = paritySchemaReport(coreGuards.parity)
    }

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

    case .consumerBoundary,
         .coreBoundary:
        let coreRoot = paths.package.appendingPathComponent("Sources/\(contract.goalCoreTarget)")
        let coreImports = try imports(in: coreRoot, forbidden: forbidden, repository: paths.repository)
        let targetPresent = packageManifest.contains("name: \"\(contract.goalCoreTarget)\"")
            && FileManager.default.fileExists(atPath: coreRoot.path)
        report["core_target_present"] = targetPresent
        report["core_forbidden_imports"] = coreImports
        let compiler: JSONObject
        let dependencies: JSONObject
        if targetPresent, let coreGuards {
            compiler = try compilerPublicAPIReport(
                target: contract.goalCoreTarget,
                paths: paths,
                developerDirectory: developerDirectory,
                contract: coreGuards
            )
            dependencies = try coreTargetDependencyReport(
                target: contract.goalCoreTarget,
                paths: paths,
                developerDirectory: developerDirectory,
                contract: coreGuards
            )
        }
        else {
            compiler = [
                "status": "unmet",
                "reason": "target \(contract.goalCoreTarget) is absent",
                "passed": false
            ]
            dependencies = [
                "status": "unmet",
                "reason": "target \(contract.goalCoreTarget) is absent",
                "passed": false
            ]
        }
        report["compiler_public_api"] = compiler
        report["core_target_dependencies"] = dependencies
        let corePassed = targetPresent
            && coreImports.isEmpty
            && compiler["passed"] as? Bool == true
            && dependencies["passed"] as? Bool == true
        if stage == .consumerBoundary {
            let consumerPassed = (report["external_consumer_boundary"] as? JSONObject)?["passed"] as? Bool == true
            report["passed"] = corePassed && consumerPassed
        }
        else {
            report["passed"] = corePassed
        }
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

    let expression = try NSRegularExpression(pattern: swiftImportDeclarationPattern)
    var hits: [JSONObject] = []

    for file in try regularFiles(in: root, extensions: ["swift"]).sorted(by: { $0.path < $1.path }) {
        let text = try String(contentsOf: file, encoding: .utf8)
        for declaration in swiftImportedModules(in: text, matching: expression) {
            if forbidden.contains(declaration.module) {
                hits.append([
                    "file": relativePath(file, to: repository),
                    "line": declaration.line,
                    "module": declaration.module
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

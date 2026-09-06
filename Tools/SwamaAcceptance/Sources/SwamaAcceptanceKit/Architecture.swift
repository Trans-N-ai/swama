import Foundation

func compilerImportedModules(
    in file: URL,
    developerDirectory: URL
) throws -> [(module: String, line: Int)] {
    let swift = developerDirectory.appendingPathComponent(
        "Toolchains/XcodeDefault.xctoolchain/usr/bin/swift"
    )
    guard FileManager.default.isExecutableFile(atPath: swift.path) else {
        throw AcceptanceFailure.unknown("missing Swift frontend: \(swift.path)")
    }

    let sentinelModule = "__SwamaAcceptanceImportSentinel"
    let source = try String(contentsOf: file, encoding: .utf8)
    let expandedSource = sourceRemovingConditionalCompilationDirectives(source)
    let inputs = expandedSource == source ? [source] : [source, expandedSource]
    var declarations: [(module: String, line: Int, offset: Int)] = []
    var seen: Set<String> = []

    for input in inputs {
        for declaration in try compilerImportedModules(
            in: input,
            sourceFile: file,
            swift: swift,
            developerDirectory: developerDirectory,
            sentinelModule: sentinelModule
        ) {
            let key = "\(declaration.offset)\u{0}\(declaration.module)"
            if seen.insert(key).inserted {
                declarations.append(declaration)
            }
        }
    }

    return declarations
        .sorted { lhs, rhs in
            lhs.offset == rhs.offset ? lhs.module < rhs.module : lhs.offset < rhs.offset
        }
        .map { ($0.module, $0.line) }
}

private func compilerImportedModules(
    in source: String,
    sourceFile: URL,
    swift: URL,
    developerDirectory: URL,
    sentinelModule: String
) throws -> [(module: String, line: Int, offset: Int)] {
    let compilerSource = "\(source)\nimport \(sentinelModule)\n"
    let compilerInput = FileManager.default
        .temporaryDirectory
        .appendingPathComponent("swama-imports-\(UUID().uuidString).swift")
    defer { try? FileManager.default.removeItem(at: compilerInput) }
    try Data(compilerSource.utf8).write(to: compilerInput, options: .atomic)

    let result = try runCommand(
        [
            swift.path,
            "-frontend",
            "-dump-parse",
            "-enable-bare-slash-regex",
            compilerInput.path
        ],
        currentDirectory: compilerInput.deletingLastPathComponent(),
        environment: developerEnvironment(developerDirectory),
        timeout: 30,
        sampleMemory: false,
        timeoutFailureKind: .unknown,
        timeoutContext: "Swift import parser"
    )
    guard result.returnCode == 0 else {
        throw AcceptanceFailure.unknown(
            "cannot parse Swift imports in \(sourceFile.lastPathComponent):\n\(commandFailureSummary(result))"
        )
    }

    do {
        let declarations = try parseCompilerImportAST(
            result.stdout,
            sentinelModule: sentinelModule
        )
        guard declarations.count(where: { $0.module == sentinelModule }) == 1 else {
            throw AcceptanceFailure.unknown("Swift import parser omitted or duplicated its sentinel")
        }

        return declarations.filter { $0.module != sentinelModule }
    }
    catch {
        throw AcceptanceFailure.unknown(
            "cannot interpret Swift import AST for \(sourceFile.lastPathComponent): \(error)"
        )
    }
}

func parseCompilerImportAST(
    _ output: String,
    sentinelModule: String
) throws -> [(module: String, line: Int, offset: Int)] {
    try validateCompilerImportAST(output, sentinelModule: sentinelModule)

    let moduleExpression = try NSRegularExpression(pattern: #"module="([^"]+)""#)
    let rangeExpression = try NSRegularExpression(
        pattern: #"range=\[.*:(\d+):(\d+) - line:\d+:\d+\]"#
    )
    var declarations: [(module: String, line: Int, offset: Int)] = []
    for outputLine in output.split(separator: "\n").map(String.init)
        where outputLine.contains("(import_decl")
    {
        guard let module = regexCapture(moduleExpression, in: outputLine, group: 1),
              let lineValue = regexCapture(rangeExpression, in: outputLine, group: 1),
              let columnValue = regexCapture(rangeExpression, in: outputLine, group: 2),
              let line = Int(lineValue),
              let column = Int(columnValue),
              line > 0,
              column > 0,
              let rootModule = module.split(separator: ".").first.map(String.init)
        else {
            throw AcceptanceFailure.unknown(
                "Swift import parser emitted an unsupported import declaration: \(outputLine)"
            )
        }

        let (lineOffset, lineOverflow) = line.multipliedReportingOverflow(by: 1_000_000)
        let (offset, columnOverflow) = lineOffset.addingReportingOverflow(column)
        guard !lineOverflow, !columnOverflow else {
            throw AcceptanceFailure.unknown("Swift import parser emitted an overflowing source location")
        }

        declarations.append((rootModule, line, offset))
    }
    return declarations
}

private func validateCompilerImportAST(
    _ output: String,
    sentinelModule: String
) throws {
    var lines = output.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    while lines.last?.isEmpty == true {
        lines.removeLast()
    }
    guard let first = lines.first, first.hasPrefix("(source_file ") else {
        throw AcceptanceFailure.unknown("Swift import parser emitted a missing or noisy source_file root")
    }
    guard lines.dropFirst().allSatisfy({ !$0.hasPrefix("(") }) else {
        throw AcceptanceFailure.unknown("Swift import parser emitted an additional top-level record")
    }

    let sentinelField = "module=\"\(sentinelModule)\""
    guard output.components(separatedBy: sentinelField).count == 2 else {
        throw AcceptanceFailure.unknown("Swift import parser omitted or duplicated its sentinel")
    }

    let escapedSentinel = NSRegularExpression.escapedPattern(for: sentinelModule)
    let sentinelExpression = try NSRegularExpression(
        pattern: #"^\s+\(import_decl .* module=""# + escapedSentinel + #""\)\)$"#
    )
    guard let last = lines.last else {
        throw AcceptanceFailure.unknown("Swift import parser emitted an empty AST")
    }

    let lastRange = NSRange(last.startIndex ..< last.endIndex, in: last)
    guard sentinelExpression.firstMatch(in: last, range: lastRange)?.range == lastRange else {
        throw AcceptanceFailure.unknown("Swift import parser emitted a truncated or noisy AST suffix")
    }
}

private func regexCapture(
    _ expression: NSRegularExpression,
    in text: String,
    group: Int
) -> String? {
    let range = NSRange(text.startIndex ..< text.endIndex, in: text)
    guard let match = expression.firstMatch(in: text, range: range),
          let capture = Range(match.range(at: group), in: text)
    else {
        return nil
    }

    return String(text[capture])
}

func sourceRemovingConditionalCompilationDirectives(_ source: String) -> String {
    var bytes = Array(source.utf8)
    var lineStart = 0
    while lineStart < bytes.count {
        var lineEnd = lineStart
        while lineEnd < bytes.count, bytes[lineEnd] != 0x0A, bytes[lineEnd] != 0x0D {
            lineEnd += 1
        }

        var tokenStart = lineStart
        while tokenStart < lineEnd, bytes[tokenStart] == 0x20 || bytes[tokenStart] == 0x09 {
            tokenStart += 1
        }
        let directiveTokens = ["#elseif", "#endif", "#else", "#if"]
        let isDirective = directiveTokens.contains { token in
            let tokenBytes = Array(token.utf8)
            guard tokenStart + tokenBytes.count <= lineEnd,
                  Array(bytes[tokenStart ..< tokenStart + tokenBytes.count]) == tokenBytes
            else {
                return false
            }

            let boundary = tokenStart + tokenBytes.count
            return boundary == lineEnd || bytes[boundary] == 0x20 || bytes[boundary] == 0x09
        }
        if isDirective {
            for index in lineStart ..< lineEnd {
                bytes[index] = 0x20
            }
        }

        lineStart = lineEnd
        while lineStart < bytes.count, bytes[lineStart] == 0x0A || bytes[lineStart] == 0x0D {
            lineStart += 1
        }
    }

    return String(decoding: bytes, as: UTF8.self)
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
    let legacyImports = try imports(
        in: swamaKit,
        forbidden: forbidden,
        repository: paths.repository,
        developerDirectory: developerDirectory
    )
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
        let coreImports = try imports(
            in: coreRoot,
            forbidden: forbidden,
            repository: paths.repository,
            developerDirectory: developerDirectory
        )
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
    repository: URL,
    developerDirectory: URL
) throws -> [JSONObject] {
    guard FileManager.default.fileExists(atPath: root.path) else {
        return []
    }

    var hits: [JSONObject] = []

    for file in try regularFiles(in: root, extensions: ["swift"]).sorted(by: { $0.path < $1.path }) {
        for declaration in try compilerImportedModules(in: file, developerDirectory: developerDirectory) {
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

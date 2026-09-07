import Foundation
import SwiftParser
import SwiftSyntax

func compilerImportedModules(
    in file: URL,
    developerDirectory: URL
) throws -> [(module: String, line: Int)] {
    try validateSwiftParserToolchain(developerDirectory)
    let source = try String(contentsOf: file, encoding: .utf8)
    return try parsedSwiftImports(source: source, file: file)
}

private func validateSwiftParserToolchain(_ developerDirectory: URL) throws {
    let swift = developerDirectory.appendingPathComponent(
        "Toolchains/XcodeDefault.xctoolchain/usr/bin/swift"
    )
    guard FileManager.default.isExecutableFile(atPath: swift.path) else {
        throw AcceptanceFailure.unknown("missing Swift 6.3 toolchain for SwiftParser: \(swift.path)")
    }

    let result = try runCommand(
        [swift.path, "--version"],
        currentDirectory: FileManager.default.temporaryDirectory,
        environment: developerEnvironment(developerDirectory),
        timeout: 10,
        sampleMemory: false,
        timeoutFailureKind: .unknown,
        timeoutContext: "SwiftParser toolchain identity"
    )
    guard result.returnCode == 0 else {
        throw AcceptanceFailure.unknown(
            "cannot identify selected Swift toolchain:\n\(commandFailureSummary(result))"
        )
    }

    let versionExpression = try NSRegularExpression(pattern: #"\bSwift version 6\.3(?:\.\d+)?\b"#)
    let range = NSRange(result.stdout.startIndex ..< result.stdout.endIndex, in: result.stdout)
    guard versionExpression.firstMatch(in: result.stdout, range: range) != nil else {
        throw AcceptanceFailure.unknown(
            "SwiftParser revision requires selected Swift 6.3; got: \(result.stdout.trimmingCharacters(in: .whitespacesAndNewlines))"
        )
    }
}

func parsedSwiftImports(
    source: String,
    file: URL
) throws -> [(module: String, line: Int)] {
    let tree = Parser.parse(source: source)
    guard !tree.hasError else {
        throw AcceptanceFailure.unknown("cannot parse Swift imports in \(file.lastPathComponent)")
    }

    let collector = SwiftImportCollector(file: file, tree: tree)
    collector.walk(tree)
    guard !collector.encounteredInvalidImport else {
        throw AcceptanceFailure.unknown("SwiftParser emitted an import without a module")
    }

    return collector.declarations
}

// MARK: - SwiftImportCollector

private final class SwiftImportCollector: SyntaxVisitor {
    private let converter: SourceLocationConverter
    fileprivate var declarations: [(module: String, line: Int)] = []
    fileprivate var encounteredInvalidImport = false

    init(file: URL, tree: SourceFileSyntax) {
        converter = SourceLocationConverter(fileName: file.path, tree: tree)
        super.init(viewMode: .sourceAccurate)
    }

    override func visit(_ node: ImportDeclSyntax) -> SyntaxVisitorContinueKind {
        guard let first = node.path.first, !first.name.text.isEmpty else {
            encounteredInvalidImport = true
            return .skipChildren
        }

        let module = first.name.text
        let location = converter.location(for: node.importKeyword.positionAfterSkippingLeadingTrivia)
        declarations.append((module, location.line))
        return .skipChildren
    }
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

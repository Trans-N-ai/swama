import Foundation

// MARK: - Compiler-derived public API boundary

func compilerPublicAPIReport(
    target: String,
    paths: WorkspacePaths,
    developerDirectory: URL,
    contract: CoreGuardContract
) throws -> JSONObject {
    let scratch = paths.repository.appendingPathComponent(".build/swama-core-symbol-graph")
    let symbolGraphOutput = scratch.appendingPathComponent("symbolgraph")
    if FileManager.default.fileExists(atPath: symbolGraphOutput.path) {
        for file in try regularFiles(in: symbolGraphOutput, extensions: ["json"])
            where file.lastPathComponent.hasSuffix(".symbols.json")
        {
            try FileManager.default.removeItem(at: file)
        }
    }
    else {
        try FileManager.default.createDirectory(
            at: symbolGraphOutput,
            withIntermediateDirectories: true
        )
    }

    var environment = try developerEnvironment(developerDirectory)
    environment["SWIFTPM_MODULECACHE_OVERRIDE"] = scratch.appendingPathComponent("module-cache").path
    let swift = developerDirectory.appendingPathComponent(
        "Toolchains/XcodeDefault.xctoolchain/usr/bin/swift"
    )
    let extractor = developerDirectory.appendingPathComponent(
        "Toolchains/XcodeDefault.xctoolchain/usr/bin/swift-symbolgraph-extract"
    )
    guard FileManager.default.isExecutableFile(atPath: swift.path),
          FileManager.default.isExecutableFile(atPath: extractor.path)
    else {
        throw AcceptanceFailure.unknown("selected Xcode is missing Swift symbol-graph tools")
    }

    let buildResult = try runCommand(
        symbolGraphBuildCommand(
            package: paths.package,
            scratch: scratch,
            target: target,
            swift: swift
        ),
        currentDirectory: paths.repository,
        environment: environment,
        timeout: contract.compilerTimeoutSeconds,
        sampleMemory: false,
        timeoutFailureKind: .unknown,
        timeoutContext: "SwamaCore target build"
    )
    guard buildResult.returnCode == 0 else {
        throw AcceptanceFailure.failed(
            "SwamaCore target build failed:\n\(commandFailureSummary(buildResult))"
        )
    }

    let binPathResult = try runCommand(
        symbolGraphBinPathCommand(package: paths.package, scratch: scratch, swift: swift),
        currentDirectory: paths.repository,
        environment: environment,
        timeout: 60,
        sampleMemory: false,
        timeoutFailureKind: .unknown,
        timeoutContext: "SwamaCore build path"
    )
    guard binPathResult.returnCode == 0 else {
        throw AcceptanceFailure.unknown(
            "cannot locate SwamaCore build products:\n\(commandFailureSummary(binPathResult))"
        )
    }

    let binPath = try strictAbsolutePath(binPathResult.stdout, context: "SwamaCore build path")
    let modules = binPath.appendingPathComponent("Modules")
    guard FileManager.default.fileExists(
        atPath: modules.appendingPathComponent("\(target).swiftmodule").path
    )
    else {
        throw AcceptanceFailure.unknown("SwamaCore build produced no target module")
    }

    let targetInfoResult = try runCommand(
        [swift.path, "-print-target-info"],
        currentDirectory: paths.repository,
        environment: environment,
        timeout: 60,
        sampleMemory: false,
        timeoutFailureKind: .unknown,
        timeoutContext: "Swift target identity"
    )
    guard targetInfoResult.returnCode == 0 else {
        throw AcceptanceFailure.unknown(
            "cannot identify selected Swift target:\n\(commandFailureSummary(targetInfoResult))"
        )
    }

    let targetTriple = try swiftTargetTriple(targetInfoResult.stdout)
    let sdk = try URL(fileURLWithPath: environmentValue(environment, key: "SDKROOT"))
    let clangArguments = try symbolGraphClangArguments(
        description: binPath.appendingPathComponent("description.json"),
        target: target
    )

    let extractResult = try runCommand(
        symbolGraphExtractCommand(
            extractor: extractor,
            target: target,
            targetTriple: targetTriple,
            sdk: sdk,
            modules: modules,
            output: symbolGraphOutput,
            clangArguments: clangArguments
        ),
        currentDirectory: paths.repository,
        environment: environment,
        timeout: contract.compilerTimeoutSeconds,
        sampleMemory: false,
        timeoutFailureKind: .unknown,
        timeoutContext: "SwamaCore symbol graph extraction"
    )
    guard extractResult.returnCode == 0 else {
        throw AcceptanceFailure.failed(
            "SwamaCore symbol graph extraction failed:\n\(commandFailureSummary(extractResult))"
        )
    }

    var graphs: [JSONObject] = []
    var graphFiles: [JSONObject] = []
    for file in try regularFiles(in: symbolGraphOutput, extensions: ["json"])
        .filter({ $0.lastPathComponent.hasSuffix(".symbols.json") })
        .sorted(by: { $0.path < $1.path })
    {
        let graph = try loadJSONObject(file)
        guard (try? graph.object("module").string("name")) == target else {
            continue
        }

        graphs.append(graph)
        try graphFiles.append([
            "file": relativePathForGuard(file, to: symbolGraphOutput),
            "sha256": sha256File(file)
        ])
    }
    guard !graphs.isEmpty else {
        throw AcceptanceFailure.unknown("compiler produced no symbol graph for target: \(target)")
    }

    var report = try analyzePublicAPISymbolGraphs(
        graphs,
        target: target,
        allowedModules: Set(contract.allowedPublicModules),
        developerDirectory: developerDirectory
    )
    let reachabilityHits = try publicReachabilityAttributeHits(
        in: paths.package.appendingPathComponent("Sources/\(target)"),
        repository: paths.repository
    )
    report["status"] = "ready"
    report["symbol_graph_files"] = graphFiles
    report["reachability_attribute_hits"] = reachabilityHits
    report["passed"] = report["passed"] as? Bool == true && reachabilityHits.isEmpty
    report["duration_ms"] = buildResult.durationMilliseconds
        + binPathResult.durationMilliseconds
        + targetInfoResult.durationMilliseconds
        + extractResult.durationMilliseconds
    return report
}

func symbolGraphBuildCommand(
    package: URL,
    scratch: URL,
    target: String,
    swift: URL
) -> [String] {
    [
        swift.path,
        "build",
        "--package-path",
        package.path,
        "--scratch-path",
        scratch.path,
        "--force-resolved-versions",
        "--target",
        target
    ]
}

func symbolGraphBinPathCommand(package: URL, scratch: URL, swift: URL) -> [String] {
    [
        swift.path,
        "build",
        "--package-path",
        package.path,
        "--scratch-path",
        scratch.path,
        "--force-resolved-versions",
        "--show-bin-path"
    ]
}

func symbolGraphExtractCommand(
    extractor: URL,
    target: String,
    targetTriple: String,
    sdk: URL,
    modules: URL,
    output: URL,
    clangArguments: [String] = []
) -> [String] {
    [
        extractor.path,
        "-module-name",
        target,
        "-target",
        targetTriple,
        "-sdk",
        sdk.path,
        "-I",
        modules.path,
        "-minimum-access-level",
        "public",
        "-skip-synthesized-members",
        "-emit-extension-block-symbols",
        "-output-dir",
        output.path
    ] + clangArguments
}

func symbolGraphClangArguments(description: URL, target: String) throws -> [String] {
    let root = try loadJSONObject(description)
    let commands = try root.object("swiftCommands")
    let matches = try commands.values.compactMap { raw -> JSONObject? in
        guard let command = raw as? JSONObject else {
            throw AcceptanceFailure.unknown("Swift build description command is not an object")
        }

        return try command.string("moduleName") == target ? command : nil
    }
    guard matches.count == 1, let command = matches.first else {
        throw AcceptanceFailure.unknown("Swift build description has no unique command for \(target)")
    }

    let arguments = try command.array("otherArguments")
    var result: [String] = []
    var index = 0
    while index < arguments.count {
        guard let argument = arguments[index] as? String else {
            throw AcceptanceFailure.unknown("Swift build description has a non-string argument")
        }

        if argument == "-Xcc" {
            guard index + 1 < arguments.count,
                  let clangArgument = arguments[index + 1] as? String,
                  !clangArgument.isEmpty
            else {
                throw AcceptanceFailure.unknown("Swift build description has a truncated -Xcc argument")
            }

            result.append(contentsOf: [argument, clangArgument])
            index += 2
        }
        else {
            index += 1
        }
    }
    return result
}

func swiftTargetTriple(_ output: String) throws -> String {
    let object: JSONObject
    do {
        guard let decoded = try JSONSerialization.jsonObject(with: Data(output.utf8)) as? JSONObject else {
            throw AcceptanceFailure.unknown("Swift target identity root is not an object")
        }

        object = decoded
    }
    catch let error as AcceptanceFailure {
        throw error
    }
    catch {
        throw AcceptanceFailure.unknown("Swift target identity is invalid JSON: \(error)")
    }

    let target = try object.object("target")
    let triple = try target.string("triple")
    let unversionedTriple = try target.string("unversionedTriple")
    let platform = try target.string("platform")
    let arch = try target.string("arch")
    guard platform == "macosx",
          ["arm64", "x86_64"].contains(arch),
          unversionedTriple == "\(arch)-apple-macosx"
    else {
        throw AcceptanceFailure.unknown("Swift target identity is not a supported macOS target")
    }

    let expression = try NSRegularExpression(
        pattern: "^\(NSRegularExpression.escapedPattern(for: unversionedTriple))"
            + #"[0-9]+(?:\.[0-9]+){0,2}$"#
    )
    let range = NSRange(triple.startIndex ..< triple.endIndex, in: triple)
    guard expression.firstMatch(in: triple, range: range)?.range == range else {
        throw AcceptanceFailure.unknown("Swift target identity has an invalid versioned triple")
    }

    return triple
}

private func strictAbsolutePath(_ output: String, context: String) throws -> URL {
    let lines = output.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    guard lines.count == 2,
          lines[1].isEmpty,
          lines[0].hasPrefix("/")
    else {
        throw AcceptanceFailure.unknown("\(context) emitted an unsupported path")
    }

    return URL(fileURLWithPath: lines[0]).standardizedFileURL
}

private func environmentValue(_ environment: [String: String], key: String) throws -> String {
    guard let value = environment[key], !value.isEmpty else {
        throw AcceptanceFailure.unknown("selected toolchain environment is missing \(key)")
    }

    return value
}

func analyzePublicAPISymbolGraphs(
    _ graphs: [JSONObject],
    target: String,
    allowedModules: Set<String>,
    developerDirectory: URL = URL(fileURLWithPath: "/Applications/Xcode.app/Contents/Developer")
) throws -> JSONObject {
    let localExtensionIdentifiers = try localExtensionIdentifiers(in: graphs, target: target)
    let moduleResolver = try PreciseIdentifierModuleResolver(
        developerDirectory: developerDirectory,
        preciseIdentifiers: preciseIdentifiersNeedingResolution(in: graphs, target: target)
    )
    let knownAccessLevels: Set<String> = ["fileprivate", "internal", "open", "package", "private", "public"]
    let knownRelationshipKinds: Set<String> = [
        "conformsTo",
        "defaultImplementationOf",
        "extensionTo",
        "inheritsFrom",
        "memberOf",
        "optionalRequirementOf",
        "overloadOf",
        "overrides",
        "requirementOf"
    ]
    var canonicalSymbols: [JSONObject] = []
    var violations: [JSONObject] = []

    for graph in graphs {
        guard try graph.object("module").string("name") == target else {
            continue
        }

        let symbols = try strictObjectArray(graph, key: "symbols", context: "symbol graph")
        let relationships = try strictObjectArray(graph, key: "relationships", context: "symbol graph")
        var relationshipsBySource: [String: [JSONObject]] = [:]
        for relationship in relationships {
            let source = try relationship.string("source")
            let relationshipKind = try relationship.string("kind")
            guard knownRelationshipKinds.contains(relationshipKind) else {
                throw AcceptanceFailure.unknown(
                    "public symbol relationship kind is unknown: \(relationshipKind)"
                )
            }

            _ = try relationship.string("target")
            if relationship["targetFallback"] != nil {
                _ = try relationship.string("targetFallback")
            }
            relationshipsBySource[source, default: []].append(relationship)
        }

        for symbol in symbols {
            let access = try symbol.string("accessLevel")
            guard knownAccessLevels.contains(access) else {
                throw AcceptanceFailure.unknown("public symbol accessLevel is unknown: \(access)")
            }
            guard access == "public" || access == "open" else {
                continue
            }

            let identifier = try symbol.object("identifier").string("precise")
            let kind = try symbol.object("kind").string("identifier")
            let symbolRelationships = relationshipsBySource[identifier] ?? []
            if kind == "swift.extension" {
                let extensionTargets = try symbolRelationships.filter {
                    try $0.string("kind") == "extensionTo"
                }
                let extensionTarget = try extensionTargets.first?.string("target")
                guard identifier.hasPrefix("s:e:"),
                      extensionTargets.count == 1,
                      extensionTarget?.hasPrefix("s:e:") == false,
                      extensionTarget != identifier
                else {
                    throw AcceptanceFailure.unknown(
                        "public extension symbol is missing its unique extensionTo relationship: \(identifier)"
                    )
                }
            }
            let path = try strictStringArray(symbol, key: "pathComponents", context: "public symbol")
            let fragments = try strictObjectArray(
                symbol,
                key: "declarationFragments",
                context: "public symbol"
            )
            let declaration = fragments.compactMap { $0["spelling"] as? String }.joined()
            let canonicalFragments = try fragments.map { fragment -> JSONObject in
                let fragmentKind = try fragment.string("kind")
                let allowedFragmentKinds: Set<String> = [
                    "attribute",
                    "externalParam",
                    "genericParameter",
                    "identifier",
                    "internalParam",
                    "keyword",
                    "number",
                    "string",
                    "text",
                    "typeIdentifier"
                ]
                guard allowedFragmentKinds.contains(fragmentKind) else {
                    throw AcceptanceFailure.unknown(
                        "public declaration fragment kind is unknown: \(fragmentKind)"
                    )
                }

                var value: JSONObject = try [
                    "kind": fragmentKind,
                    "spelling": fragment.string("spelling")
                ]
                if fragment["preciseIdentifier"] != nil {
                    guard ["attribute", "typeIdentifier"].contains(fragmentKind) else {
                        throw AcceptanceFailure.unknown(
                            "unsupported declaration fragment carries preciseIdentifier: \(fragmentKind)"
                        )
                    }

                    let precise = try fragment.string("preciseIdentifier")
                    guard let module = try moduleResolver.moduleName(in: precise) else {
                        throw AcceptanceFailure.unknown(
                            "public declaration has an unparseable preciseIdentifier: \(precise)"
                        )
                    }

                    value["precise_identifier"] = precise
                    value["module"] = module
                }
                return value
            }
            var references: Set<String> = []

            for fragment in fragments
                where ["attribute", "typeIdentifier"].contains(fragment["kind"] as? String ?? "")
            {
                guard fragment["preciseIdentifier"] != nil else {
                    continue
                }

                let precise = try fragment.string("preciseIdentifier")
                guard let module = try moduleResolver.moduleName(in: precise) else {
                    throw AcceptanceFailure.unknown(
                        "public typeIdentifier has an unparseable module: \(precise)"
                    )
                }

                references.insert(module)
                if !allowedModules.contains(module) {
                    violations.append(publicAPIViolation(
                        identifier: identifier,
                        path: path,
                        module: module,
                        source: "declaration"
                    ))
                }
            }

            var conformances: [String] = []
            for relationship in symbolRelationships {
                let relationshipKind = try relationship.string("kind")
                let targetIdentifier = try relationship.string("target")
                let fallback = try relationship["targetFallback"] == nil
                    ? nil
                    : relationship.string("targetFallback")
                let module: String
                if relationshipKind == "memberOf",
                   localExtensionIdentifiers.contains(targetIdentifier)
                {
                    module = target
                }
                else {
                    guard let resolved = try moduleResolver.moduleName(in: targetIdentifier) else {
                        throw AcceptanceFailure.unknown(
                            "public relationship has an unparseable target: \(targetIdentifier)"
                        )
                    }

                    module = resolved
                }

                references.insert(module)
                conformances.append(fallback ?? targetIdentifier)
                if !allowedModules.contains(module) {
                    violations.append(publicAPIViolation(
                        identifier: identifier,
                        path: path,
                        module: module,
                        source: relationshipKind
                    ))
                }
            }

            canonicalSymbols.append([
                "identifier": identifier,
                "kind": kind,
                "path": path,
                "declaration": declaration,
                "declaration_fragments": canonicalFragments,
                "referenced_modules": references.sorted(),
                "conformances": conformances.sorted()
            ])
        }
    }

    canonicalSymbols.sort { lhs, rhs in
        (lhs["identifier"] as? String ?? "") < (rhs["identifier"] as? String ?? "")
    }
    let uniqueViolations = Dictionary(grouping: violations) { violation in
        [
            violation["identifier"] as? String ?? "",
            violation["module"] as? String ?? "",
            violation["source"] as? String ?? ""
        ].joined(separator: "\u{0}")
    }
    .values
    .compactMap(\.first)
    .sorted { lhs, rhs in
        let left = "\(lhs["identifier"] ?? ""):\(lhs["module"] ?? ""):\(lhs["source"] ?? "")"
        let right = "\(rhs["identifier"] ?? ""):\(rhs["module"] ?? ""):\(rhs["source"] ?? "")"
        return left < right
    }

    return try [
        "target": target,
        "graph_count": graphs.count,
        "symbol_count": canonicalSymbols.count,
        "allowed_modules": allowedModules.sorted(),
        "symbols": canonicalSymbols,
        "manifest_sha256": sha256(compactJSONData(canonicalSymbols)),
        "violations": uniqueViolations,
        "passed": uniqueViolations.isEmpty
    ]
}

func publicReachabilityAttributeHits(in root: URL, repository: URL) throws -> [JSONObject] {
    guard FileManager.default.fileExists(atPath: root.path) else {
        return []
    }

    let expression = try NSRegularExpression(
        pattern: #"^\s*(?:@[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?\s+)*@(inlinable|usableFromInline)\b"#
    )
    var hits: [JSONObject] = []
    for file in try regularFiles(in: root, extensions: ["swift"]).sorted(by: { $0.path < $1.path }) {
        let lines = try String(contentsOf: file, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        for (index, line) in lines.enumerated() {
            let range = NSRange(line.startIndex ..< line.endIndex, in: line)
            for match in expression.matches(in: line, range: range) {
                guard let attributeRange = Range(match.range(at: 1), in: line) else {
                    continue
                }

                hits.append([
                    "file": relativePathForGuard(file, to: repository),
                    "line": index + 1,
                    "attribute": "@\(line[attributeRange])"
                ])
            }
        }
    }
    return hits
}

private func publicAPIViolation(
    identifier: String,
    path: [String],
    module: String,
    source: String
) -> JSONObject {
    [
        "identifier": identifier,
        "path": path,
        "module": module,
        "source": source
    ]
}

private func strictObjectArray(_ object: JSONObject, key: String, context: String) throws -> [JSONObject] {
    try object.array(key).enumerated().map { index, value in
        guard let value = value as? JSONObject else {
            throw AcceptanceFailure.unknown(
                "\(context) \(key)[\(index)] is not an object"
            )
        }

        return value
    }
}

private func strictStringArray(_ object: JSONObject, key: String, context: String) throws -> [String] {
    try object.array(key).enumerated().map { index, value in
        guard let value = value as? String else {
            throw AcceptanceFailure.unknown(
                "\(context) \(key)[\(index)] is not a string"
            )
        }

        return value
    }
}

private func optionalStrictStringArray(_ object: JSONObject, key: String, context: String) throws -> [String] {
    guard object[key] != nil else {
        return []
    }

    return try strictStringArray(object, key: key, context: context)
}

// MARK: - Precise identifier discovery

private func localExtensionIdentifiers(
    in graphs: [JSONObject],
    target: String
) throws -> Set<String> {
    var identifiers: Set<String> = []
    for graph in graphs where try graph.object("module").string("name") == target {
        for symbol in try strictObjectArray(graph, key: "symbols", context: "symbol graph") {
            let precise = try symbol.object("identifier").string("precise")
            let kind = try symbol.object("kind").string("identifier")
            let access = try symbol.string("accessLevel")
            if ["public", "open"].contains(access),
               kind == "swift.extension",
               precise.hasPrefix("s:e:")
            {
                identifiers.insert(precise)
            }
        }
    }
    return identifiers
}

private func preciseIdentifiersNeedingResolution(
    in graphs: [JSONObject],
    target: String
) throws -> Set<String> {
    var identifiers: Set<String> = []
    for graph in graphs where try graph.object("module").string("name") == target {
        let symbols = try strictObjectArray(graph, key: "symbols", context: "symbol graph")
        var publicSymbolIdentifiers: Set<String> = []
        for symbol in symbols {
            let access = try symbol.string("accessLevel")
            guard access == "public" || access == "open" else {
                continue
            }

            try publicSymbolIdentifiers.insert(symbol.object("identifier").string("precise"))
            for fragment in try strictObjectArray(
                symbol,
                key: "declarationFragments",
                context: "public symbol"
            ) where fragment["preciseIdentifier"] != nil {
                try identifiers.insert(fragment.string("preciseIdentifier"))
            }
        }
        for relationship in try strictObjectArray(graph, key: "relationships", context: "symbol graph") {
            guard try publicSymbolIdentifiers.contains(relationship.string("source")) else {
                continue
            }

            try identifiers.insert(relationship.string("target"))
        }
    }
    return identifiers
}

// MARK: - PreciseIdentifierModuleResolver

private final class PreciseIdentifierModuleResolver {
    private enum Resolution {
        case module(String)
        case invalid
    }

    private let demangler: URL
    private var cache: [String: Resolution] = [:]

    init(developerDirectory: URL, preciseIdentifiers: Set<String>) throws {
        demangler = developerDirectory.appendingPathComponent(
            "Toolchains/XcodeDefault.xctoolchain/usr/bin/swift-demangle"
        )
        guard FileManager.default.isExecutableFile(atPath: demangler.path) else {
            throw AcceptanceFailure.unknown("missing Swift demangler: \(demangler.path)")
        }

        let swiftIdentifiers = preciseIdentifiers
            .filter { $0.hasPrefix("s:") }
            .sorted()
        guard !swiftIdentifiers.isEmpty else {
            return
        }

        let mangledNames = swiftIdentifiers.map { "$s\($0.dropFirst(2))" }
        let result = try runCommand(
            [demangler.path, "--expand", "--tree-only"] + mangledNames,
            currentDirectory: FileManager.default.temporaryDirectory,
            timeout: 30,
            sampleMemory: false,
            timeoutFailureKind: .unknown,
            timeoutContext: "Swift USR demangler"
        )
        guard result.returnCode == 0, result.stderr.isEmpty else {
            throw AcceptanceFailure.unknown(
                "cannot demangle Swift preciseIdentifiers:\n" + commandFailureSummary(result)
            )
        }

        let modules = try parseBatchDemanglerOutput(
            result.stdout,
            preciseIdentifiers: swiftIdentifiers,
            mangledNames: mangledNames
        )
        for (identifier, module) in zip(swiftIdentifiers, modules) {
            cache[identifier] = module.map(Resolution.module) ?? .invalid
        }
    }

    func moduleName(in preciseIdentifier: String) throws -> String? {
        if let cached = cache[preciseIdentifier] {
            switch cached {
            case let .module(module):
                return module
            case .invalid:
                return nil
            }
        }

        let module = try resolve(preciseIdentifier)
        cache[preciseIdentifier] = module.map(Resolution.module) ?? .invalid
        return module
    }

    private func resolve(_ preciseIdentifier: String) throws -> String? {
        let knownClangModules = ["c:@T@NSTimeInterval": "Foundation"]
        if let module = knownClangModules[preciseIdentifier] {
            return module
        }
        guard preciseIdentifier.hasPrefix("s:") else {
            return preciseIdentifier.contains(":") ? "__foreign__" : nil
        }

        throw AcceptanceFailure.unknown(
            "Swift preciseIdentifier was not included in the sealed demangler batch: \(preciseIdentifier)"
        )
    }
}

func parseBatchDemanglerOutput(
    _ output: String,
    preciseIdentifiers: [String],
    mangledNames: [String]
) throws -> [String?] {
    guard preciseIdentifiers.count == mangledNames.count else {
        throw AcceptanceFailure.unknown("Swift demangler batch identity count is inconsistent")
    }
    guard output.last == "\n", !output.contains("\r") else {
        throw AcceptanceFailure.unknown("Swift demangler batch has an incomplete record terminator")
    }

    let lines = output.dropLast().split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    var lineIndex = 0
    var modules: [String?] = []
    for (preciseIdentifier, mangled) in zip(preciseIdentifiers, mangledNames) {
        guard lineIndex < lines.count,
              lines[lineIndex] == "Demangling for \(mangled)"
        else {
            throw AcceptanceFailure.unknown(
                "Swift demangler batch returned a reordered or unsupported record"
            )
        }

        lineIndex += 1

        guard lineIndex < lines.count else {
            throw AcceptanceFailure.unknown("Swift demangler batch omitted its semantic tree")
        }

        if lines[lineIndex] == "<<NULL>>" {
            modules.append(nil)
            lineIndex += 1
            continue
        }

        guard lines[lineIndex] == "kind=Global" else {
            throw AcceptanceFailure.unknown("Swift demangler batch omitted its Global root")
        }

        lineIndex += 1

        var semanticLines: [String] = []
        while lineIndex < lines.count,
              !lines[lineIndex].hasPrefix("Demangling for ")
        {
            if lines[lineIndex].isEmpty {
                lineIndex += 1
                break
            }
            semanticLines.append(lines[lineIndex])
            lineIndex += 1
        }
        guard !semanticLines.isEmpty else {
            throw AcceptanceFailure.unknown("Swift demangler batch returned an empty Global root")
        }

        var topLevelNodeCount = 0
        var discoveredModules: Set<String> = []
        let moduleExpression = try NSRegularExpression(
            pattern: #"^\s+kind=Module, text="([A-Za-z_][A-Za-z0-9_]*)"$"#
        )
        for semanticLine in semanticLines {
            let indentation = semanticLine.prefix(while: { $0 == " " }).count
            guard indentation >= 2,
                  indentation.isMultiple(of: 2),
                  semanticLine.dropFirst(indentation).hasPrefix("kind=")
            else {
                throw AcceptanceFailure.unknown(
                    "Swift demangler batch returned a malformed semantic tree for \(preciseIdentifier)"
                )
            }

            if indentation == 2 {
                topLevelNodeCount += 1
            }

            let range = NSRange(semanticLine.startIndex ..< semanticLine.endIndex, in: semanticLine)
            if let match = moduleExpression.firstMatch(in: semanticLine, range: range),
               let moduleRange = Range(match.range(at: 1), in: semanticLine)
            {
                discoveredModules.insert(String(semanticLine[moduleRange]))
            }
        }
        guard topLevelNodeCount == 1, discoveredModules.count == 1 else {
            modules.append(nil)
            continue
        }

        modules.append(discoveredModules.first)
    }
    guard lineIndex == lines.count else {
        throw AcceptanceFailure.unknown("Swift demangler batch returned trailing output")
    }

    return modules
}

// MARK: - External consumer and target dependency boundary

func externalConsumerBoundaryReport(
    fixture: URL,
    expectedPackage: URL,
    contract: CoreGuardContract,
    developerDirectory: URL = URL(fileURLWithPath: "/Applications/Xcode.app/Contents/Developer")
) throws -> JSONObject {
    let result = try runCommand(
        [
            "xcrun",
            "swift",
            "package",
            "--package-path",
            fixture.path,
            "dump-package"
        ],
        currentDirectory: fixture,
        environment: developerEnvironment(developerDirectory),
        timeout: 60,
        sampleMemory: false,
        timeoutFailureKind: .unknown,
        timeoutContext: "external consumer manifest"
    )
    guard result.returnCode == 0,
          let data = result.stdout.data(using: .utf8),
          let description = try JSONSerialization.jsonObject(with: data) as? JSONObject
    else {
        throw AcceptanceFailure.unknown(
            "cannot inspect external consumer manifest:\n\(commandFailureSummary(result))"
        )
    }

    var imports: Set<String> = []
    let sourceRoot = fixture.appendingPathComponent("Sources")
    if FileManager.default.fileExists(atPath: sourceRoot.path) {
        for file in try regularFiles(in: sourceRoot, extensions: ["swift"]) {
            for declaration in try compilerImportedModules(
                in: file,
                developerDirectory: developerDirectory
            ) {
                imports.insert(declaration.module)
            }
        }
    }
    return try analyzeExternalConsumerPackage(
        description,
        imports: imports,
        expectedPackagePath: canonicalFilesystemPath(expectedPackage.path),
        contract: contract
    )
}

func analyzeExternalConsumerPackage(
    _ description: JSONObject,
    imports: Set<String>,
    expectedPackagePath: String,
    contract: CoreGuardContract
) throws -> JSONObject {
    let packageDependencies = try strictObjectArray(
        description,
        key: "dependencies",
        context: "external consumer package"
    )
    var packageDescriptors: [JSONObject] = []
    for dependency in packageDependencies {
        guard dependency.count == 1, let kind = dependency.keys.first else {
            throw AcceptanceFailure.unknown("external consumer package dependency has an unsupported shape")
        }

        let items = try strictObjectArray(
            dependency,
            key: kind,
            context: "external consumer package dependency"
        )
        for item in items {
            let identity = try item.string("identity")
            let path: String =
                if kind == "fileSystem" {
                    try canonicalFilesystemPath(item.string("path"))
                }
                else {
                    ""
                }
            packageDescriptors.append([
                "kind": kind,
                "identity": identity,
                "path": path,
                "path_sha256": sha256(Data(path.utf8))
            ])
        }
    }
    let allowedPackageDependencies = packageDescriptors.filter {
        $0["kind"] as? String == "fileSystem"
            && $0["identity"] as? String == "swama"
            && $0["path"] as? String == expectedPackagePath
    }
    let unexpectedPackageDependencies = packageDescriptors.filter {
        !($0["kind"] as? String == "fileSystem"
            && $0["identity"] as? String == "swama"
            && $0["path"] as? String == expectedPackagePath
        )
    }

    let targets = try strictObjectArray(description, key: "targets", context: "external consumer package")
    var targetDependencies: [JSONObject] = []
    for target in targets {
        let targetName = try target.string("name")
        for dependency in try strictObjectArray(
            target,
            key: "dependencies",
            context: "external consumer target \(targetName)"
        ) {
            guard dependency.count == 1, let kind = dependency.keys.first else {
                throw AcceptanceFailure.unknown(
                    "external consumer target dependency has an unsupported shape"
                )
            }

            let parts = try dependency.array(kind)
            guard let name = parts.first as? String else {
                throw AcceptanceFailure.unknown(
                    "external consumer target dependency name is not a string"
                )
            }

            let package: String
            if parts.count > 1, !(parts[1] is NSNull) {
                guard let value = parts[1] as? String else {
                    throw AcceptanceFailure.unknown(
                        "external consumer target dependency package is not a string"
                    )
                }

                package = value
            }
            else {
                package = ""
            }
            targetDependencies.append([
                "target": targetName,
                "kind": kind,
                "name": name,
                "package": package
            ])
        }
    }
    let allowedTargetDependencies = targetDependencies.filter {
        $0["kind"] as? String == "product"
            && $0["name"] as? String == contract.fixtureProduct
            && $0["package"] as? String == "swama"
    }
    let unexpectedTargetDependencies = targetDependencies.filter {
        !($0["kind"] as? String == "product"
            && $0["name"] as? String == contract.fixtureProduct
            && $0["package"] as? String == "swama"
        )
    }
    let products = targetDependencies.compactMap { item -> String? in
        item["kind"] as? String == "product" ? item["name"] as? String : nil
    }
    let unexpectedProducts = Array(Set(products).subtracting([contract.fixtureProduct])).sorted()
    let missingProducts = allowedTargetDependencies.isEmpty ? [contract.fixtureProduct] : []
    let allowedImports = Set(contract.fixtureAllowedImports)
    let unexpectedImports = Array(imports.subtracting(allowedImports)).sorted()
    let missingImports = !imports.contains(contract.fixtureProduct) ? [contract.fixtureProduct] : []
    let passed = targets.count == 1
        && packageDescriptors.count == 1
        && allowedPackageDependencies.count == 1
        && unexpectedPackageDependencies.isEmpty
        && targetDependencies.count == 1
        && allowedTargetDependencies.count == 1
        && unexpectedTargetDependencies.isEmpty
        && unexpectedImports.isEmpty
        && missingImports.isEmpty

    return [
        "status": passed ? "ready" : "unmet",
        "target_count": targets.count,
        "expected_package_path": expectedPackagePath,
        "expected_package_path_sha256": sha256(Data(expectedPackagePath.utf8)),
        "package_dependencies": packageDescriptors,
        "unexpected_package_dependencies": unexpectedPackageDependencies,
        "target_dependencies": targetDependencies,
        "unexpected_target_dependencies": unexpectedTargetDependencies,
        "expected_products": [contract.fixtureProduct],
        "actual_products": products.sorted(),
        "unexpected_products": unexpectedProducts,
        "missing_products": missingProducts,
        "actual_imports": imports.sorted(),
        "unexpected_imports": unexpectedImports,
        "missing_imports": missingImports,
        "passed": passed
    ]
}

private func canonicalFilesystemPath(_ path: String) -> String {
    URL(fileURLWithPath: path)
        .standardizedFileURL
        .resolvingSymlinksInPath()
        .path
}

func coreTargetDependencyReport(
    target: String,
    paths: WorkspacePaths,
    developerDirectory: URL,
    contract: CoreGuardContract
) throws -> JSONObject {
    let environment = try developerEnvironment(developerDirectory)
    let graphResult = try runCommand(
        [
            "xcrun",
            "swift",
            "package",
            "--package-path",
            paths.package.path,
            "--force-resolved-versions",
            "show-dependencies",
            "--format",
            "json"
        ],
        currentDirectory: paths.repository,
        environment: environment,
        timeout: 60,
        sampleMemory: false,
        timeoutFailureKind: .unknown,
        timeoutContext: "SwamaCore resolved package graph"
    )
    guard graphResult.returnCode == 0,
          let data = graphResult.stdout.data(using: .utf8),
          let resolvedGraph = try JSONSerialization.jsonObject(with: data) as? JSONObject
    else {
        throw AcceptanceFailure.unknown(
            "cannot inspect SwamaCore resolved package graph:\n\(commandFailureSummary(graphResult))"
        )
    }

    let descriptors = try resolvedPackageDescriptors(resolvedGraph)
    var manifests: [String: JSONObject] = [:]
    for descriptor in descriptors.sorted(by: { $0.key < $1.key }) {
        let manifestResult = try runCommand(
            [
                "xcrun",
                "swift",
                "package",
                "--package-path",
                descriptor.value.path,
                "dump-package"
            ],
            currentDirectory: URL(fileURLWithPath: descriptor.value.path),
            environment: environment,
            timeout: 60,
            sampleMemory: false,
            timeoutFailureKind: .unknown,
            timeoutContext: "resolved package manifest \(descriptor.key)"
        )
        guard manifestResult.returnCode == 0,
              let manifestData = manifestResult.stdout.data(using: .utf8),
              let manifest = try JSONSerialization.jsonObject(with: manifestData) as? JSONObject
        else {
            throw AcceptanceFailure.unknown(
                "cannot inspect resolved package manifest \(descriptor.key):\n"
                    + commandFailureSummary(manifestResult)
            )
        }

        manifests[descriptor.key] = manifest
    }

    return try analyzeResolvedTargetDependencyGraph(
        rootIdentity: resolvedGraph.string("identity"),
        manifests: manifests,
        packageAliases: descriptors.mapValues(\.directPackageAliases),
        activeTraits: descriptors.mapValues(\.activeTraits),
        target: target,
        forbiddenProducts: Set(contract.forbiddenTransitiveProducts)
    )
}

func analyzeResolvedTargetDependencyGraph(
    rootIdentity: String,
    manifests: [String: JSONObject],
    packageAliases: [String: [String: String]]? = nil,
    activeTraits: [String: Set<String>]? = nil,
    target: String,
    forbiddenProducts: Set<String>
) throws -> JSONObject {
    guard manifests[rootIdentity] != nil else {
        throw AcceptanceFailure.unknown("resolved package graph is missing root manifest: \(rootIdentity)")
    }

    var productsByPackage: [String: [String: PackageProduct]] = [:]
    var targetsByPackage: [String: [String: JSONObject]] = [:]
    var aliasesByPackage = packageAliases ?? [:]
    for (identity, manifest) in manifests {
        productsByPackage[identity] = try packageProducts(manifest)
        targetsByPackage[identity] = try packageTargets(manifest)
        let declaredAliases = try packageDependencyAliases(manifest)
        var aliases = aliasesByPackage[identity] ?? [:]
        for (alias, packageIdentity) in declaredAliases {
            if let existing = aliases[alias], existing != packageIdentity {
                throw AcceptanceFailure.unknown(
                    "package dependency alias is ambiguous: \(identity):\(alias)"
                )
            }
            aliases[alias] = packageIdentity
        }
        aliasesByPackage[identity] = aliases
    }

    let rootProducts = productsByPackage[rootIdentity] ?? [:]
    let rootTargets = targetsByPackage[rootIdentity] ?? [:]
    let libraryProductTargets = rootProducts[target]?.targets ?? []
    let libraryProductPresent = rootProducts[target]?.isLibrary == true
        && libraryProductTargets == [target]
        && rootTargets[target] != nil

    var directTargets: [String] = []
    var directProducts: [String] = []
    var directByName: [String] = []
    if let rootTarget = rootTargets[target] {
        let traits = activeTraits?[rootIdentity] ?? ["default"]
        for dependency in try targetDependencyDescriptors(rootTarget) {
            guard try dependency.isActive(platform: "macos", traits: traits) else {
                continue
            }

            switch dependency.kind {
            case "target":
                directTargets.append("\(rootIdentity):\(dependency.name)")
            case "product":
                directProducts.append("\(dependency.package ?? "<implicit>"):\(dependency.name)")
            case "byName":
                directByName.append(dependency.name)
            default:
                throw AcceptanceFailure.unknown("resolved direct target dependency kind is unsupported")
            }
        }
    }

    var queue = [ResolvedTarget(package: rootIdentity, name: target)]
    var visitedTargets: Set<ResolvedTarget> = []
    var visitedProducts: Set<ResolvedProduct> = []
    var unresolved: Set<String> = []
    while let node = queue.popLast() {
        guard visitedTargets.insert(node).inserted else {
            continue
        }
        guard manifests[node.package] != nil else {
            unresolved.insert("missing package manifest \(node.package)")
            continue
        }

        let targets = targetsByPackage[node.package] ?? [:]
        guard let targetDescription = targets[node.name] else {
            unresolved.insert("missing target \(node.package):\(node.name)")
            continue
        }

        // Macro and plugin targets execute in the host toolchain while building; their own
        // dependencies do not enter the linked runtime closure being audited here. Some Swift
        // versions consequently omit those host-only packages from `show-dependencies`, even
        // though `dump-package` still describes their target edges.
        if let rawType = targetDescription["type"] {
            guard let targetType = rawType as? String, !targetType.isEmpty else {
                throw AcceptanceFailure.unknown(
                    "resolved target type is invalid: \(node.package):\(node.name)"
                )
            }

            if targetType == "macro" || targetType == "plugin" {
                continue
            }
        }

        let aliases = aliasesByPackage[node.package] ?? [:]
        let traits = activeTraits?[node.package] ?? ["default"]

        for dependency in try targetDependencyDescriptors(targetDescription) {
            guard try dependency.isActive(platform: "macos", traits: traits) else {
                continue
            }

            switch dependency.kind {
            case "target":
                queue.append(ResolvedTarget(package: node.package, name: dependency.name))

            case "product":
                resolveProductDependency(
                    name: dependency.name,
                    requestedPackage: dependency.package,
                    sourcePackage: node.package,
                    packageAliases: aliases,
                    productsByPackage: productsByPackage,
                    queue: &queue,
                    visitedProducts: &visitedProducts,
                    unresolved: &unresolved
                )

            case "byName":
                if targets[dependency.name] != nil {
                    queue.append(ResolvedTarget(package: node.package, name: dependency.name))
                }
                else {
                    resolveProductDependency(
                        name: dependency.name,
                        requestedPackage: nil,
                        sourcePackage: node.package,
                        packageAliases: aliases,
                        productsByPackage: productsByPackage,
                        queue: &queue,
                        visitedProducts: &visitedProducts,
                        unresolved: &unresolved
                    )
                }

            default:
                unresolved.insert(
                    "unsupported dependency \(node.package):\(node.name) \(dependency.kind):\(dependency.name)"
                )
            }
        }
    }

    let forbidden = Set(visitedProducts.map(\.name)).intersection(forbiddenProducts).sorted()
    let passed = libraryProductPresent && unresolved.isEmpty && forbidden.isEmpty
    var manifestHashes: JSONObject = [:]
    var resolvedTraits: JSONObject = [:]
    for identity in Set(visitedTargets.map(\.package)).sorted() {
        if let manifest = manifests[identity] {
            manifestHashes[identity] = try sha256(compactJSONData(manifest))
        }
        resolvedTraits[identity] = (activeTraits?[identity] ?? ["default"]).sorted()
    }
    return [
        "status": passed ? "ready" : (libraryProductPresent && unresolved.isEmpty ? "violated" : "unmet"),
        "target": target,
        "root_package": rootIdentity,
        "library_product_present": libraryProductPresent,
        "library_product_targets": libraryProductTargets,
        "direct_target_dependencies": directTargets.sorted(),
        "direct_product_dependencies": directProducts.sorted(),
        "direct_by_name_dependencies": directByName.sorted(),
        "target_dependencies": visitedTargets.map(\.description).sorted(),
        "product_dependencies": visitedProducts.map(\.description).sorted(),
        "forbidden_products": forbidden,
        "unresolved_dependencies": unresolved.sorted(),
        "manifest_sha256": manifestHashes,
        "resolved_package_traits": resolvedTraits,
        "passed": passed
    ]
}

// MARK: - ResolvedPackageDescriptor

private struct ResolvedPackageDescriptor {
    let path: String
    let directPackageAliases: [String: String]
    let activeTraits: Set<String>
}

// MARK: - ResolvedTarget

private struct ResolvedTarget: Hashable {
    let package: String
    let name: String

    var description: String { "\(package):\(name)" }
}

// MARK: - ResolvedProduct

private struct ResolvedProduct: Hashable {
    let package: String
    let name: String

    var description: String { "\(package):\(name)" }
}

// MARK: - PackageProduct

private struct PackageProduct {
    let targets: [String]
    let isLibrary: Bool
}

// MARK: - TargetDependencyDescriptor

private struct TargetDependencyDescriptor {
    let kind: String
    let name: String
    let package: String?
    let condition: JSONObject?

    func isActive(platform: String, traits: Set<String>) throws -> Bool {
        guard let condition else {
            return true
        }
        guard Set(condition.keys).isSubset(of: ["platformNames", "traits"]) else {
            throw AcceptanceFailure.unknown("resolved target dependency condition has unknown keys")
        }

        let platforms = try Set(optionalStrictStringArray(
            condition,
            key: "platformNames",
            context: "resolved target dependency condition"
        ))
        let requiredTraits = try Set(optionalStrictStringArray(
            condition,
            key: "traits",
            context: "resolved target dependency condition"
        ))
        return (platforms.isEmpty || platforms.contains(platform))
            && requiredTraits.isSubset(of: traits)
    }
}

private func resolvedPackageDescriptors(_ root: JSONObject) throws -> [String: ResolvedPackageDescriptor] {
    var result: [String: ResolvedPackageDescriptor] = [:]
    var queue = [root]
    while let node = queue.popLast() {
        let identity = try node.string("identity")
        let path = try node.string("path")
        let dependencies = try strictObjectArray(node, key: "dependencies", context: "resolved package graph")
        var aliases: [String: String] = [:]
        for dependency in dependencies {
            let dependencyIdentity = try dependency.string("identity")
            let dependencyName = try dependency.string("name")
            for alias in [dependencyIdentity, dependencyName] {
                if let existing = aliases[alias], existing != dependencyIdentity {
                    throw AcceptanceFailure.unknown(
                        "resolved package alias is ambiguous: \(identity):\(alias)"
                    )
                }
                aliases[alias] = dependencyIdentity
            }
        }
        let descriptor = try ResolvedPackageDescriptor(
            path: path,
            directPackageAliases: aliases,
            activeTraits: Set(optionalStrictStringArray(
                node,
                key: "traits",
                context: "resolved package graph"
            ))
        )
        if let existing = result[identity],
           existing.path != descriptor.path
           || existing.directPackageAliases != descriptor.directPackageAliases
           || existing.activeTraits != descriptor.activeTraits
        {
            throw AcceptanceFailure.unknown(
                "resolved package identity has inconsistent descriptors: \(identity)"
            )
        }
        result[identity] = descriptor
        queue.append(contentsOf: dependencies)
    }
    return result
}

private func packageProducts(_ manifest: JSONObject) throws -> [String: PackageProduct] {
    var result: [String: PackageProduct] = [:]
    for product in try strictObjectArray(manifest, key: "products", context: "resolved package manifest") {
        let name = try product.string("name")
        guard result[name] == nil else {
            throw AcceptanceFailure.unknown("resolved package has duplicate product: \(name)")
        }

        let type = try product.object("type")
        guard type.count == 1, let productKind = type.keys.first else {
            throw AcceptanceFailure.unknown("resolved package product type has an unsupported shape")
        }

        let isLibrary: Bool
        switch productKind {
        case "library":
            let payload = try strictStringArray(type, key: "library", context: "resolved library product")
            guard payload.count == 1,
                  let linkage = payload.first,
                  ["automatic", "dynamic", "static"].contains(linkage)
            else {
                throw AcceptanceFailure.unknown("resolved library product linkage is invalid")
            }

            isLibrary = true

        case "executable",
             "plugin",
             "snippet",
             "test":
            guard type[productKind] is NSNull else {
                throw AcceptanceFailure.unknown("resolved non-library product payload is not null")
            }

            isLibrary = false

        default:
            throw AcceptanceFailure.unknown("resolved package product kind is unknown: \(productKind)")
        }
        result[name] = try PackageProduct(
            targets: strictStringArray(product, key: "targets", context: "resolved package product"),
            isLibrary: isLibrary
        )
    }
    return result
}

private func packageTargets(_ manifest: JSONObject) throws -> [String: JSONObject] {
    var result: [String: JSONObject] = [:]
    for target in try strictObjectArray(manifest, key: "targets", context: "resolved package manifest") {
        let name = try target.string("name")
        guard result[name] == nil else {
            throw AcceptanceFailure.unknown("resolved package has duplicate target: \(name)")
        }

        result[name] = target
    }
    return result
}

private func packageDependencyAliases(_ manifest: JSONObject) throws -> [String: String] {
    var result: [String: String] = [:]
    for dependency in try strictObjectArray(
        manifest,
        key: "dependencies",
        context: "resolved package manifest"
    ) {
        guard dependency.count == 1 else {
            throw AcceptanceFailure.unknown("resolved package dependency has an unsupported shape")
        }

        for descriptors in dependency.values {
            guard let descriptors = descriptors as? [Any] else {
                throw AcceptanceFailure.unknown("resolved package dependency payload is not an array")
            }

            for (index, value) in descriptors.enumerated() {
                guard let descriptor = value as? JSONObject else {
                    throw AcceptanceFailure.unknown(
                        "resolved package dependency descriptor[\(index)] is not an object"
                    )
                }

                let identity = try descriptor.string("identity")
                var aliases = [identity]
                if descriptor["nameForTargetDependencyResolutionOnly"] != nil {
                    try aliases.append(descriptor.string("nameForTargetDependencyResolutionOnly"))
                }
                for alias in aliases {
                    if let existing = result[alias], existing != identity {
                        throw AcceptanceFailure.unknown(
                            "package dependency alias maps to multiple identities: \(alias)"
                        )
                    }
                    result[alias] = identity
                }
            }
        }
    }
    return result
}

private func targetDependencyDescriptors(_ target: JSONObject) throws -> [TargetDependencyDescriptor] {
    var result: [TargetDependencyDescriptor] = []
    for dependency in try strictObjectArray(target, key: "dependencies", context: "resolved package target") {
        guard dependency.count == 1,
              let kind = dependency.keys.first,
              let parts = dependency[kind] as? [Any],
              ["byName", "product", "target"].contains(kind),
              parts.count == (kind == "product" ? 4 : 2),
              let name = parts.first as? String
        else {
            throw AcceptanceFailure.unknown("resolved target dependency has an unsupported shape")
        }

        let package: String?
        if kind == "product", !(parts[1] is NSNull) {
            guard let value = parts[1] as? String, !value.isEmpty else {
                throw AcceptanceFailure.unknown("resolved product dependency package is missing or empty")
            }

            package = value
        }
        else {
            package = nil
        }
        if kind == "product", !(parts[2] is NSNull) {
            guard let aliases = parts[2] as? JSONObject,
                  aliases.values.allSatisfy({ $0 is String })
            else {
                throw AcceptanceFailure.unknown("resolved product dependency module aliases are invalid")
            }
        }

        try result.append(TargetDependencyDescriptor(
            kind: kind,
            name: name,
            package: package,
            condition: dependencyCondition(kind: kind, parts: parts)
        ))
    }
    return result
}

private func dependencyCondition(kind: String, parts: [Any]) throws -> JSONObject? {
    let index = kind == "product" ? 3 : 1
    guard !(parts[index] is NSNull) else {
        return nil
    }
    guard let condition = parts[index] as? JSONObject else {
        throw AcceptanceFailure.unknown("resolved target dependency condition is not an object")
    }

    return condition
}

private func resolveProductDependency(
    name: String,
    requestedPackage: String?,
    sourcePackage: String,
    packageAliases: [String: String],
    productsByPackage: [String: [String: PackageProduct]],
    queue: inout [ResolvedTarget],
    visitedProducts: inout Set<ResolvedProduct>,
    unresolved: inout Set<String>
) {
    let candidates: [String] =
        if let requestedPackage, !requestedPackage.isEmpty {
            packageAliases[requestedPackage].map { [$0] } ?? []
        }
        else {
            Set(packageAliases.values)
                .filter { identity in
                    productsByPackage[identity]?[name] != nil
                }
                .sorted()
        }
    guard candidates.count == 1,
          let package = candidates.first,
          let product = productsByPackage[package]?[name],
          !product.targets.isEmpty
    else {
        unresolved.insert("unresolved product \(sourcePackage):\(name)")
        return
    }

    visitedProducts.insert(ResolvedProduct(package: package, name: name))
    queue.append(contentsOf: product.targets.map { ResolvedTarget(package: package, name: $0) })
}

// MARK: - Canonical three-route parity records

func paritySchemaReport(_ contract: ParityContract) -> JSONObject {
    [
        "status": "ready",
        "schema_version": contract.schemaVersion,
        "required_routes": contract.requiredRoutes,
        "event_types": contract.eventTypes,
        "terminal_kinds": contract.terminalKinds,
        "finish_reasons": contract.finishReasons,
        "execution_status": "unmet"
    ]
}

func validateParityRecord(_ record: JSONObject, contract: ParityContract) throws -> JSONObject {
    try requireExactKeys(
        record,
        expected: ["schema_version", "case_id", "route", "events", "terminal"],
        context: "parity record"
    )
    guard try record.integer("schema_version") == contract.schemaVersion else {
        throw AcceptanceFailure.unknown("parity record schema_version is unsupported")
    }
    guard try !record.string("case_id").isEmpty else {
        throw AcceptanceFailure.unknown("parity record case_id is empty")
    }

    let route = try record.string("route")
    guard contract.requiredRoutes.contains(route) else {
        throw AcceptanceFailure.unknown("parity record route is unknown: \(route)")
    }

    let events = try record.array("events")
    for (index, value) in events.enumerated() {
        guard let event = value as? JSONObject else {
            throw AcceptanceFailure.unknown("parity event at index \(index) is not an object")
        }
        guard try event.integer("sequence") == index else {
            throw AcceptanceFailure.unknown("parity event sequence is not contiguous at index \(index)")
        }

        let type = try event.string("type")
        guard contract.eventTypes.contains(type) else {
            throw AcceptanceFailure.unknown("parity event type is unknown: \(type)")
        }

        switch type {
        case "text_delta":
            try requireExactKeys(event, expected: ["sequence", "type", "text"], context: "text_delta event")
            _ = try event.string("text")

        case "tool_call":
            try requireExactKeys(
                event,
                expected: ["sequence", "type", "name", "arguments"],
                context: "tool_call event"
            )
            guard try !event.string("name").isEmpty else {
                throw AcceptanceFailure.unknown("tool_call event name is empty")
            }

            _ = try event.object("arguments")

        default:
            throw AcceptanceFailure.unknown("parity event type is not implemented: \(type)")
        }
    }

    let terminal = try record.object("terminal")
    let kind = try terminal.string("kind")
    guard contract.terminalKinds.contains(kind) else {
        throw AcceptanceFailure.unknown("parity terminal kind is unknown: \(kind)")
    }

    switch kind {
    case "response":
        try validateResponseTerminal(terminal, contract: contract)

    case "cancelled":
        try requireExactKeys(terminal, expected: ["kind"], context: "cancelled terminal")

    case "error":
        try requireExactKeys(terminal, expected: ["kind", "code"], context: "error terminal")
        guard try !terminal.string("code").isEmpty else {
            throw AcceptanceFailure.unknown("parity error code is empty")
        }

    default:
        throw AcceptanceFailure.unknown("parity terminal kind is not implemented: \(kind)")
    }
    return record
}

private func validateResponseTerminal(_ terminal: JSONObject, contract: ParityContract) throws {
    try requireExactKeys(
        terminal,
        expected: ["kind", "output", "tool_calls", "finish_reason", "usage"],
        context: "response terminal"
    )
    _ = try terminal.string("output")
    let finishReason = try terminal.string("finish_reason")
    guard contract.finishReasons.contains(finishReason) else {
        throw AcceptanceFailure.unknown("parity finish_reason is unknown: \(finishReason)")
    }

    for (index, value) in try terminal.array("tool_calls").enumerated() {
        guard let call = value as? JSONObject else {
            throw AcceptanceFailure.unknown("parity tool call at index \(index) is not an object")
        }

        try requireExactKeys(call, expected: ["name", "arguments"], context: "terminal tool call")
        guard try !call.string("name").isEmpty else {
            throw AcceptanceFailure.unknown("parity terminal tool call is incomplete")
        }

        _ = try call.object("arguments")
    }
    let usage = try terminal.object("usage")
    try requireExactKeys(
        usage,
        expected: ["prompt_tokens", "completion_tokens", "total_tokens"],
        context: "parity usage"
    )
    let prompt = try usage.integer("prompt_tokens")
    let completion = try usage.integer("completion_tokens")
    let total = try usage.integer("total_tokens")
    guard prompt >= 0, completion >= 0, total == prompt + completion else {
        throw AcceptanceFailure.unknown("parity usage token counts are invalid")
    }
}

private func requireExactKeys(_ object: JSONObject, expected: Set<String>, context: String) throws {
    let actual = Set(object.keys)
    guard actual == expected else {
        let missing = expected.subtracting(actual).sorted()
        let unknown = actual.subtracting(expected).sorted()
        throw AcceptanceFailure.unknown(
            "\(context) keys are invalid; missing=\(missing), unknown=\(unknown)"
        )
    }
}

private func relativePathForGuard(_ file: URL, to root: URL) -> String {
    file.resolvingSymlinksInPath().path.replacingOccurrences(
        of: root.resolvingSymlinksInPath().path + "/",
        with: ""
    )
}

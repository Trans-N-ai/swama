import Foundation

// MARK: - Compiler-derived public API boundary

func compilerPublicAPIReport(
    target: String,
    paths: WorkspacePaths,
    developerDirectory: URL,
    contract: CoreGuardContract
) throws -> JSONObject {
    let scratch = paths.repository.appendingPathComponent(".build/swama-core-symbol-graph")
    if FileManager.default.fileExists(atPath: scratch.path) {
        for file in try regularFiles(in: scratch, extensions: ["json"])
            where file.lastPathComponent.hasSuffix(".symbols.json")
        {
            try FileManager.default.removeItem(at: file)
        }
    }

    var environment = try developerEnvironment(developerDirectory)
    environment["SWIFTPM_MODULECACHE_OVERRIDE"] = scratch.appendingPathComponent("module-cache").path
    let result = try runCommand(
        [
            "xcrun",
            "swift",
            "package",
            "--package-path",
            paths.package.path,
            "--scratch-path",
            scratch.path,
            "--force-resolved-versions",
            "dump-symbol-graph",
            "--minimum-access-level",
            "public",
            "--skip-synthesized-members"
        ],
        currentDirectory: paths.repository,
        environment: environment,
        timeout: contract.compilerTimeoutSeconds,
        sampleMemory: false,
        timeoutFailureKind: .unknown,
        timeoutContext: "SwamaCore symbol graph"
    )
    guard result.returnCode == 0 else {
        throw AcceptanceFailure.failed(
            "SwamaCore symbol graph build failed:\n\(commandFailureSummary(result))"
        )
    }

    var graphs: [JSONObject] = []
    var graphFiles: [JSONObject] = []
    for file in try regularFiles(in: scratch, extensions: ["json"])
        .filter({ $0.lastPathComponent.hasSuffix(".symbols.json") })
        .sorted(by: { $0.path < $1.path })
    {
        let graph = try loadJSONObject(file)
        guard (try? graph.object("module").string("name")) == target else {
            continue
        }

        graphs.append(graph)
        try graphFiles.append([
            "file": relativePathForGuard(file, to: scratch),
            "sha256": sha256File(file)
        ])
    }
    guard !graphs.isEmpty else {
        throw AcceptanceFailure.unknown("compiler produced no symbol graph for target: \(target)")
    }

    var report = try analyzePublicAPISymbolGraphs(
        graphs,
        target: target,
        allowedModules: Set(contract.allowedPublicModules)
    )
    let reachabilityHits = try publicReachabilityAttributeHits(
        in: paths.package.appendingPathComponent("Sources/\(target)"),
        repository: paths.repository
    )
    report["status"] = "ready"
    report["symbol_graph_files"] = graphFiles
    report["reachability_attribute_hits"] = reachabilityHits
    report["passed"] = report["passed"] as? Bool == true && reachabilityHits.isEmpty
    report["duration_ms"] = result.durationMilliseconds
    return report
}

func analyzePublicAPISymbolGraphs(
    _ graphs: [JSONObject],
    target: String,
    allowedModules: Set<String>
) throws -> JSONObject {
    var canonicalSymbols: [JSONObject] = []
    var violations: [JSONObject] = []

    for graph in graphs {
        guard try graph.object("module").string("name") == target else {
            continue
        }

        let symbols = try graph.object("symbols")
        let relationships = (try? graph.array("relationships").compactMap { $0 as? JSONObject }) ?? []
        let relationshipsBySource = Dictionary(grouping: relationships) { $0["source"] as? String ?? "" }

        for symbol in symbols.values.compactMap({ $0 as? JSONObject }) {
            let access = (symbol["accessLevel"] as? String) ?? "public"
            guard access == "public" || access == "open" else {
                continue
            }

            let identifier = try symbol.object("identifier").string("precise")
            let kind = try symbol.object("kind").string("identifier")
            let path = try symbol.array("pathComponents").compactMap { $0 as? String }
            let fragments = try symbol.array("declarationFragments").compactMap { $0 as? JSONObject }
            let declaration = fragments.compactMap { $0["spelling"] as? String }.joined()
            let canonicalFragments = fragments.map { fragment -> JSONObject in
                var value: JSONObject = [
                    "kind": fragment["kind"] as? String ?? "unknown",
                    "spelling": fragment["spelling"] as? String ?? ""
                ]
                if let precise = fragment["preciseIdentifier"] as? String {
                    value["precise_identifier"] = precise
                    value["module"] = moduleName(in: precise) ?? "unknown"
                }
                return value
            }
            var references: Set<String> = []

            for fragment in fragments where fragment["kind"] as? String == "typeIdentifier" {
                guard let precise = fragment["preciseIdentifier"] as? String,
                      let module = moduleName(in: precise)
                else {
                    continue
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
            for relationship in relationshipsBySource[identifier] ?? [] {
                guard let relationshipKind = relationship["kind"] as? String,
                      ["conformsTo", "inheritsFrom", "requirementOf"].contains(relationshipKind),
                      let targetIdentifier = relationship["target"] as? String
                else {
                    continue
                }

                let fallback = relationship["targetFallback"] as? String
                let module = moduleName(in: targetIdentifier) ?? fallback?.split(separator: ".").first.map(String.init)
                if let module {
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

private func moduleName(in preciseIdentifier: String) -> String? {
    guard preciseIdentifier.hasPrefix("s:") else {
        return preciseIdentifier.contains(":") ? "__foreign__" : nil
    }

    let payload = preciseIdentifier.dropFirst(2)
    var digits = ""
    for character in payload {
        guard character.isNumber else {
            break
        }

        digits.append(character)
    }
    guard let length = Int(digits), length > 0 else {
        return "Swift"
    }

    let moduleStart = payload.index(payload.startIndex, offsetBy: digits.count)
    guard let moduleEnd = payload.index(moduleStart, offsetBy: length, limitedBy: payload.endIndex) else {
        return nil
    }

    return String(payload[moduleStart ..< moduleEnd])
}

// MARK: - External consumer and target dependency boundary

func externalConsumerBoundaryReport(
    fixture: URL,
    contract: CoreGuardContract
) throws -> JSONObject {
    let manifestURL = fixture.appendingPathComponent("Package.swift")
    let manifest = try String(contentsOf: manifestURL, encoding: .utf8)
    let products = try captures(
        #"\.product\s*\(\s*name:\s*\"([^\"]+)\""#,
        in: manifest
    )
    let remotePackages = try captures(#"\.package\s*\(\s*url:\s*\"([^\"]+)\""#, in: manifest)
    let expectedProducts = [contract.fixtureProduct]
    let unexpectedProducts = Array(Set(products).subtracting(expectedProducts)).sorted()
    let missingProducts = Array(Set(expectedProducts).subtracting(products)).sorted()

    let allowedImports = Set(contract.fixtureAllowedImports)
    let expression = try NSRegularExpression(pattern: swiftImportDeclarationPattern)
    var imports: Set<String> = []
    let sourceRoot = fixture.appendingPathComponent("Sources")
    if FileManager.default.fileExists(atPath: sourceRoot.path) {
        for file in try regularFiles(in: sourceRoot, extensions: ["swift"]) {
            let lines = try String(contentsOf: file, encoding: .utf8)
                .split(separator: "\n", omittingEmptySubsequences: false)
            for line in lines {
                if let module = swiftImportedModule(in: String(line), matching: expression) {
                    imports.insert(module)
                }
            }
        }
    }
    let unexpectedImports = Array(imports.subtracting(allowedImports)).sorted()
    let missingImports = allowedImports.contains(contract.fixtureProduct) && !imports.contains(contract.fixtureProduct)
        ? [contract.fixtureProduct]
        : []
    let passed = remotePackages.isEmpty
        && unexpectedProducts.isEmpty
        && missingProducts.isEmpty
        && unexpectedImports.isEmpty
        && missingImports.isEmpty

    return [
        "status": passed ? "ready" : "unmet",
        "expected_products": expectedProducts,
        "actual_products": products.sorted(),
        "unexpected_products": unexpectedProducts,
        "missing_products": missingProducts,
        "remote_packages": remotePackages.sorted(),
        "actual_imports": imports.sorted(),
        "unexpected_imports": unexpectedImports,
        "missing_imports": missingImports,
        "passed": passed
    ]
}

func coreTargetDependencyReport(
    target: String,
    paths: WorkspacePaths,
    developerDirectory: URL,
    contract: CoreGuardContract
) throws -> JSONObject {
    let result = try runCommand(
        [
            "xcrun",
            "swift",
            "package",
            "--package-path",
            paths.package.path,
            "describe",
            "--type",
            "json"
        ],
        currentDirectory: paths.repository,
        environment: developerEnvironment(developerDirectory),
        timeout: 60,
        sampleMemory: false,
        timeoutFailureKind: .unknown,
        timeoutContext: "SwamaCore dependency graph"
    )
    guard result.returnCode == 0,
          let data = result.stdout.data(using: .utf8),
          let description = try JSONSerialization.jsonObject(with: data) as? JSONObject
    else {
        throw AcceptanceFailure.unknown(
            "cannot inspect SwamaCore dependency graph:\n\(commandFailureSummary(result))"
        )
    }

    return try analyzeTargetDependencyGraph(
        description,
        target: target,
        forbiddenProducts: Set(contract.forbiddenTransitiveProducts)
    )
}

func analyzeTargetDependencyGraph(
    _ description: JSONObject,
    target: String,
    forbiddenProducts: Set<String>
) throws -> JSONObject {
    let targetObjects = try description.array("targets").compactMap { $0 as? JSONObject }
    let targetsByName = Dictionary(uniqueKeysWithValues: targetObjects.compactMap { item -> (String, JSONObject)? in
        guard let name = item["name"] as? String else {
            return nil
        }

        return (name, item)
    })
    guard targetsByName[target] != nil else {
        return [
            "status": "unmet",
            "target": target,
            "target_dependencies": [],
            "product_dependencies": [],
            "forbidden_products": [],
            "passed": false
        ]
    }

    var queue = [target]
    var visited: Set<String> = []
    var products: Set<String> = []
    while let name = queue.popLast() {
        guard visited.insert(name).inserted, let item = targetsByName[name] else {
            continue
        }

        let targetDependencies = (item["target_dependencies"] as? [String]) ?? []
        let productDependencies = (item["product_dependencies"] as? [String]) ?? []
        queue.append(contentsOf: targetDependencies)
        products.formUnion(productDependencies)
    }
    let forbidden = products.intersection(forbiddenProducts).sorted()
    return [
        "status": forbidden.isEmpty ? "ready" : "violated",
        "target": target,
        "target_dependencies": visited.sorted(),
        "product_dependencies": products.sorted(),
        "forbidden_products": forbidden,
        "passed": forbidden.isEmpty
    ]
}

private func captures(_ pattern: String, in text: String) throws -> [String] {
    let expression = try NSRegularExpression(pattern: pattern)
    let range = NSRange(text.startIndex ..< text.endIndex, in: text)
    return expression.matches(in: text, range: range).compactMap { match in
        guard let capture = Range(match.range(at: 1), in: text) else {
            return nil
        }

        return String(text[capture])
    }
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
            guard event["arguments"] != nil else {
                throw AcceptanceFailure.unknown("tool_call event arguments are missing")
            }

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
        guard try !call.string("name").isEmpty, call["arguments"] != nil else {
            throw AcceptanceFailure.unknown("parity terminal tool call is incomplete")
        }
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

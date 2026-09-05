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
        symbolGraphCommand(package: paths.package, scratch: scratch),
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

func symbolGraphCommand(package: URL, scratch: URL) -> [String] {
    [
        "xcrun",
        "swift",
        "package",
        "--package-path",
        package.path,
        "--scratch-path",
        scratch.path,
        "--force-resolved-versions",
        "dump-symbol-graph",
        "--minimum-access-level",
        "public",
        "--skip-synthesized-members",
        "--emit-extension-block-symbols"
    ]
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

        let symbols = try strictObjectArray(graph, key: "symbols", context: "symbol graph")
        let relationships = try strictObjectArray(graph, key: "relationships", context: "symbol graph")
        var relationshipsBySource: [String: [JSONObject]] = [:]
        for relationship in relationships {
            let source = try relationship.string("source")
            _ = try relationship.string("kind")
            _ = try relationship.string("target")
            if relationship["targetFallback"] != nil {
                _ = try relationship.string("targetFallback")
            }
            relationshipsBySource[source, default: []].append(relationship)
        }

        for symbol in symbols {
            let access = try symbol.string("accessLevel")
            guard access == "public" || access == "open" else {
                continue
            }

            let identifier = try symbol.object("identifier").string("precise")
            let kind = try symbol.object("kind").string("identifier")
            let path = try strictStringArray(symbol, key: "pathComponents", context: "public symbol")
            let fragments = try strictObjectArray(
                symbol,
                key: "declarationFragments",
                context: "public symbol"
            )
            let declaration = fragments.compactMap { $0["spelling"] as? String }.joined()
            let canonicalFragments = try fragments.map { fragment -> JSONObject in
                var value: JSONObject = try [
                    "kind": fragment.string("kind"),
                    "spelling": fragment.string("spelling")
                ]
                if fragment["preciseIdentifier"] != nil {
                    let precise = try fragment.string("preciseIdentifier")
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
                let relationshipKind = try relationship.string("kind")
                guard ["conformsTo", "extensionTo", "inheritsFrom", "memberOf", "requirementOf"]
                    .contains(relationshipKind)
                else {
                    continue
                }

                let targetIdentifier = try relationship.string("target")
                let fallback = try relationship["targetFallback"] == nil
                    ? nil
                    : relationship.string("targetFallback")
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

        let aliases = aliasesByPackage[node.package] ?? [:]
        let traits = activeTraits?[node.package] ?? ["default"]

        for dependency in try targetDependencyDescriptors(targetDescription) {
            guard dependency.isActive(platform: "macos", traits: traits) else {
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

    func isActive(platform: String, traits: Set<String>) -> Bool {
        guard let condition else {
            return true
        }

        let platforms = Set((condition["platformNames"] as? [String]) ?? [])
        let requiredTraits = Set((condition["traits"] as? [String]) ?? [])
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
        let dependencies = try node.array("dependencies").compactMap { $0 as? JSONObject }
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
        let descriptor = ResolvedPackageDescriptor(
            path: path,
            directPackageAliases: aliases,
            activeTraits: Set((node["traits"] as? [String]) ?? [])
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
    for value in try manifest.array("products") {
        guard let product = value as? JSONObject else {
            throw AcceptanceFailure.unknown("resolved package product is not an object")
        }

        let name = try product.string("name")
        guard result[name] == nil else {
            throw AcceptanceFailure.unknown("resolved package has duplicate product: \(name)")
        }

        result[name] = try PackageProduct(
            targets: product.array("targets").compactMap { $0 as? String },
            isLibrary: (try? product.object("type").array("library")) != nil
        )
    }
    return result
}

private func packageTargets(_ manifest: JSONObject) throws -> [String: JSONObject] {
    var result: [String: JSONObject] = [:]
    for value in try manifest.array("targets") {
        guard let target = value as? JSONObject else {
            throw AcceptanceFailure.unknown("resolved package target is not an object")
        }

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
    for value in try manifest.array("dependencies") {
        guard let dependency = value as? JSONObject else {
            throw AcceptanceFailure.unknown("resolved package dependency is not an object")
        }

        for descriptors in dependency.values {
            guard let descriptors = descriptors as? [Any] else {
                throw AcceptanceFailure.unknown("resolved package dependency payload is not an array")
            }

            for value in descriptors {
                guard let descriptor = value as? JSONObject else {
                    throw AcceptanceFailure.unknown("resolved package dependency descriptor is not an object")
                }

                let identity = try descriptor.string("identity")
                let aliases = [identity, descriptor["nameForTargetDependencyResolutionOnly"] as? String]
                    .compactMap(\.self)
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
    for value in try target.array("dependencies") {
        guard let dependency = value as? JSONObject, dependency.count == 1,
              let kind = dependency.keys.first,
              let parts = dependency[kind] as? [Any],
              let name = parts.first as? String
        else {
            throw AcceptanceFailure.unknown("resolved target dependency has an unsupported shape")
        }

        result.append(TargetDependencyDescriptor(
            kind: kind,
            name: name,
            package: kind == "product" && parts.count > 1 ? parts[1] as? String : nil,
            condition: dependencyCondition(kind: kind, parts: parts)
        ))
    }
    return result
}

private func dependencyCondition(kind: String, parts: [Any]) -> JSONObject? {
    let index = kind == "product" ? 3 : 1
    guard parts.indices.contains(index) else {
        return nil
    }

    return parts[index] as? JSONObject
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

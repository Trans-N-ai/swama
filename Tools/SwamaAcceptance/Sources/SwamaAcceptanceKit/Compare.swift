import Foundation

func compareReports(
    baseline: JSONObject,
    candidate: JSONObject,
    contract: AcceptanceContract,
    contractURL: URL,
    paths: WorkspacePaths
) throws -> JSONObject {
    try verifyReport(baseline)
    try verifyReport(candidate)
    let currentContractHash = try sha256File(contractURL)
    guard baseline["contract_sha256"] as? String == currentContractHash else {
        throw AcceptanceFailure.unknown("baseline contract hash does not match current --contract")
    }
    guard candidate["contract_sha256"] as? String == currentContractHash else {
        throw AcceptanceFailure.unknown("candidate contract hash does not match current --contract")
    }

    let currentInstrument = try currentInstrumentIdentity(paths: paths)
    let baselineInstrument = try baseline.object("instrument")
    let candidateInstrument = try candidate.object("instrument")
    let immutableInstrumentKeys = [
        "harness_binary_sha256",
        "harness_sources_sha256",
        "harness_tests_sha256",
        "core_probe_sha256",
        "fixture_manifest_sha256"
    ]
    for key in immutableInstrumentKeys {
        let current = try currentInstrument.string(key)
        guard baselineInstrument[key] as? String == current else {
            throw AcceptanceFailure.unknown("baseline instrument hash does not match current \(key)")
        }
        guard candidateInstrument[key] as? String == current else {
            throw AcceptanceFailure.unknown("candidate instrument hash does not match current \(key)")
        }
    }

    let baselineProvenance = try baseline.object("provenance")
    let candidateProvenance = try candidate.object("provenance")
    var comparable: JSONObject = try [
        "platform": jsonEqual(baselineProvenance["platform"], candidateProvenance["platform"]),
        "toolchain": jsonEqual(baselineProvenance["toolchain"], candidateProvenance["toolchain"]),
        "models": jsonEqual(baselineProvenance["models"], candidateProvenance["models"]),
        "contract": baseline["contract_sha256"] as? String == candidate["contract_sha256"] as? String,
        "instrument": Dictionary(uniqueKeysWithValues: immutableInstrumentKeys.map { key in
            (key, baselineInstrument[key] as? String == candidateInstrument[key] as? String)
        })
    ]

    let baselineMetal = try baseline.object("build").object("metal_build")
    let candidateMetal = try candidate.object("build").object("metal_build")
    let metalKeys = [
        "metal_version",
        "metallib_version",
        "metal_executable_sha256",
        "metallib_executable_sha256",
        "mlx_swift_revision",
        "sha256"
    ]
    var metalComparable: JSONObject = [:]
    for key in metalKeys {
        metalComparable[key] = try baselineMetal.string(key) == candidateMetal.string(key)
    }
    metalComparable["source_inputs"] = try metalSourceIdentities(baselineMetal) == metalSourceIdentities(candidateMetal)
        && !metalSourceIdentities(baselineMetal).isEmpty
    comparable["metal_toolchain"] = metalComparable

    guard try allJSONBooleansTrue(comparable) else {
        throw try AcceptanceFailure
            .unknown("reports are not comparable: \(String(decoding: compactJSONData(comparable), as: UTF8.self))")
    }

    var findings: [JSONObject] = []
    let baselineBenchmarks = try baseline.object("benchmarks")
    let candidateBenchmarks = try candidate.object("benchmarks")
    guard !baselineBenchmarks.isEmpty else {
        throw AcceptanceFailure.unknown("baseline contains no benchmark models")
    }

    let benchmarkRoutes = ["core", "http"]
    var comparedRouteCount = 0
    for model in baselineBenchmarks.keys.sorted() {
        guard let baselineRoutes = baselineBenchmarks[model] as? JSONObject,
              let candidateRoutes = candidateBenchmarks[model] as? JSONObject
        else {
            throw AcceptanceFailure.unknown("candidate is missing benchmark model: \(model)")
        }

        for route in benchmarkRoutes {
            let baselineSummary = try baselineRoutes.object(route).object("summary")
            let candidateSummary = try candidateRoutes.object(route).object("summary")
            let baseTTFT = try baselineSummary.double("ttft_median_ms")
            let candidateTTFT = try candidateSummary.double("ttft_median_ms")
            let ttftLimit = max(
                baseTTFT * contract.benchmark.comparison.ttftRatioMax,
                baseTTFT + contract.benchmark.comparison.ttftAbsoluteSlackMilliseconds
            )
            let baseSpeed = try baselineSummary.double("tokens_per_second_median")
            let candidateSpeed = try candidateSummary.double("tokens_per_second_median")
            let speedFloor = baseSpeed * contract.benchmark.comparison.tokensPerSecondRatioMin
            let baseMemory = try baselineSummary.double("peak_resident_bytes")
            let candidateMemory = try candidateSummary.double("peak_resident_bytes")
            let memoryLimit = max(
                baseMemory * contract.benchmark.comparison.peakResidentRatioMax,
                baseMemory + Double(contract.benchmark.comparison.peakResidentAbsoluteSlackBytes)
            )
            if candidateTTFT > ttftLimit {
                findings.append(metricFinding(model, route, "ttft", baseTTFT, candidateTTFT, ttftLimit))
            }
            if candidateSpeed < speedFloor {
                findings.append(metricFinding(model, route, "tokens_per_second", baseSpeed, candidateSpeed, speedFloor))
            }
            if candidateMemory > memoryLimit {
                findings.append(metricFinding(
                    model,
                    route,
                    "peak_resident_bytes",
                    baseMemory,
                    candidateMemory,
                    memoryLimit
                ))
            }
            comparedRouteCount += 1
        }
    }
    if candidate["passed"] as? Bool != true {
        findings.append([
            "metric": "candidate_contract",
            "candidate": "failed reliability or architecture gate"
        ])
    }
    return [
        "comparable": comparable,
        "comparison_coverage": [
            "benchmark_models": baselineBenchmarks.count,
            "benchmark_routes": comparedRouteCount
        ],
        "findings": findings,
        "passed": findings.isEmpty
    ]
}

private func metricFinding(
    _ model: String,
    _ route: String,
    _ metric: String,
    _ baseline: Double,
    _ candidate: Double,
    _ limit: Double
) -> JSONObject {
    [
        "model": model,
        "route": route,
        "metric": metric,
        "baseline": baseline,
        "candidate": candidate,
        "limit": limit
    ]
}

private func metalSourceIdentities(_ metal: JSONObject) throws -> [String] {
    try metal.array("sources").compactMap { value in
        guard let item = value as? JSONObject,
              let path = item["path"] as? String,
              let hash = item["sha256"] as? String
        else {
            return nil
        }

        return "\(path):\(hash)"
    }
}

private func jsonEqual(_ lhs: Any?, _ rhs: Any?) throws -> Bool {
    guard let lhs, let rhs else {
        return lhs == nil && rhs == nil
    }

    return try compactJSONData([lhs]) == compactJSONData([rhs])
}

private func allJSONBooleansTrue(_ object: JSONObject) throws -> Bool {
    for value in object.values {
        if let boolean = value as? Bool {
            if !boolean {
                return false
            }
        }
        else if let nested = value as? JSONObject {
            if try !allJSONBooleansTrue(nested) {
                return false
            }
        }
        else {
            throw AcceptanceFailure.unknown("comparability report contains a non-boolean leaf")
        }
    }
    return true
}

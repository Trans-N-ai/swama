import Foundation

func runBaseline(
    contract: AcceptanceContract,
    contractURL: URL,
    developerDirectory: URL,
    metalDeveloperDirectory: URL,
    architectureStage: ArchitectureStage,
    paths: WorkspacePaths
) async throws -> JSONObject {
    var environment = try developerEnvironment(developerDirectory)
    environment["SWAMA_PROMPT_CACHE"] = "0"
    let architecture = try architectureReport(
        contract: contract.architecture,
        stage: architectureStage,
        paths: paths
    )
    guard architecture["passed"] as? Bool == true else {
        throw AcceptanceFailure.failed("architecture gate failed")
    }

    let build = try buildProducts(
        developerDirectory: developerDirectory,
        metalDeveloperDirectory: metalDeveloperDirectory,
        paths: paths
    )
    let timeout = contract.reliability.timeoutSeconds
    var benchmarks: JSONObject = [:]
    var surfaces: JSONObject = [:]

    for model in contract.models where model.benchmark {
        var core = try coreBenchmark(
            probe: build.probe,
            model: model.id,
            prompt: contract.benchmark.prompt,
            maxTokens: contract.benchmark.maxTokens,
            rounds: contract.benchmark.rounds,
            environment: environment,
            timeout: timeout,
            paths: paths
        )
        let coreRecords = try core.array("records").compactMap { $0 as? JSONObject }
        core["summary"] = try summarizeCore(
            coreRecords,
            warmups: contract.benchmark.warmupRounds,
            minimumMeasured: contract.benchmark.minimumMeasuredRounds
        )
        surfaces[model.id] = try [
            "cli": cliContract(
                swama: build.swama,
                model: model.id,
                prompt: contract.benchmark.prompt,
                maxTokens: 32,
                environment: environment,
                timeout: timeout,
                paths: paths
            )
        ]

        let serverRun = try await withServer(
            executable: build.swama,
            environment: environment,
            paths: paths
        ) { port in
            var records: [JSONObject] = []
            for _ in 0 ..< contract.benchmark.rounds {
                try await records.append(httpStream(
                    port: port,
                    model: model.id,
                    prompt: contract.benchmark.prompt,
                    maxTokens: contract.benchmark.maxTokens
                ))
            }
            return records
        }
        let httpRecords = serverRun.value.map { record -> JSONObject in
            var value = record
            value["external_peak_resident_bytes"] = serverRun.peakResidentBytes
            return value
        }
        let http: JSONObject = try [
            "records": httpRecords,
            "summary": summarizeHTTP(
                httpRecords,
                warmups: contract.benchmark.warmupRounds,
                minimumMeasured: contract.benchmark.minimumMeasuredRounds
            )
        ]
        benchmarks[model.id] = ["core": core, "http": http]
    }

    guard let reliabilityModel = contract.models.first(where: { $0.role == "reliability" })?.id else {
        throw AcceptanceFailure.unknown("contract has no reliability model")
    }

    var reliability: JSONObject = [:]

    let cancelResult = try runCommand(
        [
            build.probe.path, "cancel", reliabilityModel,
            String(contract.reliability.cancelMaxTokens),
            "Write a long numbered list so generation continues until cancelled."
        ],
        currentDirectory: paths.fixture,
        environment: environment,
        timeout: timeout
    )
    guard cancelResult.returnCode == 0 else {
        throw AcceptanceFailure.failed("cancellation contract failed: \(cancelResult.stderr.suffix(2000))")
    }

    reliability["cancel_then_recover"] = try jsonObjectFromProcess(cancelResult)

    let concurrentResult = try runCommand(
        [
            build.probe.path, "concurrent", reliabilityModel,
            String(contract.reliability.concurrentMaxTokens),
            String(contract.reliability.concurrentRequests),
            "Return a short deterministic sentence."
        ],
        currentDirectory: paths.fixture,
        environment: environment,
        timeout: timeout
    )
    guard concurrentResult.returnCode == 0 else {
        throw AcceptanceFailure.failed("concurrency contract failed: \(concurrentResult.stderr.suffix(2000))")
    }

    reliability["same_model_concurrency"] = try jsonObjectFromProcess(concurrentResult)

    let repeatResult = try coreBenchmark(
        probe: build.probe,
        model: reliabilityModel,
        prompt: "Return the marker REPEAT_OK and one integer.",
        maxTokens: contract.reliability.repeatMaxTokens,
        rounds: contract.reliability.repeatRounds,
        environment: environment,
        timeout: timeout,
        paths: paths
    )
    let repeatRecords = try repeatResult.array("records").compactMap { $0 as? JSONObject }
    reliability["repeat_generation"] = [
        "completed_rounds": repeatRecords.count,
        "outputs_nonempty": repeatRecords.allSatisfy { ($0["output"] as? String)?.isEmpty == false }
    ]

    let switchModels = contract.models.map(\.id)
    let switchResult = try runCommand(
        [
            build.probe.path, "switch", switchModels.joined(separator: ","),
            String(contract.reliability.switchMaxTokens),
            String(contract.reliability.switchCycles),
            String(contract.reliability.releaseSettleMilliseconds),
            "Return a short marker for the model-switch test."
        ],
        currentDirectory: paths.fixture,
        environment: environment,
        timeout: timeout * 2
    )
    guard switchResult.returnCode == 0 else {
        throw AcceptanceFailure.failed("model-switch contract failed: \(switchResult.stderr.suffix(2000))")
    }

    var switchRecord = try jsonObjectFromProcess(switchResult)
    let rssValues = try switchRecord.array("residentBytesAfterClear").compactMap { ($0 as? NSNumber)?.uint64Value }
    let expectedSwitchRuns = switchModels.count * contract.reliability.switchCycles
    guard rssValues.count == expectedSwitchRuns, let firstSmallRSS = rssValues.first,
          let finalRSS = rssValues.last
    else {
        throw AcceptanceFailure.unknown("model-switch probe omitted post-clear RSS readings")
    }

    let releaseLimit = max(
        Double(firstSmallRSS) * contract.reliability.releaseRSSRatioMax,
        Double(firstSmallRSS + contract.reliability.releaseRSSAbsoluteSlackBytes)
    )
    switchRecord["release_limit_bytes"] = releaseLimit
    switchRecord["release_gate_passed"] = Double(finalRSS) <= releaseLimit
    reliability["model_switch_and_release"] = switchRecord

    let disconnectRun = try await withServer(
        executable: build.swama,
        environment: environment,
        paths: paths
    ) { port -> JSONObject in
        let disconnected = try await httpStream(
            port: port,
            model: reliabilityModel,
            prompt: "Write a long essay that will be interrupted.",
            maxTokens: 512,
            disconnectAfterFirst: true
        )
        try await Task.sleep(for: .milliseconds(500))
        let followup = try await httpStream(
            port: port,
            model: reliabilityModel,
            prompt: "Reply with RECOVERED.",
            maxTokens: 24
        )
        return ["disconnect": disconnected, "followup": followup]
    }
    var disconnect = disconnectRun.value
    disconnect["server_peak_resident_bytes"] = disconnectRun.peakResidentBytes
    reliability["http_disconnect_then_recover"] = disconnect

    let reliabilityGates = try reliabilityGateStatus(reliability, contract: contract.reliability)
    return try [
        "schema_version": contract.schemaVersion,
        "contract_sha256": sha256File(contractURL),
        "instrument": currentInstrumentIdentity(paths: paths),
        "provenance": captureProvenance(
            developerDirectory: developerDirectory,
            models: contract.models.map(\.id),
            paths: paths
        ),
        "architecture": architecture,
        "build": build.report,
        "benchmarks": benchmarks,
        "surface_contracts": surfaces,
        "reliability": reliability,
        "reliability_gates": reliabilityGates,
        "passed": reliabilityGates.values.allSatisfy(\.self)
    ]
}

private func coreBenchmark(
    probe: URL,
    model: String,
    prompt: String,
    maxTokens: Int,
    rounds: Int,
    environment: [String: String],
    timeout: Double,
    paths: WorkspacePaths
) throws -> JSONObject {
    let result = try runCommand(
        [probe.path, "generate", model, String(maxTokens), String(rounds), prompt],
        currentDirectory: paths.fixture,
        environment: environment,
        timeout: timeout
    )
    guard result.returnCode == 0 else {
        throw AcceptanceFailure.failed("core benchmark failed for \(model): \(result.stderr.suffix(2000))")
    }
    guard var records = try lastJSONValue(result.stdout) as? [JSONObject], records.count == rounds else {
        throw AcceptanceFailure.unknown("core benchmark returned incomplete records for \(model)")
    }

    for index in records.indices {
        records[index]["external_peak_resident_bytes"] = result.peakResidentBytes
    }
    return ["records": records, "stderr": result.stderr]
}

private func cliContract(
    swama: URL,
    model: String,
    prompt: String,
    maxTokens: Int,
    environment: [String: String],
    timeout: Double,
    paths: WorkspacePaths
) throws -> JSONObject {
    let result = try runCommand(
        [
            swama.path, "run", model, prompt, "--direct", "--no-stream",
            "--temperature", "0", "--max-tokens", String(maxTokens)
        ],
        currentDirectory: paths.package,
        environment: environment,
        timeout: timeout
    )
    guard result.returnCode == 0, result.stdout.contains("Generation completed") else {
        throw AcceptanceFailure.failed("CLI contract failed for \(model)")
    }

    return [
        "returncode": result.returnCode,
        "duration_ms": result.durationMilliseconds,
        "peak_resident_bytes": result.peakResidentBytes,
        "output_sha256": sha256(Data(result.stdout.utf8)),
        "completed": true
    ]
}

private func summarizeCore(
    _ records: [JSONObject],
    warmups: Int,
    minimumMeasured: Int
) throws -> JSONObject {
    let measured = Array(records.dropFirst(warmups))
    guard measured.count >= minimumMeasured else {
        throw AcceptanceFailure.unknown("core benchmark has too few measured rounds")
    }

    let ttft = try measured.map { try $0.double("ttftMilliseconds") }
    let speed = try measured.map { try $0.double("tokensPerSecond") }
    let total = try measured.map { try $0.double("totalMilliseconds") }
    let peak = measured.map {
        max(
            ($0["peakResidentBytes"] as? NSNumber)?.uint64Value ?? 0,
            ($0["external_peak_resident_bytes"] as? NSNumber)?.uint64Value ?? 0
        )
    }
    .max() ?? 0
    return [
        "rounds": measured.count,
        "ttft_median_ms": median(ttft),
        "tokens_per_second_median": median(speed),
        "total_median_ms": median(total),
        "peak_resident_bytes": peak
    ]
}

private func summarizeHTTP(
    _ records: [JSONObject],
    warmups: Int,
    minimumMeasured: Int
) throws -> JSONObject {
    let measured = Array(records.dropFirst(warmups))
    guard measured.count >= minimumMeasured else {
        throw AcceptanceFailure.unknown("HTTP benchmark has too few measured rounds")
    }

    let speed = try measured.map { try $0.object("usage").double("response_token/s") }
    return try [
        "rounds": measured.count,
        "ttft_median_ms": median(measured.map { try $0.double("ttft_ms") }),
        "tokens_per_second_median": median(speed),
        "total_median_ms": median(measured.map { try $0.double("total_ms") }),
        "peak_resident_bytes": measured.compactMap {
            ($0["external_peak_resident_bytes"] as? NSNumber)?.uint64Value
        }
        .max() ?? 0
    ]
}

func reliabilityGateStatus(
    _ report: JSONObject,
    contract: ReliabilityContract
) throws -> [String: Bool] {
    let cancel = try report.object("cancel_then_recover")
    let concurrent = try report.object("same_model_concurrency")
    let repeatGeneration = try report.object("repeat_generation")
    let switchReport = try report.object("model_switch_and_release")
    let disconnect = try report.object("http_disconnect_then_recover")
    let followup = try disconnect.object("followup")
    return try [
        "cancel_then_recover": cancel.integer("observedTokensBeforeCancel") <= contract.cancelObservedTokenCeiling
            && cancel.double("cancellationMilliseconds") <= contract.cancellationLatencyMillisecondsMax
            && !cancel.string("followupOutput").isEmpty,
        "same_model_concurrency": concurrent.array("outputs").count == contract.concurrentRequests,
        "repeat_generation": repeatGeneration.boolean("outputs_nonempty"),
        "model_switch_and_release": switchReport.boolean("release_gate_passed"),
        "http_disconnect_then_recover": !followup.string("output").isEmpty
            && followup.double("ttft_ms") <= contract.disconnectRecoveryTTFTMillisecondsMax
    ]
}

private func jsonObjectFromProcess(_ result: CommandResult) throws -> JSONObject {
    guard let object = try lastJSONValue(result.stdout) as? JSONObject else {
        throw AcceptanceFailure.unknown("probe did not return a JSON object")
    }

    return object
}

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    guard !sorted.isEmpty else {
        return .nan
    }

    let middle = sorted.count / 2
    return sorted.count.isMultiple(of: 2)
        ? (sorted[middle - 1] + sorted[middle]) / 2
        : sorted[middle]
}

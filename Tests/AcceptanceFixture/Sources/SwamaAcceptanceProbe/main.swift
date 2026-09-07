import Foundation
import SwamaCore

// MARK: - ProbeError

private enum ProbeError: Error, LocalizedError {
    case invalidArguments(String)
    case cancellationDidNotStart
    case cancellationReturnedResponse

    var errorDescription: String? {
        switch self {
        case let .invalidArguments(message):
            message
        case .cancellationDidNotStart:
            "generation completed before the cancellation witness observed two tokens"
        case .cancellationReturnedResponse:
            "cancelled generation returned a response instead of throwing CancellationError"
        }
    }
}

// MARK: - RunRecord

private struct RunRecord: Codable, Sendable {
    let route: String
    let model: String
    let round: Int
    let promptTokens: Int
    let generatedTokens: Int
    let ttftMilliseconds: Double?
    let totalMilliseconds: Double
    let tokensPerSecond: Double?
    let output: String
    let peakResidentBytes: UInt64
}

// MARK: - CancelRecord

private struct CancelRecord: Codable, Sendable {
    let route: String
    let model: String
    let observedTokensBeforeCancel: Int
    let cancellationMilliseconds: Double
    let cancelledOutput: String
    let followupOutput: String
    let followupGeneratedTokens: Int
    let peakResidentBytes: UInt64
}

// MARK: - ConcurrentRecord

private struct ConcurrentRecord: Codable, Sendable {
    let route: String
    let model: String
    let requestCount: Int
    let totalMilliseconds: Double
    let outputs: [String]
    let peakResidentBytes: UInt64
}

// MARK: - SwitchRecord

private struct SwitchRecord: Codable, Sendable {
    let route: String
    let models: [String]
    let cycles: Int
    let runs: [RunRecord]
    let residentBytesAfterClear: [UInt64]
    let peakResidentBytes: UInt64
}

// MARK: - LifecycleRecord

private struct LifecycleRecord: Codable, Sendable {
    let operation: String
    let model: String?
    let completed: Bool
}

// MARK: - TokenRecorder

private final class TokenRecorder: @unchecked Sendable {
    private let lock: NSLock = .init()
    private var firstTokenUptime: UInt64?
    private var pieces: [String] = []
    private var waiters: [(threshold: Int, continuation: CheckedContinuation<Void, Never>)] = []

    func append(_ token: String) {
        let now = DispatchTime.now().uptimeNanoseconds
        var ready: [CheckedContinuation<Void, Never>] = []

        lock.lock()
        if firstTokenUptime == nil {
            firstTokenUptime = now
        }
        pieces.append(token)
        let count = pieces.count
        waiters.removeAll { waiter in
            if count >= waiter.threshold {
                ready.append(waiter.continuation)
                return true
            }
            return false
        }
        lock.unlock()

        for continuation in ready {
            continuation.resume()
        }
    }

    func wait(untilCount threshold: Int) async {
        await withCheckedContinuation { continuation in
            lock.lock()
            if pieces.count >= threshold {
                lock.unlock()
                continuation.resume()
            }
            else {
                waiters.append((threshold, continuation))
                lock.unlock()
            }
        }
    }

    func snapshot() -> (firstTokenUptime: UInt64?, count: Int, output: String) {
        lock.lock()
        defer { lock.unlock() }
        return (firstTokenUptime, pieces.count, pieces.joined())
    }
}

private let engine: SwamaEngine = .init()

private func currentResidentBytes() -> UInt64 {
    let process = Process()
    let pipe = Pipe()
    process.executableURL = URL(fileURLWithPath: "/bin/ps")
    process.arguments = ["-o", "rss=", "-p", String(ProcessInfo.processInfo.processIdentifier)]
    process.standardOutput = pipe
    process.standardError = FileHandle.nullDevice

    do {
        try process.run()
        process.waitUntilExit()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        guard process.terminationStatus == 0,
              let text = String(data: data, encoding: .utf8),
              let kibibytes = UInt64(text.trimmingCharacters(in: .whitespacesAndNewlines))
        else {
            return 0
        }

        return kibibytes * 1024
    }
    catch {
        return 0
    }
}

private func parameters(maxTokens: Int) -> GenerationOptions {
    GenerationOptions(
        maxTokens: maxTokens,
        temperature: 0,
        topP: 1,
        seed: 1
    )
}

private func runOnce(
    model: String,
    prompt: String,
    maxTokens: Int,
    round: Int
) async throws -> RunRecord {
    let recorder = TokenRecorder()
    let start = DispatchTime.now().uptimeNanoseconds
    let result = try await engine.generate(
        GenerationRequest(
            model: .init(model),
            messages: [.init(role: .user, text: prompt)],
            options: parameters(maxTokens: maxTokens)
        ),
        onEvent: { event in
            if case let .textDelta(token) = event {
                recorder.append(token)
            }
        }
    )
    let end = DispatchTime.now().uptimeNanoseconds
    let snapshot = recorder.snapshot()

    return RunRecord(
        route: "core",
        model: model,
        round: round,
        promptTokens: result.usage.promptTokens,
        generatedTokens: result.usage.completionTokens,
        ttftMilliseconds: snapshot.firstTokenUptime.map { Double($0 - start) / 1_000_000 },
        totalMilliseconds: Double(end - start) / 1_000_000,
        tokensPerSecond: result.metrics?.tokensPerSecond,
        output: snapshot.output.isEmpty ? result.output : snapshot.output,
        peakResidentBytes: currentResidentBytes()
    )
}

private func runCancellation(
    model: String,
    prompt: String,
    maxTokens: Int
) async throws -> CancelRecord {
    // Warm the model first so the cancellation arm exercises generation, not
    // dependency resolution or model loading.
    _ = try await runOnce(model: model, prompt: "Warm up.", maxTokens: 8, round: 0)

    let recorder = TokenRecorder()
    let task = Task {
        try await engine.generate(
            GenerationRequest(
                model: .init(model),
                messages: [.init(role: .user, text: prompt)],
                options: parameters(maxTokens: maxTokens)
            ),
            onEvent: { event in
                if case let .textDelta(token) = event {
                    recorder.append(token)
                }
            }
        )
    }

    await recorder.wait(untilCount: 2)
    guard !task.isCancelled else {
        throw ProbeError.cancellationDidNotStart
    }

    let cancelStart = DispatchTime.now().uptimeNanoseconds
    task.cancel()
    do {
        _ = try await task.value
        throw ProbeError.cancellationReturnedResponse
    }
    catch is CancellationError {
        // The public Core contract never returns a partial terminal response on cancellation.
    }
    let cancelEnd = DispatchTime.now().uptimeNanoseconds
    let cancelledSnapshot = recorder.snapshot()

    let followup = try await runOnce(
        model: model,
        prompt: "Reply with exactly: AFTER_CANCEL_OK",
        maxTokens: 24,
        round: 1
    )

    return CancelRecord(
        route: "core-cancel",
        model: model,
        observedTokensBeforeCancel: cancelledSnapshot.count,
        cancellationMilliseconds: Double(cancelEnd - cancelStart) / 1_000_000,
        cancelledOutput: cancelledSnapshot.output,
        followupOutput: followup.output,
        followupGeneratedTokens: followup.generatedTokens,
        peakResidentBytes: currentResidentBytes()
    )
}

private func runConcurrent(
    model: String,
    prompt: String,
    maxTokens: Int,
    requestCount: Int
) async throws -> ConcurrentRecord {
    _ = try await runOnce(model: model, prompt: "Warm up.", maxTokens: 8, round: 0)
    let start = DispatchTime.now().uptimeNanoseconds
    let outputs = try await withThrowingTaskGroup(of: String.self) { group in
        for request in 0 ..< requestCount {
            group.addTask {
                let record = try await runOnce(
                    model: model,
                    prompt: "\(prompt) Request \(request).",
                    maxTokens: maxTokens,
                    round: request
                )
                return record.output
            }
        }

        var values: [String] = []
        for try await value in group {
            values.append(value)
        }
        return values.sorted()
    }
    let end = DispatchTime.now().uptimeNanoseconds

    return ConcurrentRecord(
        route: "core-concurrent",
        model: model,
        requestCount: requestCount,
        totalMilliseconds: Double(end - start) / 1_000_000,
        outputs: outputs,
        peakResidentBytes: currentResidentBytes()
    )
}

private func runSwitching(
    models: [String],
    prompt: String,
    maxTokens: Int,
    cycles: Int,
    settleMilliseconds: Int
) async throws -> SwitchRecord {
    var runs: [RunRecord] = []
    var residentBytesAfterClear: [UInt64] = []

    for cycle in 0 ..< cycles {
        for (index, model) in models.enumerated() {
            try await runs.append(
                runOnce(
                    model: model,
                    prompt: "\(prompt) Cycle \(cycle), model \(index).",
                    maxTokens: maxTokens,
                    round: cycle * models.count + index
                )
            )
            await engine.clearCache()
            try await Task.sleep(nanoseconds: UInt64(settleMilliseconds) * 1_000_000)
            residentBytesAfterClear.append(currentResidentBytes())
        }
    }

    return SwitchRecord(
        route: "core-model-switch",
        models: models,
        cycles: cycles,
        runs: runs,
        residentBytesAfterClear: residentBytesAfterClear,
        peakResidentBytes: currentResidentBytes()
    )
}

private func encode(_ value: some Encodable) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let data = try encoder.encode(value)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))
}

// MARK: - SwamaAcceptanceProbe

@main
private struct SwamaAcceptanceProbe {
    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard let command = arguments.first else {
            throw ProbeError.invalidArguments(
                "expected generate, cancel, concurrent, switch, embed, models, fetch, remove, or clear"
            )
        }

        switch command {
        case "generate":
            guard arguments.count >= 5,
                  let maxTokens = Int(arguments[2]),
                  let rounds = Int(arguments[3]),
                  maxTokens > 0,
                  rounds > 0
            else {
                throw ProbeError.invalidArguments(
                    "usage: SwamaAcceptanceProbe generate MODEL MAX_TOKENS ROUNDS PROMPT"
                )
            }

            let model = arguments[1]
            let prompt = arguments.dropFirst(4).joined(separator: " ")
            var records: [RunRecord] = []
            for round in 0 ..< rounds {
                try await records.append(
                    runOnce(
                        model: model,
                        prompt: prompt,
                        maxTokens: maxTokens,
                        round: round
                    )
                )
            }
            try encode(records)

        case "cancel":
            guard arguments.count >= 4,
                  let maxTokens = Int(arguments[2]),
                  maxTokens > 2
            else {
                throw ProbeError.invalidArguments(
                    "usage: SwamaAcceptanceProbe cancel MODEL MAX_TOKENS PROMPT"
                )
            }

            try await encode(
                runCancellation(
                    model: arguments[1],
                    prompt: arguments.dropFirst(3).joined(separator: " "),
                    maxTokens: maxTokens
                )
            )

        case "concurrent":
            guard arguments.count >= 5,
                  let maxTokens = Int(arguments[2]),
                  let requests = Int(arguments[3]),
                  maxTokens > 0,
                  requests > 1
            else {
                throw ProbeError.invalidArguments(
                    "usage: SwamaAcceptanceProbe concurrent MODEL MAX_TOKENS REQUESTS PROMPT"
                )
            }

            try await encode(
                runConcurrent(
                    model: arguments[1],
                    prompt: arguments.dropFirst(4).joined(separator: " "),
                    maxTokens: maxTokens,
                    requestCount: requests
                )
            )

        case "switch":
            guard arguments.count >= 6,
                  let maxTokens = Int(arguments[2]),
                  let cycles = Int(arguments[3]),
                  let settleMilliseconds = Int(arguments[4]),
                  maxTokens > 0,
                  cycles > 0,
                  settleMilliseconds >= 0
            else {
                throw ProbeError.invalidArguments(
                    "usage: SwamaAcceptanceProbe switch MODEL_A,MODEL_B MAX_TOKENS CYCLES SETTLE_MS PROMPT"
                )
            }

            let models = arguments[1].split(separator: ",").map(String.init)
            guard models.count >= 2 else {
                throw ProbeError.invalidArguments("switch requires at least two comma-separated models")
            }

            try await encode(
                runSwitching(
                    models: models,
                    prompt: arguments.dropFirst(5).joined(separator: " "),
                    maxTokens: maxTokens,
                    cycles: cycles,
                    settleMilliseconds: settleMilliseconds
                )
            )

        case "embed":
            guard arguments.count >= 3 else {
                throw ProbeError.invalidArguments("usage: SwamaAcceptanceProbe embed MODEL INPUT...")
            }

            try await encode(engine.embed(.init(
                model: .init(arguments[1]),
                inputs: Array(arguments.dropFirst(2))
            )))

        case "models":
            try await encode(engine.models())

        case "fetch":
            guard arguments.count == 2 else {
                throw ProbeError.invalidArguments("usage: SwamaAcceptanceProbe fetch MODEL")
            }

            try await engine.fetch(.init(arguments[1]))
            try encode(LifecycleRecord(operation: "fetch", model: arguments[1], completed: true))

        case "remove":
            guard arguments.count == 2 else {
                throw ProbeError.invalidArguments("usage: SwamaAcceptanceProbe remove MODEL")
            }

            try await engine.remove(.init(arguments[1]))
            try encode(LifecycleRecord(operation: "remove", model: arguments[1], completed: true))

        case "clear":
            guard arguments.count <= 2 else {
                throw ProbeError.invalidArguments("usage: SwamaAcceptanceProbe clear [MODEL]")
            }

            if let model = arguments.dropFirst().first {
                await engine.clearCache(for: .init(model))
            }
            else {
                await engine.clearCache()
            }
            try encode(LifecycleRecord(
                operation: "clear",
                model: arguments.dropFirst().first,
                completed: true
            ))

        default:
            throw ProbeError.invalidArguments("unknown command: \(command)")
        }
    }
}

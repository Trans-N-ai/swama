import Darwin
import Foundation

// MARK: - PeakResidentSampler

actor PeakResidentSampler {
    private var peak: UInt64 = 0

    func observe(pid: pid_t) {
        peak = max(peak, residentBytes(pid: pid))
    }

    func value() -> UInt64 { peak }
}

func withServer<T>(
    executable: URL,
    environment: [String: String],
    paths: WorkspacePaths,
    operation: (Int) async throws -> T
) async throws -> (value: T, peakResidentBytes: UInt64, stderr: String) {
    let port = try freePort()
    let temporary = FileManager.default
        .temporaryDirectory
        .appendingPathComponent("swama-acceptance-server-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }

    let stdoutURL = temporary.appendingPathComponent("stdout")
    let stderrURL = temporary.appendingPathComponent("stderr")
    FileManager.default.createFile(atPath: stdoutURL.path, contents: nil)
    FileManager.default.createFile(atPath: stderrURL.path, contents: nil)
    let stdoutHandle = try FileHandle(forWritingTo: stdoutURL)
    let stderrHandle = try FileHandle(forWritingTo: stderrURL)
    defer {
        try? stdoutHandle.close()
        try? stderrHandle.close()
    }

    let process = Process()
    process.executableURL = executable
    process.arguments = ["serve", "--host", "127.0.0.1", "--port", String(port)]
    process.currentDirectoryURL = paths.package
    process.environment = environment
    process.standardOutput = stdoutHandle
    process.standardError = stderrHandle
    try process.run()

    let pid = process.processIdentifier
    let sampler = PeakResidentSampler()
    let samplingTask = Task.detached {
        while !Task.isCancelled {
            await sampler.observe(pid: pid)
            try? await Task.sleep(for: .milliseconds(20))
        }
    }

    func stop() async throws -> (UInt64, String) {
        if process.isRunning {
            kill(pid, SIGINT)
            let deadline = Date().addingTimeInterval(10)
            while process.isRunning, Date() < deadline {
                try? await Task.sleep(for: .milliseconds(20))
            }
            if process.isRunning {
                kill(pid, SIGKILL)
            }
        }
        process.waitUntilExit()
        samplingTask.cancel()
        _ = await samplingTask.result
        let peak = await sampler.value()
        try stdoutHandle.synchronize()
        try stderrHandle.synchronize()
        let stderr = try String(decoding: Data(contentsOf: stderrURL), as: UTF8.self)
        let returnCode = process.terminationReason == .uncaughtSignal
            ? -process.terminationStatus
            : process.terminationStatus
        guard returnCode == 0 || returnCode == -SIGINT else {
            throw AcceptanceFailure.failed("server exited unexpectedly (\(returnCode)): \(stderr.suffix(2000))")
        }

        return (peak, stderr)
    }

    do {
        try await waitForServer(port: port)
        let value = try await operation(port)
        guard process.isRunning else {
            throw AcceptanceFailure.failed("server crashed before the operation completed")
        }

        let (peak, stderr) = try await stop()
        return (value, peak, stderr)
    }
    catch {
        _ = try? await stop()
        throw error
    }
}

func httpStream(
    port: Int,
    model: String,
    prompt: String,
    maxTokens: Int,
    disconnectAfterFirst: Bool = false
) async throws -> JSONObject {
    let payload: JSONObject = [
        "model": model,
        "messages": [["role": "user", "content": prompt]],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": maxTokens,
        "stream": true
    ]
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
    request.httpMethod = "POST"
    request.timeoutInterval = 300
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try compactJSONData(payload)

    let clock = ContinuousClock()
    let started = clock.now
    let (bytes, response) = try await URLSession.shared.bytes(for: request)
    guard let http = response as? HTTPURLResponse else {
        throw AcceptanceFailure.unknown("HTTP completion returned a non-HTTP response")
    }
    guard http.statusCode == 200 else {
        throw AcceptanceFailure.failed("HTTP completion failed (\(http.statusCode))")
    }

    var firstContent: ContinuousClock.Instant?
    var output = ""
    var usage: JSONObject?
    var eventCount = 0
    for try await rawLine in bytes.lines {
        let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
        guard line.hasPrefix("data: ") else { continue }

        let dataText = String(line.dropFirst(6))
        if dataText == "[DONE]" { break }
        guard let data = dataText.data(using: .utf8),
              let event = try JSONSerialization.jsonObject(with: data) as? JSONObject
        else {
            throw AcceptanceFailure.unknown("invalid SSE JSON: \(dataText)")
        }

        eventCount += 1
        if let eventUsage = event["usage"] as? JSONObject {
            usage = eventUsage
        }
        guard let choices = event["choices"] as? [JSONObject],
              let first = choices.first,
              let delta = first["delta"] as? JSONObject,
              let content = delta["content"] as? String,
              !content.isEmpty
        else {
            continue
        }

        if firstContent == nil {
            firstContent = clock.now
        }
        output += content
        if disconnectAfterFirst {
            return [
                "disconnected": true,
                "ttft_ms": milliseconds(from: started, to: firstContent!),
                "events": eventCount
            ]
        }
    }

    guard let firstContent else {
        throw AcceptanceFailure.failed("HTTP stream completed without a content event")
    }

    return [
        "ttft_ms": milliseconds(from: started, to: firstContent),
        "total_ms": milliseconds(from: started, to: clock.now),
        "events": eventCount,
        "output": output,
        "usage": usage ?? NSNull()
    ]
}

private func waitForServer(port: Int) async throws {
    let deadline = Date().addingTimeInterval(30)
    while Date() < deadline {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/models")!)
        request.timeoutInterval = 1
        if let (_, response) = try? await URLSession.shared.data(for: request),
           (response as? HTTPURLResponse)?.statusCode == 200
        {
            return
        }
        try? await Task.sleep(for: .milliseconds(100))
    }
    throw AcceptanceFailure.failed("server did not become ready on port \(port)")
}

private func freePort() throws -> Int {
    let descriptor = socket(AF_INET, SOCK_STREAM, 0)
    guard descriptor >= 0 else {
        throw AcceptanceFailure.unknown("cannot create socket for port allocation")
    }

    defer { close(descriptor) }

    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = 0
    address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
    let bindResult = withUnsafePointer(to: &address) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(descriptor, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    guard bindResult == 0 else {
        throw AcceptanceFailure.unknown("cannot bind socket for port allocation")
    }

    var length = socklen_t(MemoryLayout<sockaddr_in>.size)
    let nameResult = withUnsafeMutablePointer(to: &address) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            getsockname(descriptor, $0, &length)
        }
    }
    guard nameResult == 0 else {
        throw AcceptanceFailure.unknown("cannot read allocated port")
    }

    return Int(UInt16(bigEndian: address.sin_port))
}

func milliseconds(
    from start: ContinuousClock.Instant,
    to end: ContinuousClock.Instant
) -> Double {
    let duration = start.duration(to: end)
    return Double(duration.components.seconds) * 1000
        + Double(duration.components.attoseconds) / 1_000_000_000_000_000
}

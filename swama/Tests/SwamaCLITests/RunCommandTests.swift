import ArgumentParser
import Foundation
@testable import Swama
import SwamaCore
@testable import SwamaRuntime
import Testing

// MARK: - RunCommandTests

@Suite("swama run Core adapter", .serialized)
struct RunCommandTests {
    @Test func inProcessCoreIsTheDefaultAndDirectRemainsCompatible() throws {
        let defaultCommand = try Run.parse(["org/model", "hello"])
        #expect(try defaultCommand.executionRoute() == .core)

        let directCommand = try Run.parse(["org/model", "hello", "--direct"])
        #expect(try directCommand.executionRoute() == .core)

        let serverCommand = try Run.parse(["org/model", "hello", "--server"])
        #expect(try serverCommand.executionRoute() == .server)

        let conflictingCommand = try Run.parse(["org/model", "hello", "--direct", "--server"])
        #expect(throws: ValidationError.self) {
            _ = try conflictingCommand.executionRoute()
        }
    }

    @Test func everyCoreOptionMapsWithoutAnUpstreamType() throws {
        let command = try Run.parse([
            "org/model", "describe", "--temperature", "0.25", "--top-p", "0.75",
            "--max-tokens", "42", "--repetition-penalty", "1.1", "--context-limit", "4096",
            "--image-paths", "/tmp/example.png", "--no-stream"
        ])

        let request = command.makeCoreRequest(modelName: "resolved/model")
        #expect(request.model == ModelID("resolved/model"))
        #expect(request.options.maxTokens == 42)
        #expect(request.options.temperature == 0.25)
        #expect(request.options.topP == 0.75)
        #expect(request.options.repetitionPenalty == 1.1)
        #expect(request.options.contextLimit == 4096)
        #expect(request.messages.count == 1)
        #expect(request.messages[0].role == .user)
        #expect(request.messages[0].content == [
            .text("describe"),
            .imageURL(URL(fileURLWithPath: "/tmp/example.png"))
        ])
    }

    @Test func executionFetchesOnceAndRoutesTheResolvedModel() async throws {
        let resolved = ModelID("org/resolved")
        let coreRecorder = RunExecutionRecorder()
        let coreCommand = try Run.parse(["alias", "hello"])
        try await coreCommand.execute(using: .init(
            fetch: {
                await coreRecorder.recordFetch($0)
                return resolved
            },
            core: { await coreRecorder.recordCore($0) },
            server: { await coreRecorder.recordServer($0) }
        ))
        #expect(await coreRecorder.snapshot() == .init(
            fetched: [ModelID("alias")],
            core: [resolved],
            server: []
        ))

        let serverRecorder = RunExecutionRecorder()
        let serverCommand = try Run.parse(["alias", "hello", "--server"])
        try await serverCommand.execute(using: .init(
            fetch: {
                await serverRecorder.recordFetch($0)
                return resolved
            },
            core: { await serverRecorder.recordCore($0) },
            server: { await serverRecorder.recordServer($0) }
        ))
        #expect(await serverRecorder.snapshot() == .init(
            fetched: [ModelID("alias")],
            core: [],
            server: [resolved]
        ))
    }

    @Test func serverFailureNeverFallsBackToCore() async throws {
        let recorder = RunExecutionRecorder()
        let command = try Run.parse(["alias", "hello", "--server"])
        await #expect(throws: RunExecutionTestError.self) {
            try await command.execute(using: .init(
                fetch: {
                    await recorder.recordFetch($0)
                    return .init("org/resolved")
                },
                core: { await recorder.recordCore($0) },
                server: {
                    await recorder.recordServer($0)
                    throw RunExecutionTestError.serverFailed
                }
            ))
        }
        #expect(await recorder.snapshot() == .init(
            fetched: [ModelID("alias")],
            core: [],
            server: [ModelID("org/resolved")]
        ))
    }

    @Test func readinessCancellationReturnsPromptly() async throws {
        let command = try Run.parse(["org/model", "hello", "--server"])
        let clock = ContinuousClock()
        let start = clock.now
        let task = Task {
            try await command.waitForServerReady(
                timeout: .seconds(1),
                checkInterval: .milliseconds(500),
                readiness: { false }
            )
        }

        try await Task.sleep(for: .milliseconds(20))
        task.cancel()
        await #expect(throws: CancellationError.self) {
            _ = try await task.value
        }
        #expect(start.duration(to: clock.now) < .milliseconds(200))
    }

    @Test func cliDiagnosticSessionEndsCancelled() async throws {
        let primary = LockedData()
        let recorder = SwamaDiagnosticRecorder(
            enabled: true,
            sessionID: UUID(uuidString: "11111111-2222-3333-4444-555555555555")!,
            primaryWrite: { primary.append($0) },
            fallbackWrite: { _ in },
            now: { Date(timeIntervalSince1970: 0) }
        )
        let previous = SwamaDiagnostics.installRecorderForTesting(recorder)
        defer { SwamaDiagnostics.restoreRecorderForTesting(previous) }

        await #expect(throws: CancellationError.self) {
            try await SwamaEngine.withCLIDiagnostics {
                throw CancellationError()
            }
        }
        recorder.flush()

        let events: [SwamaDiagnosticEvent]
        switch SwamaDiagnosticTimeline.parse(primary.data) {
        case let .valid(value):
            events = value
        case .degraded,
             .unknown:
            throw RunExecutionTestError.invalidDiagnostics
        }
        #expect(events.map(\.event) == [.sessionStarted, .sessionStopped])
        guard case let .string(mode)? = events.first?.data?["mode"] else {
            Issue.record("CLI diagnostic session is missing its mode")
            return
        }

        #expect(mode == "cli")
        #expect(events.last?.outcome == .cancelled)
    }

    @Test func explicitServerModeMapsSupportedOptionsAndRejectsContextLimit() throws {
        let command = try Run.parse([
            "org/model", "hello", "--server", "--temperature", "0.2", "--top-p", "0.8",
            "--max-tokens", "17", "--repetition-penalty", "1.05", "--no-stream"
        ])
        try command.validateServerOptions()

        let object = try #require(
            JSONSerialization.jsonObject(with: command.encodedServerRequest(modelName: "resolved/model"))
                as? [String: Any]
        )
        #expect(object["model"] as? String == "resolved/model")
        #expect((object["temperature"] as? NSNumber)?.floatValue == 0.2)
        #expect((object["top_p"] as? NSNumber)?.floatValue == 0.8)
        #expect((object["max_tokens"] as? NSNumber)?.intValue == 17)
        #expect((object["repetition_penalty"] as? NSNumber)?.floatValue == 1.05)
        #expect((object["stream"] as? NSNumber)?.boolValue == false)

        let unsupported = try Run.parse([
            "org/model", "hello", "--server", "--context-limit", "4096"
        ])
        #expect(throws: ValidationError.self) {
            try unsupported.validateServerOptions()
        }
    }
}

// MARK: - RunExecutionRecorder

private actor RunExecutionRecorder {
    func recordFetch(_ model: ModelID) {
        fetched.append(model)
    }

    func recordCore(_ model: ModelID) {
        core.append(model)
    }

    func recordServer(_ model: ModelID) {
        server.append(model)
    }

    func snapshot() -> Snapshot {
        .init(fetched: fetched, core: core, server: server)
    }

    private var fetched: [ModelID] = []
    private var core: [ModelID] = []
    private var server: [ModelID] = []
}

// MARK: - Snapshot

private struct Snapshot: Equatable {
    let fetched: [ModelID]
    let core: [ModelID]
    let server: [ModelID]
}

// MARK: - RunExecutionTestError

private enum RunExecutionTestError: Error {
    case serverFailed
    case invalidDiagnostics
}

// MARK: - LockedData

private final class LockedData: @unchecked Sendable {
    var data: Data {
        lock.withLock { storage }
    }

    func append(_ data: Data) {
        lock.withLock {
            storage.append(data)
        }
    }

    private let lock: NSLock = .init()
    private var storage: Data = .init()
}

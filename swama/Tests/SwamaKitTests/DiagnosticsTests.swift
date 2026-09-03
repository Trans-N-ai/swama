import Foundation
import MLX
import MLXEmbedders
@preconcurrency import MLXLMCommon
import MLXNN
@testable import SwamaKit
import Testing

// MARK: - DiagnosticsTests

@Suite("Agent-friendly diagnostics", .serialized)
struct DiagnosticsTests {
    @Test func envelopeIsStableAndOperationsAreJoinable() throws {
        let primary = LockedData()
        let recorder = makeRecorder(primary: primary)
        recorder.start(mode: .serve)
        let operation = recorder.makeOperation()
        recorder.record(
            level: .info,
            subsystem: "model",
            event: .modelLoadStarted,
            operation: operation,
            model: "Qwen/Qwen3-4B-MLX-4bit",
            data: [:]
        )
        recorder.record(
            level: .info,
            subsystem: "model",
            event: .modelLoadCompleted,
            operation: operation,
            model: "Qwen/Qwen3-4B-MLX-4bit",
            durationMs: 12.5,
            outcome: .ok,
            data: [
                "phases": .object([
                    "config_read": .null,
                    "config_decode": .null,
                    "model_graph": .null,
                    "tokenizer": .number(10),
                    "weights": .null
                ])
            ]
        )
        recorder.stop(outcome: .ok)

        let events = try validEvents(primary.data)
        #expect(events.map(\.seq) == [0, 1, 2, 3])
        #expect(events.map(\.event) == [
            .sessionStarted,
            .modelLoadStarted,
            .modelLoadCompleted,
            .sessionStopped
        ])
        #expect(events[1].op == operation.identifier)
        #expect(events[2].op == operation.identifier)
        #expect(events[2].durationMs == 12.5)
        #expect(events[2].outcome == .ok)
        #expect(SwamaDiagnosticTimeline.hasCompleteOperations(events))

        let firstLine = try #require(primary.data.split(separator: 0x0A).first)
        let object = try #require(JSONSerialization.jsonObject(with: Data(firstLine)) as? [String: Any])
        #expect(Set(object.keys) == ["schema", "ts", "seq", "level", "subsystem", "event", "session", "data"])
    }

    @Test func removingATerminalEventMakesTheTimelineIncomplete() throws {
        let primary = LockedData()
        let recorder = makeRecorder(primary: primary)
        recorder.start(mode: .cli)
        let operation = recorder.makeOperation()
        recorder.record(
            level: .info,
            subsystem: "generation",
            event: .generationStarted,
            operation: operation,
            model: "model"
        )
        recorder.record(
            level: .info,
            subsystem: "generation",
            event: .generationCompleted,
            operation: operation,
            model: "model",
            durationMs: 1,
            outcome: .ok,
            data: [
                "input_tokens": .integer(1),
                "output_tokens": .integer(1)
            ]
        )
        recorder.flush()

        let events = try validEvents(primary.data)
        #expect(SwamaDiagnosticTimeline.hasCompleteOperations(events))
        #expect(!SwamaDiagnosticTimeline.hasCompleteOperations(events.filter { $0.event != .generationCompleted }))
    }

    @Test func malformedOrTruncatedJSONLIsUnknown() {
        #expect(SwamaDiagnosticTimeline.parse(Data("{not-json}\n".utf8)) == .unknown)
        #expect(SwamaDiagnosticTimeline.parse(Data("{}".utf8)) == .unknown)
        #expect(!SwamaDiagnostics.isValidSnapshot(Data("{\"schema\":\"swama.diag/1\"}".utf8)))
    }

    @Test func unknownEnvelopeOrEventPayloadKeysFailClosed() throws {
        let primary = LockedData()
        let recorder = makeRecorder(primary: primary)
        recorder.start(mode: .cli)
        recorder.flush()
        let validLine = try #require(String(data: primary.data, encoding: .utf8))

        let extraEnvelope = "{\"prompt\":\"secret\"," + String(validLine.dropFirst())
        #expect(SwamaDiagnosticTimeline.parse(Data(extraEnvelope.utf8)) == .unknown)

        let extraPayload = validLine.replacingOccurrences(
            of: "\"mode\":\"cli\"",
            with: "\"mode\":\"cli\",\"response\":\"secret\""
        )
        #expect(SwamaDiagnosticTimeline.parse(Data(extraPayload.utf8)) == .unknown)

        let badTimestamp = validLine.replacingOccurrences(
            of: "1970-01-01T00:00:00.000Z",
            with: "today"
        )
        #expect(SwamaDiagnosticTimeline.parse(Data(badTimestamp.utf8)) == .unknown)
    }

    @Test func sequenceGapIsDegradedRatherThanNeverHappened() throws {
        let primary = LockedData()
        let recorder = makeRecorder(primary: primary)
        recorder.start(mode: .cli)
        recorder.stop(outcome: .ok)
        let changed = try #require(
            String(data: primary.data, encoding: .utf8)?.replacingOccurrences(of: "\"seq\":1", with: "\"seq\":2")
        )

        guard case let .degraded(events) = SwamaDiagnosticTimeline.parse(Data(changed.utf8)) else {
            Issue.record("a sequence gap must be observable as degraded")
            return
        }

        #expect(events.count == 2)
        #expect(SwamaDiagnostics.isValidSnapshot(Data(changed.utf8)))
    }

    @Test func primarySinkFailureDoesNotEscapeAndEmitsDropEvents() throws {
        let fallback = LockedData()
        let recorder = SwamaDiagnosticRecorder(
            enabled: true,
            sessionID: fixedSessionID,
            primaryWrite: { _ in throw CocoaError(.fileWriteNoPermission) },
            fallbackWrite: { fallback.append($0) },
            now: { Date(timeIntervalSince1970: 0) }
        )

        recorder.start(mode: .cli)
        recorder.stop(outcome: .ok)

        let lines = fallback.data.split(separator: 0x0A, omittingEmptySubsequences: true)
        #expect(lines.count == 2)
        for line in lines {
            let event = try JSONDecoder().decode(SwamaDiagnosticEvent.self, from: Data(line))
            #expect(event.event == .logDropped)
            #expect(event.data?["dropped_count"] == .integer(1))
        }
        #expect(!String(decoding: fallback.data, as: UTF8.self).contains("fileWriteNoPermission"))
    }

    @Test func slowPrimarySinkHasBoundedQueueAndObservableDrop() throws {
        let primary = LockedData()
        let fallback = LockedData()
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let fallbackWritten = DispatchSemaphore(value: 0)
        let recorder = SwamaDiagnosticRecorder(
            enabled: true,
            sessionID: fixedSessionID,
            primaryWrite: { data in
                entered.signal()
                release.wait()
                primary.append(data)
            },
            fallbackWrite: {
                fallback.append($0)
                fallbackWritten.signal()
            },
            maximumPendingWrites: 1,
            now: { Date(timeIntervalSince1970: 0) }
        )

        recorder.start(mode: .cli)
        #expect(entered.wait(timeout: .now() + 1) == .success)
        recorder.record(level: .info, subsystem: "session", event: .sessionStarted)
        #expect(fallbackWritten.wait(timeout: .now() + 1) == .success)
        release.signal()
        recorder.flush()

        let primaryEvents = try validEvents(primary.data)
        #expect(primaryEvents.map(\.seq) == [0])
        let fallbackLine = try #require(fallback.data.split(separator: 0x0A).first)
        let drop = try JSONDecoder().decode(SwamaDiagnosticEvent.self, from: Data(fallbackLine))
        #expect(drop.seq == 2)
        #expect(drop.event == .logDropped)
        #expect(drop.data?["dropped_count"] == .integer(1))
    }

    @Test func absoluteModelPathIsHashedAndSensitivePayloadHasNoInputSurface() throws {
        let primary = LockedData()
        let recorder = makeRecorder(primary: primary)
        recorder.start(mode: .cli)
        recorder.record(
            level: .info,
            subsystem: "model",
            event: .modelLoadStarted,
            operation: recorder.makeOperation(),
            model: "/Users/alice/private/prompt-secret"
        )
        recorder.flush()

        let text = String(decoding: primary.data, as: UTF8.self)
        #expect(!text.contains("/Users/alice"))
        #expect(!text.contains("prompt-secret"))
        let events = try validEvents(primary.data)
        #expect(events.last?.model?.hasPrefix("local:") == true)
    }

    @Test func disabledRecorderHasNoSideEffects() {
        let primary = LockedData()
        let recorder = SwamaDiagnosticRecorder(
            enabled: false,
            primaryWrite: { primary.append($0) },
            fallbackWrite: { primary.append($0) }
        )
        recorder.start(mode: .cli)
        recorder.record(level: .info, subsystem: "session", event: .sessionStarted)
        recorder.stop(outcome: .ok)
        #expect(primary.data.isEmpty)
    }

    @Test func operationCreatedBeforeSessionCannotLeaveAnOrphanTerminal() throws {
        let primary = LockedData()
        let recorder = makeRecorder(primary: primary)
        let inactiveOperation = recorder.makeOperation()
        recorder.record(
            level: .info,
            subsystem: "model",
            event: .modelLoadStarted,
            operation: inactiveOperation,
            model: "inactive"
        )
        recorder.start(mode: .app)
        recorder.record(
            level: .info,
            subsystem: "model",
            event: .modelLoadCompleted,
            operation: inactiveOperation,
            model: "inactive",
            durationMs: 1,
            outcome: .ok,
            data: ["phases": .object([
                "config_read": .null,
                "config_decode": .null,
                "model_graph": .null,
                "tokenizer": .null,
                "weights": .null
            ])]
        )
        recorder.stop(outcome: .ok)

        let events = try validEvents(primary.data)
        #expect(events.map(\.event) == [.sessionStarted, .sessionStopped])
        #expect(SwamaDiagnosticTimeline.hasCompleteOperations(events))
    }

    @Test func fileWriterRotatesBeforeCrossingItsBound() throws {
        let directory = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-diagnostics-tests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("events.jsonl")
        let writer = SwamaDiagnosticFileWriter(url: url, maximumBytes: 10)

        try writer.write(Data("123456".utf8))
        try writer.write(Data("abcdef".utf8))

        #expect(try Data(contentsOf: url) == Data("abcdef".utf8))
        #expect(try Data(contentsOf: url.appendingPathExtension("1")) == Data("123456".utf8))
        #expect(try writer.readSnapshot() == Data("123456abcdef".utf8))
    }

    @Test func customPathNeverChangesExistingParentPermissions() throws {
        let directory = FileManager.default
            .temporaryDirectory
            .appendingPathComponent("swama-diagnostics-permissions-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: directory.path)
        let writer = SwamaDiagnosticFileWriter(url: directory.appendingPathComponent("events.jsonl"))

        try writer.write(Data("{}\n".utf8))

        let attributes = try FileManager.default.attributesOfItem(atPath: directory.path)
        #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o755)
    }

    @Test func modelPoolLoadAndEvictionUseOneJoinableProductionTimeline() async throws {
        let primary = LockedData()
        let recorder = makeRecorder(primary: primary)
        recorder.start(mode: .cli)
        let previous = SwamaDiagnostics.installRecorderForTesting(recorder)
        defer { SwamaDiagnostics.restoreRecorderForTesting(previous) }
        let runner = makeDiagnosticEmbeddingRunner()
        let pool = ModelPool(
            memoryHooks: .init(activeMemory: { 0 }, clearCache: {}),
            loadOverrides: .init(
                modelExistsLocally: { _ in true },
                determineIsVLM: { _ in false },
                loadLanguage: { _, _ in fatalError("unexpected language load") },
                loadSpeechToText: { _ in fatalError("unexpected STT load") },
                loadTTS: { _, _, _ in fatalError("unexpected TTS load") },
                loadEmbedding: { _ in runner }
            )
        )

        let value = try await pool.runEmbeddingWithConcurrencyControl(modelName: "diagnostic-embedding") { _ in
            "ok"
        }
        #expect(value == "ok")
        await pool.remove(modelName: "diagnostic-embedding")
        recorder.flush()

        let events = try validEvents(primary.data)
        let modelEvents = events.filter { $0.model == "diagnostic-embedding" }
        #expect(modelEvents.map(\.event) == [
            .modelLoadStarted,
            .modelLoadCompleted,
            .modelEvictionStarted,
            .modelEvictionCompleted
        ])
        #expect(modelEvents[0].op == modelEvents[1].op)
        #expect(modelEvents[2].op == modelEvents[3].op)
        #expect(modelEvents[0].op != modelEvents[2].op)
        #expect(SwamaDiagnosticTimeline.hasCompleteOperations(modelEvents))
    }

    @Test func modelLoadFailureUsesBoundedCodeWithoutRawErrorText() async throws {
        let primary = LockedData()
        let recorder = makeRecorder(primary: primary)
        recorder.start(mode: .cli)
        let previous = SwamaDiagnostics.installRecorderForTesting(recorder)
        defer { SwamaDiagnostics.restoreRecorderForTesting(previous) }
        let pool = ModelPool(
            memoryHooks: .init(activeMemory: { 0 }, clearCache: {}),
            loadOverrides: .init(
                modelExistsLocally: { _ in true },
                determineIsVLM: { _ in false },
                loadLanguage: { _, _ in fatalError("unexpected language load") },
                loadSpeechToText: { _ in fatalError("unexpected STT load") },
                loadTTS: { _, _, _ in fatalError("unexpected TTS load") },
                loadEmbedding: { _ in throw DiagnosticSensitiveError() }
            )
        )

        do {
            _ = try await pool.runEmbeddingWithConcurrencyControl(modelName: "failure-model") { _ in "bad" }
            Issue.record("the synthetic loader must fail")
        }
        catch is DiagnosticSensitiveError {}
        recorder.flush()

        let text = String(decoding: primary.data, as: UTF8.self)
        #expect(!text.contains("prompt-secret"))
        #expect(!text.contains("api-key-secret"))
        #expect(!text.contains("/Users/alice/private"))
        let events = try validEvents(primary.data)
        let failed = try #require(events.first { $0.event == .modelLoadFailed })
        #expect(failed.error == .init(code: .modelLoadFailed, transient: false))
        #expect(SwamaDiagnosticTimeline.hasCompleteOperations(events.filter { $0.model == "failure-model" }))
    }

    @Test func teardownWinsWithCancelledLoadTerminalAndNeverCompleted() async throws {
        let primary = LockedData()
        let recorder = makeRecorder(primary: primary)
        recorder.start(mode: .cli)
        let previous = SwamaDiagnostics.installRecorderForTesting(recorder)
        defer { SwamaDiagnostics.restoreRecorderForTesting(previous) }
        let barrier = DiagnosticLoadBarrier(value: makeDiagnosticEmbeddingRunner())
        let pool = ModelPool(
            memoryHooks: .init(activeMemory: { 0 }, clearCache: {}),
            loadOverrides: .init(
                modelExistsLocally: { _ in true },
                determineIsVLM: { _ in false },
                loadLanguage: { _, _ in fatalError("unexpected language load") },
                loadSpeechToText: { _ in fatalError("unexpected STT load") },
                loadTTS: { _, _, _ in fatalError("unexpected TTS load") },
                loadEmbedding: { _ in await barrier.load() }
            )
        )
        let request = Task {
            try await pool.runEmbeddingWithConcurrencyControl(modelName: "cancelled-load") { _ in "bad" }
        }
        await barrier.waitUntilEntered()
        await pool.clearCache()
        await barrier.release()

        do {
            _ = try await request.value
            Issue.record("an invalidated load must not reach its operation")
        }
        catch is CancellationError {}
        recorder.flush()

        let events = try validEvents(primary.data)
        let modelEvents = events.filter { $0.model == "cancelled-load" }
        #expect(modelEvents.map(\.event) == [
            .modelLoadStarted,
            .modelEvictionStarted,
            .modelEvictionCompleted,
            .modelLoadCancelled
        ])
        #expect(!modelEvents.contains { $0.event == .modelLoadCompleted })
        #expect(modelEvents.last?.outcome == .cancelled)
        #expect(modelEvents.last?.error == nil)
        #expect(SwamaDiagnosticTimeline.hasCompleteOperations(modelEvents))

        let falseResurrection = String(decoding: primary.data, as: UTF8.self)
            .replacingOccurrences(of: "model.load.cancelled", with: "model.load.completed")
            .replacingOccurrences(of: "\"outcome\":\"cancelled\"", with: "\"outcome\":\"ok\"")
        let mutatedEvents = try validEvents(Data(falseResurrection.utf8))
            .filter { $0.model == "cancelled-load" }
        #expect(!SwamaDiagnosticTimeline.hasCompleteOperations(mutatedEvents))
    }

    @Test func modelRunnerEmitsFirstTokenAndSuccessfulTerminal() async throws {
        let primary = LockedData()
        let recorder = makeRecorder(primary: primary)
        recorder.start(mode: .cli)
        let previous = SwamaDiagnostics.installRecorderForTesting(recorder)
        defer { SwamaDiagnostics.restoreRecorderForTesting(previous) }
        let container = makeLifetimeTestContainer()
        let modelName = await container.configuration.name
        let runner = ModelRunner(container: container)

        let result = try await runner.runWithChatUsage(
            chatMessages: [.user("w1 w2")],
            parameters: .init(maxTokens: 2, temperature: 0, topP: 1, seed: 1)
        )
        recorder.flush()

        let events = try validEvents(primary.data)
        let generationEvents = events.filter { $0.subsystem == "generation" && $0.model == modelName }
        #expect(generationEvents.map(\.event) == [
            .generationStarted,
            .generationFirstToken,
            .generationCompleted
        ])
        #expect(Set(generationEvents.compactMap(\.op)).count == 1)
        #expect(generationEvents.last?.data?["input_tokens"] == .integer(Int64(result.promptTokens)))
        #expect(SwamaDiagnosticTimeline.hasCompleteOperations(generationEvents))
    }

    @Test func modelRunnerCancellationHasOneCancelledTerminal() async throws {
        let primary = LockedData()
        let recorder = makeRecorder(primary: primary)
        recorder.start(mode: .cli)
        let previous = SwamaDiagnostics.installRecorderForTesting(recorder)
        defer { SwamaDiagnostics.restoreRecorderForTesting(previous) }
        let container = makeLifetimeTestContainer()
        let modelName = await container.configuration.name
        let runner = ModelRunner(container: container)
        let task = Task {
            try await runner.runChat(
                userInput: .init(chat: [.user("w1 w2")]),
                parameters: .init(maxTokens: 20, temperature: 0, topP: 1, seed: 1),
                onToken: { _ in
                    withUnsafeCurrentTask { $0?.cancel() }
                }
            )
        }
        _ = try? await task.value
        recorder.flush()

        let events = try validEvents(primary.data)
        let generationEvents = events.filter { $0.subsystem == "generation" && $0.model == modelName }
        #expect(generationEvents.first?.event == .generationStarted)
        #expect(generationEvents.last?.event == .generationCancelled)
        #expect(generationEvents.count(where: {
            [.generationCompleted, .generationCancelled, .generationFailed].contains($0.event)
        }) == 1)
        #expect(SwamaDiagnosticTimeline.hasCompleteOperations(generationEvents))
    }

    private let fixedSessionID: UUID = .init(uuidString: "11111111-2222-3333-4444-555555555555")!

    private func makeRecorder(primary: LockedData) -> SwamaDiagnosticRecorder {
        SwamaDiagnosticRecorder(
            enabled: true,
            sessionID: fixedSessionID,
            primaryWrite: { primary.append($0) },
            fallbackWrite: { _ in },
            now: { Date(timeIntervalSince1970: 0) }
        )
    }

    private func validEvents(_ data: Data) throws -> [SwamaDiagnosticEvent] {
        switch SwamaDiagnosticTimeline.parse(data) {
        case let .valid(events):
            events
        case .degraded:
            throw DiagnosticsTestError.degraded
        case .unknown:
            throw DiagnosticsTestError.unknown
        }
    }
}

private func makeDiagnosticEmbeddingRunner() -> EmbeddingRunner {
    let context = EmbedderModelContext(
        configuration: .init(id: "diagnostics-test"),
        model: DiagnosticEmbeddingModel(),
        tokenizer: DiagnosticTokenizer(),
        pooling: Pooling(strategy: .none)
    )
    return EmbeddingRunner(container: EmbedderModelContainer(context: context))
}

// MARK: - DiagnosticEmbeddingModel

private final class DiagnosticEmbeddingModel: Module, EmbeddingModel {
    let vocabularySize = 8
    let maxPositionEmbeddings: Int? = nil

    func callAsFunction(
        _: MLXArray,
        positionIds _: MLXArray?,
        tokenTypeIds _: MLXArray?,
        attentionMask _: MLXArray?
    ) -> EmbeddingModelOutput {
        fatalError("diagnostics-only model must not run inference")
    }
}

// MARK: - DiagnosticTokenizer

private struct DiagnosticTokenizer: MLXLMCommon.Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text _: String, addSpecialTokens _: Bool) -> [Int] { [] }
    func decode(tokenIds _: [Int], skipSpecialTokens _: Bool) -> String { "" }
    func convertTokenToId(_: String) -> Int? { nil }
    func convertIdToToken(_: Int) -> String? { nil }
    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

// MARK: - DiagnosticsTestError

private enum DiagnosticsTestError: Error {
    case degraded
    case unknown
}

// MARK: - DiagnosticSensitiveError

private struct DiagnosticSensitiveError: Error, LocalizedError {
    var errorDescription: String? {
        "prompt-secret api-key-secret /Users/alice/private"
    }
}

// MARK: - DiagnosticLoadBarrier

private actor DiagnosticLoadBarrier {
    init(value: EmbeddingRunner) {
        self.value = value
    }

    func load() async -> EmbeddingRunner {
        entered = true
        enteredWaiters.forEach { $0.resume() }
        enteredWaiters.removeAll()
        if !released {
            await withCheckedContinuation { continuation in
                releaseWaiter = continuation
            }
        }
        return value
    }

    func waitUntilEntered() async {
        guard !entered else {
            return
        }

        await withCheckedContinuation { continuation in
            enteredWaiters.append(continuation)
        }
    }

    func release() {
        released = true
        releaseWaiter?.resume()
        releaseWaiter = nil
    }

    private let value: EmbeddingRunner
    private var entered = false
    private var released = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var releaseWaiter: CheckedContinuation<Void, Never>?
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

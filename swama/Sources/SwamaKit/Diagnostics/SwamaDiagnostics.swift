import CryptoKit
import Darwin
import Foundation

private let maximumDiagnosticModelIdentityBytes = 256
private let maximumDiagnosticModelIdentityComponentBytes = 96

private func requiresOpaqueDiagnosticModelIdentity(_ value: String) -> Bool {
    guard value.utf8.count <= maximumDiagnosticModelIdentityBytes else {
        return true
    }

    if value.hasPrefix("local:") {
        let digest = value.dropFirst("local:".count)
        return digest.count != 64 || !digest.unicodeScalars.allSatisfy { scalar in
            (0x30 ... 0x39).contains(scalar.value) || (0x61 ... 0x66).contains(scalar.value)
        }
    }

    let components = value.split(separator: "/", omittingEmptySubsequences: false)
    guard (1 ... 2).contains(components.count) else {
        return true
    }

    return !components.allSatisfy { component in
        guard !component.isEmpty,
              component.utf8.count <= maximumDiagnosticModelIdentityComponentBytes,
              let first = component.unicodeScalars.first,
              let last = component.unicodeScalars.last,
              isASCIIAlphanumeric(first),
              isASCIIAlphanumeric(last)
        else {
            return false
        }

        return component.unicodeScalars.allSatisfy { scalar in
            isASCIIAlphanumeric(scalar) || scalar.value == 0x2D || scalar.value == 0x2E || scalar.value == 0x5F
        }
    }
}

private func isASCIIAlphanumeric(_ scalar: Unicode.Scalar) -> Bool {
    (0x30 ... 0x39).contains(scalar.value)
        || (0x41 ... 0x5A).contains(scalar.value)
        || (0x61 ... 0x7A).contains(scalar.value)
}

// MARK: - SwamaDiagnosticMode

package enum SwamaDiagnosticMode: String, Codable, Sendable {
    case app
    case serve
    case cli
}

// MARK: - SwamaDiagnosticOutcome

package enum SwamaDiagnosticOutcome: String, Codable, Sendable {
    case ok
    case error
    case cancelled
}

// MARK: - SwamaDiagnosticErrorCode

package enum SwamaDiagnosticErrorCode: String, Codable, Sendable {
    case sessionFailed = "session_failed"
    case modelNotFound = "model_not_found"
    case modelLoadFailed = "model_load_failed"
    case tokenizerRejected = "tokenizer_rejected"
    case evictionFailed = "eviction_failed"
    case generationFailed = "generation_failed"
}

// MARK: - SwamaDiagnosticError

package struct SwamaDiagnosticError: Codable, Equatable, Sendable {
    package let code: SwamaDiagnosticErrorCode
    package let transient: Bool
}

// MARK: - SwamaModelLoadPhases

/// Timings exposed by the current upstream loader. A `nil` field means the upstream public API
/// does not expose that boundary; it must never be serialized as a fabricated zero.
package struct SwamaModelLoadPhases: Codable, Equatable, Sendable {
    package let configReadMs: Double?
    package let configDecodeMs: Double?
    package let modelGraphMs: Double?
    package let tokenizerMs: Double?
    package let weightsMs: Double?

    package init(
        configReadMs: Double? = nil,
        configDecodeMs: Double? = nil,
        modelGraphMs: Double? = nil,
        tokenizerMs: Double? = nil,
        weightsMs: Double? = nil
    ) {
        self.configReadMs = configReadMs
        self.configDecodeMs = configDecodeMs
        self.modelGraphMs = modelGraphMs
        self.tokenizerMs = tokenizerMs
        self.weightsMs = weightsMs
    }

    private enum CodingKeys: String, CodingKey {
        case configReadMs = "config_read"
        case configDecodeMs = "config_decode"
        case modelGraphMs = "model_graph"
        case tokenizerMs = "tokenizer"
        case weightsMs = "weights"
    }
}

// MARK: - SwamaDiagnosticOperation

package struct SwamaDiagnosticOperation: Equatable, Sendable {
    fileprivate let id: UUID
    fileprivate let startedAtNanoseconds: UInt64
    fileprivate let isActive: Bool

    package var identifier: String { id.uuidString.lowercased() }

    fileprivate func elapsedMilliseconds(nowNanoseconds: UInt64 = DispatchTime.now().uptimeNanoseconds) -> Double {
        Double(nowNanoseconds - startedAtNanoseconds) / 1_000_000
    }
}

// MARK: - SwamaDiagnosticEventName

enum SwamaDiagnosticEventName: String, Codable, CaseIterable, Sendable {
    case sessionStarted = "swama.session.started"
    case sessionStopped = "swama.session.stopped"
    case modelLoadStarted = "model.load.started"
    case modelLoadCompleted = "model.load.completed"
    case modelLoadCancelled = "model.load.cancelled"
    case modelLoadFailed = "model.load.failed"
    case tokenizerCacheHit = "tokenizer.cache.hit"
    case tokenizerCacheMiss = "tokenizer.cache.miss"
    case tokenizerCacheEvicted = "tokenizer.cache.evicted"
    case tokenizerCacheRejected = "tokenizer.cache.rejected"
    case modelEvictionStarted = "model.eviction.started"
    case modelEvictionCompleted = "model.eviction.completed"
    case modelEvictionFailed = "model.eviction.failed"
    case generationStarted = "generation.started"
    case generationFirstToken = "generation.first_token"
    case generationCompleted = "generation.completed"
    case generationCancelled = "generation.cancelled"
    case generationFailed = "generation.failed"
    case logDropped = "log.dropped"
}

// MARK: - SwamaDiagnosticLevel

enum SwamaDiagnosticLevel: String, Codable, Sendable {
    case debug
    case info
    case warn
    case error
}

// MARK: - SwamaDiagnosticValue

enum SwamaDiagnosticValue: Codable, Equatable, Sendable {
    case null
    case string(String)
    case integer(Int64)
    case number(Double)
    case boolean(Bool)
    case object([String: SwamaDiagnosticValue])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        }
        else if let value = try? container.decode(Bool.self) {
            self = .boolean(value)
        }
        else if let value = try? container.decode(Int64.self) {
            self = .integer(value)
        }
        else if let value = try? container.decode(Double.self) {
            self = .number(value)
        }
        else if let value = try? container.decode(String.self) {
            self = .string(value)
        }
        else {
            self = try .object(container.decode([String: SwamaDiagnosticValue].self))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null:
            try container.encodeNil()
        case let .string(value):
            try container.encode(value)
        case let .integer(value):
            try container.encode(value)
        case let .number(value):
            try container.encode(value)
        case let .boolean(value):
            try container.encode(value)
        case let .object(value):
            try container.encode(value)
        }
    }
}

// MARK: - SwamaDiagnosticEvent

struct SwamaDiagnosticEvent: Codable, Equatable, Sendable {
    let schema: String
    let ts: String
    let seq: UInt64
    let level: SwamaDiagnosticLevel
    let subsystem: String
    let event: SwamaDiagnosticEventName
    let session: String
    let op: String?
    let model: String?
    let durationMs: Double?
    let outcome: SwamaDiagnosticOutcome?
    let error: SwamaDiagnosticError?
    let data: [String: SwamaDiagnosticValue]?

    private enum CodingKeys: String, CodingKey {
        case schema
        case ts
        case seq
        case level
        case subsystem
        case event
        case session
        case op
        case model
        case durationMs = "duration_ms"
        case outcome
        case error
        case data
    }
}

// MARK: - SwamaDiagnosticTimeline

enum SwamaDiagnosticTimeline: Equatable {
    case valid([SwamaDiagnosticEvent])
    case degraded([SwamaDiagnosticEvent])
    case unknown

    static func parse(_ data: Data) -> SwamaDiagnosticTimeline {
        guard data.isEmpty || data.last == 0x0A else {
            return .unknown
        }

        let lines = data.split(separator: 0x0A, omittingEmptySubsequences: true)
        let decoder = JSONDecoder()
        var events: [SwamaDiagnosticEvent] = []
        for line in lines {
            let lineData = Data(line)
            guard let object = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any],
                  let event = try? decoder.decode(SwamaDiagnosticEvent.self, from: lineData),
                  hasValidEnvelopeKeys(object, event: event),
                  hasValidEventShape(event, object: object)
            else {
                return .unknown
            }
            guard event.schema == "swama.diag/1",
                  (event.outcome == .error) == (event.error != nil)
            else {
                return .unknown
            }

            let terminalEvents: Set<SwamaDiagnosticEventName> = [
                .sessionStopped,
                .modelLoadCompleted,
                .modelLoadCancelled,
                .modelLoadFailed,
                .tokenizerCacheRejected,
                .modelEvictionCompleted,
                .modelEvictionFailed,
                .generationCompleted,
                .generationCancelled,
                .generationFailed
            ]
            let isTerminal = terminalEvents.contains(event.event)
            guard isTerminal == (event.durationMs != nil),
                  isTerminal == (event.outcome != nil)
            else {
                return .unknown
            }

            switch event.event {
            case .generationCompleted,
                 .modelEvictionCompleted,
                 .modelLoadCompleted:
                guard event.outcome == .ok else {
                    return .unknown
                }

            case .generationFailed,
                 .modelEvictionFailed,
                 .modelLoadFailed,
                 .tokenizerCacheRejected:
                guard event.outcome == .error else {
                    return .unknown
                }

            case .generationCancelled:
                guard event.outcome == .cancelled else {
                    return .unknown
                }

            case .modelLoadCancelled:
                guard event.outcome == .cancelled else {
                    return .unknown
                }

            default:
                break
            }

            if event.event == .modelLoadCompleted
                || event.event == .modelLoadCancelled
                || event.event == .modelLoadFailed
            {
                guard let duration = event.durationMs,
                      case let .object(phases)? = event.data?["phases"],
                      Set(phases.keys) == ["config_read", "config_decode", "model_graph", "tokenizer", "weights"]
                else {
                    return .unknown
                }

                for phase in phases.values {
                    switch phase {
                    case .null:
                        continue
                    case let .number(value) where value >= 0 && value <= duration:
                        continue
                    case let .integer(value) where value >= 0 && Double(value) <= duration:
                        continue
                    default:
                        return .unknown
                    }
                }
            }
            events.append(event)
        }

        var expectedSequenceBySession: [String: UInt64] = [:]
        var degraded = false
        for event in events {
            let expected = expectedSequenceBySession[event.session, default: 0]
            if event.seq != expected {
                degraded = true
            }
            expectedSequenceBySession[event.session] = event.seq &+ 1
        }
        return degraded ? .degraded(events) : .valid(events)
    }

    private static func hasValidEnvelopeKeys(_ object: [String: Any], event: SwamaDiagnosticEvent) -> Bool {
        let base: Set<String> = ["schema", "ts", "seq", "level", "subsystem", "event", "session"]
        let operation = Set(["op", "model"])
        let terminal = Set(["duration_ms", "outcome"])
        let data = Set(["data"])
        let error = Set(["error"])
        let expected: Set<String> =
            switch event.event {
            case .sessionStarted:
                base.union(data)
            case .sessionStopped:
                base.union(terminal).union(event.outcome == .error ? error : [])
            case .modelLoadStarted:
                base.union(operation).union(data)
            case .modelLoadCancelled,
                 .modelLoadCompleted:
                base.union(operation).union(terminal).union(data)
            case .modelLoadFailed:
                base.union(operation).union(terminal).union(error).union(data)
            case .tokenizerCacheEvicted,
                 .tokenizerCacheHit,
                 .tokenizerCacheMiss:
                base.union(data)
            case .tokenizerCacheRejected:
                base.union(terminal).union(error).union(data)
            case .generationStarted,
                 .modelEvictionStarted:
                base.union(operation)
            case .modelEvictionCompleted:
                base.union(operation).union(terminal).union(data)
            case .generationFailed,
                 .modelEvictionFailed:
                base.union(operation).union(terminal).union(error)
            case .generationFirstToken:
                base.union(operation).union(data)
            case .generationCancelled,
                 .generationCompleted:
                base.union(operation).union(terminal).union(data)
            case .logDropped:
                base.union(data)
            }

        return Set(object.keys) == expected && !object.values.contains { $0 is NSNull }
    }

    private static func hasValidEventShape(_ event: SwamaDiagnosticEvent, object: [String: Any]) -> Bool {
        guard UUID(uuidString: event.session) != nil,
              hasTimestamp(event.ts),
              event.durationMs.map({ $0 >= 0 }) ?? true,
              event.subsystem == expectedSubsystem(for: event.event),
              hasExpectedOperationAndModel(event),
              hasCompatibleErrorCode(event)
        else {
            return false
        }

        if let operation = event.op, UUID(uuidString: operation) == nil {
            return false
        }
        if let model = event.model, requiresOpaqueDiagnosticModelIdentity(model) {
            return false
        }
        if let rawError = object["error"] as? [String: Any], Set(rawError.keys) != ["code", "transient"] {
            return false
        }

        let dataKeys = Set(event.data?.keys.map(\.self) ?? [])
        switch event.event {
        case .sessionStarted:
            guard dataKeys == ["pid", "version", "mode"],
                  case let .integer(pid)? = event.data?["pid"],
                  pid > 0,
                  case .string? = event.data?["version"],
                  case let .string(mode)? = event.data?["mode"]
            else {
                return false
            }

            return SwamaDiagnosticMode(rawValue: mode) != nil

        case .sessionStopped:
            return dataKeys.isEmpty

        case .modelLoadStarted:
            return event.op != nil && event.model != nil && dataKeys.isEmpty

        case .modelLoadCancelled,
             .modelLoadCompleted,
             .modelLoadFailed:
            return event.op != nil && event.model != nil && dataKeys == ["phases"]

        case .tokenizerCacheHit,
             .tokenizerCacheMiss,
             .tokenizerCacheRejected:
            return dataKeys == ["fingerprint"] && hasFingerprint(event.data?["fingerprint"])

        case .tokenizerCacheEvicted:
            guard dataKeys == ["fingerprint", "reason"],
                  hasFingerprint(event.data?["fingerprint"]),
                  case let .string(reason)? = event.data?["reason"]
            else {
                return false
            }

            return reason == "budget" || reason == "explicit"

        case .modelEvictionFailed,
             .modelEvictionStarted:
            return event.op != nil && event.model != nil && dataKeys.isEmpty

        case .modelEvictionCompleted:
            return event.op != nil
                && event.model != nil
                && dataKeys == ["generation", "resident_bytes_after"]
                && hasNonnegativeInteger(event.data?["generation"])
                && hasNonnegativeInteger(event.data?["resident_bytes_after"])

        case .generationFailed,
             .generationStarted:
            return event.op != nil && event.model != nil && dataKeys.isEmpty

        case .generationFirstToken:
            return event.op != nil
                && event.model != nil
                && dataKeys == ["ttft_ms"]
                && hasNonnegativeNumber(event.data?["ttft_ms"])

        case .generationCompleted:
            return event.op != nil
                && event.model != nil
                && dataKeys == ["input_tokens", "output_tokens"]
                && hasNonnegativeInteger(event.data?["input_tokens"])
                && hasNonnegativeInteger(event.data?["output_tokens"])

        case .generationCancelled:
            return event.op != nil
                && event.model != nil
                && dataKeys == ["output_tokens"]
                && hasNonnegativeInteger(event.data?["output_tokens"])

        case .logDropped:
            guard dataKeys == ["dropped_count"],
                  case let .integer(count)? = event.data?["dropped_count"]
            else {
                return false
            }

            return count > 0
        }
    }

    private static func expectedSubsystem(for event: SwamaDiagnosticEventName) -> String {
        switch event {
        case .sessionStarted,
             .sessionStopped:
            "session"
        case .modelEvictionCompleted,
             .modelEvictionFailed,
             .modelEvictionStarted,
             .modelLoadCancelled,
             .modelLoadCompleted,
             .modelLoadFailed,
             .modelLoadStarted:
            "model"
        case .tokenizerCacheEvicted,
             .tokenizerCacheHit,
             .tokenizerCacheMiss,
             .tokenizerCacheRejected:
            "tokenizer"
        case .generationCancelled,
             .generationCompleted,
             .generationFailed,
             .generationFirstToken,
             .generationStarted:
            "generation"
        case .logDropped:
            "diagnostics"
        }
    }

    private static func hasExpectedOperationAndModel(_ event: SwamaDiagnosticEvent) -> Bool {
        switch event.event {
        case .generationCancelled,
             .generationCompleted,
             .generationFailed,
             .generationFirstToken,
             .generationStarted,
             .modelEvictionCompleted,
             .modelEvictionFailed,
             .modelEvictionStarted,
             .modelLoadCancelled,
             .modelLoadCompleted,
             .modelLoadFailed,
             .modelLoadStarted:
            event.op != nil && event.model != nil
        case .logDropped,
             .sessionStarted,
             .sessionStopped,
             .tokenizerCacheEvicted,
             .tokenizerCacheHit,
             .tokenizerCacheMiss,
             .tokenizerCacheRejected:
            event.op == nil && event.model == nil
        }
    }

    private static func hasCompatibleErrorCode(_ event: SwamaDiagnosticEvent) -> Bool {
        switch event.event {
        case .sessionStopped where event.outcome == .error:
            event.error?.code == .sessionFailed
        case .modelLoadFailed:
            event.error?.code == .modelLoadFailed || event.error?.code == .modelNotFound
        case .tokenizerCacheRejected:
            event.error?.code == .tokenizerRejected
        case .modelEvictionFailed:
            event.error?.code == .evictionFailed
        case .generationFailed:
            event.error?.code == .generationFailed
        default:
            event.error == nil
        }
    }

    private static func hasFingerprint(_ value: SwamaDiagnosticValue?) -> Bool {
        guard case let .string(fingerprint)? = value, fingerprint.count == 64 else {
            return false
        }

        return fingerprint.allSatisfy { $0.isHexDigit && !$0.isUppercase }
    }

    private static func hasTimestamp(_ value: String) -> Bool {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        return formatter.date(from: value) != nil && value.hasSuffix("Z")
    }

    private static func hasNonnegativeInteger(_ value: SwamaDiagnosticValue?) -> Bool {
        guard case let .integer(number)? = value else {
            return false
        }

        return number >= 0
    }

    private static func hasNonnegativeNumber(_ value: SwamaDiagnosticValue?) -> Bool {
        switch value {
        case let .integer(number):
            number >= 0
        case let .number(number):
            number >= 0
        default:
            false
        }
    }

    static func hasCompleteOperations(_ events: [SwamaDiagnosticEvent]) -> Bool {
        var openOperations: [String: (
            session: String,
            start: SwamaDiagnosticEventName,
            model: String?,
            invalidated: Bool
        )] = [:]
        for event in events {
            guard let operation = event.op else {
                continue
            }

            let key = "\(event.session):\(operation)"
            switch event.event {
            case .generationStarted,
                 .modelLoadStarted:
                guard openOperations[key] == nil else {
                    return false
                }

                openOperations[key] = (event.session, event.event, event.model, false)

            case .modelEvictionStarted:
                guard openOperations[key] == nil else {
                    return false
                }

                for openKey in Array(openOperations.keys) {
                    guard var open = openOperations[openKey],
                          open.session == event.session,
                          open.start == .modelLoadStarted,
                          open.model == event.model
                    else {
                        continue
                    }

                    open.invalidated = true
                    openOperations[openKey] = open
                }
                openOperations[key] = (event.session, event.event, event.model, false)

            case .generationFirstToken:
                guard let open = openOperations[key],
                      open.start == .generationStarted,
                      open.model == event.model
                else {
                    return false
                }

            case .modelLoadCompleted,
                 .modelLoadFailed:
                guard let open = openOperations.removeValue(forKey: key),
                      open.start == .modelLoadStarted,
                      open.model == event.model,
                      !open.invalidated
                else {
                    return false
                }

            case .modelLoadCancelled:
                guard let open = openOperations.removeValue(forKey: key),
                      open.start == .modelLoadStarted,
                      open.model == event.model
                else {
                    return false
                }

            case .modelEvictionCompleted,
                 .modelEvictionFailed:
                guard let open = openOperations.removeValue(forKey: key),
                      open.start == .modelEvictionStarted,
                      open.model == event.model
                else {
                    return false
                }

            case .generationCancelled,
                 .generationCompleted,
                 .generationFailed:
                guard let open = openOperations.removeValue(forKey: key),
                      open.start == .generationStarted,
                      open.model == event.model
                else {
                    return false
                }

            default:
                break
            }
        }
        return openOperations.isEmpty
    }
}

// MARK: - SwamaDiagnosticRecorder

final class SwamaDiagnosticRecorder: @unchecked Sendable {
    typealias Write = @Sendable (Data) throws -> Void
    typealias Now = @Sendable () -> Date

    init(
        enabled: Bool,
        sessionID: UUID = UUID(),
        primaryWrite: @escaping Write,
        fallbackWrite: @escaping @Sendable (Data) -> Void,
        maximumPendingWrites: Int = 1024,
        now: @escaping Now = { Date() }
    ) {
        self.enabled = enabled
        self.sessionID = sessionID
        self.primaryWrite = primaryWrite
        self.fallbackWrite = fallbackWrite
        self.maximumPendingWrites = maximumPendingWrites
        self.now = now
    }

    func start(mode: SwamaDiagnosticMode) {
        guard enabled else {
            return
        }

        lock.withLock {
            startIfNeededLocked(mode: mode)
        }
    }

    func stop(outcome: SwamaDiagnosticOutcome, error: SwamaDiagnosticError? = nil) {
        guard enabled else {
            return
        }

        lock.withLock {
            guard started, !stopped else {
                return
            }

            stopped = true
            enqueueLocked(
                level: outcome == .error ? .error : .info,
                subsystem: "session",
                event: .sessionStopped,
                durationMs: sessionStartedAt.map(elapsedMilliseconds),
                outcome: outcome,
                error: error
            )
        }
        flush()
    }

    func makeOperation() -> SwamaDiagnosticOperation {
        .init(
            id: UUID(),
            startedAtNanoseconds: DispatchTime.now().uptimeNanoseconds,
            isActive: lock.withLock { enabled && started && !stopped }
        )
    }

    func record(
        level: SwamaDiagnosticLevel,
        subsystem: String,
        event: SwamaDiagnosticEventName,
        operation: SwamaDiagnosticOperation? = nil,
        model: String? = nil,
        durationMs: Double? = nil,
        outcome: SwamaDiagnosticOutcome? = nil,
        error: SwamaDiagnosticError? = nil,
        data: [String: SwamaDiagnosticValue]? = nil
    ) {
        guard enabled else {
            return
        }

        if let operation, !operation.isActive {
            return
        }

        lock.withLock {
            guard started, !stopped else {
                return
            }

            enqueueLocked(
                level: level,
                subsystem: subsystem,
                event: event,
                operation: operation,
                model: model.map(safeModelIdentity),
                durationMs: durationMs,
                outcome: outcome,
                error: error,
                data: data
            )
        }
    }

    func flush() {
        guard enabled else {
            return
        }

        ioQueue.sync {}
        fallbackQueue.sync {}
    }

    private let enabled: Bool
    private let sessionID: UUID
    private let primaryWrite: Write
    private let fallbackWrite: @Sendable (Data) -> Void
    private let maximumPendingWrites: Int
    private let now: Now
    private let lock: NSLock = .init()
    private let ioQueue: DispatchQueue = .init(label: "ai.transn.swama.diagnostics", qos: .utility)
    private let fallbackQueue: DispatchQueue = .init(label: "ai.transn.swama.diagnostics.fallback", qos: .utility)
    private var nextSequence: UInt64 = 0
    private var pendingWrites = 0
    private var pendingDropCount = 0
    private var dropNotificationScheduled = false
    private var started = false
    private var stopped = false
    private var sessionStartedAt: UInt64?

    private func startIfNeededLocked(mode: SwamaDiagnosticMode) {
        guard !started else {
            return
        }

        started = true
        sessionStartedAt = DispatchTime.now().uptimeNanoseconds
        enqueueLocked(
            level: .info,
            subsystem: "session",
            event: .sessionStarted,
            data: [
                "pid": .integer(Int64(ProcessInfo.processInfo.processIdentifier)),
                "version": .string(Self.version),
                "mode": .string(mode.rawValue)
            ]
        )
    }

    private func enqueueLocked(
        level: SwamaDiagnosticLevel,
        subsystem: String,
        event: SwamaDiagnosticEventName,
        operation: SwamaDiagnosticOperation? = nil,
        model: String? = nil,
        durationMs: Double? = nil,
        outcome: SwamaDiagnosticOutcome? = nil,
        error: SwamaDiagnosticError? = nil,
        data: [String: SwamaDiagnosticValue]? = nil
    ) {
        let envelope = SwamaDiagnosticEvent(
            schema: "swama.diag/1",
            ts: timestamp(now()),
            seq: nextSequence,
            level: level,
            subsystem: subsystem,
            event: event,
            session: sessionID.uuidString.lowercased(),
            op: operation?.identifier,
            model: model,
            durationMs: durationMs,
            outcome: outcome,
            error: error,
            data: data
        )
        nextSequence &+= 1
        guard pendingWrites < maximumPendingWrites else {
            scheduleDropNotificationLocked()
            return
        }

        pendingWrites += 1
        ioQueue.async { [self] in
            defer { lock.withLock { pendingWrites -= 1 } }
            write(envelope)
        }
    }

    private func scheduleDropNotificationLocked() {
        pendingDropCount += 1
        guard !dropNotificationScheduled else {
            return
        }

        dropNotificationScheduled = true
        fallbackQueue.async { [self] in
            let count = lock.withLock { () -> Int in
                let count = pendingDropCount
                pendingDropCount = 0
                dropNotificationScheduled = false
                return count
            }
            writeDropFallback(count: count)
        }
    }

    private func write(_ event: SwamaDiagnosticEvent) {
        guard let line = encodeLine(event) else {
            writeDropFallback(count: 1)
            return
        }

        do {
            try primaryWrite(line)
        }
        catch {
            writeDropFallback(count: 1)
        }
    }

    private func writeDropFallback(count: Int) {
        let dropEvent = lock.withLock { () -> SwamaDiagnosticEvent in
            let event = SwamaDiagnosticEvent(
                schema: "swama.diag/1",
                ts: timestamp(now()),
                seq: nextSequence,
                level: .warn,
                subsystem: "diagnostics",
                event: .logDropped,
                session: sessionID.uuidString.lowercased(),
                op: nil,
                model: nil,
                durationMs: nil,
                outcome: nil,
                error: nil,
                data: ["dropped_count": .integer(Int64(count))]
            )
            nextSequence &+= 1
            return event
        }
        if let line = encodeLine(dropEvent) {
            fallbackWrite(line)
        }
    }

    private func encodeLine(_ event: SwamaDiagnosticEvent) -> Data? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard var data = try? encoder.encode(event) else {
            return nil
        }

        data.append(0x0A)
        return data
    }

    private func timestamp(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        return formatter.string(from: date)
    }

    private func safeModelIdentity(_ value: String) -> String {
        guard requiresOpaqueDiagnosticModelIdentity(value) else {
            return value
        }

        let digest = SHA256.hash(data: Data(value.utf8)).map { String(format: "%02x", $0) }.joined()
        return "local:\(digest)"
    }

    private func elapsedMilliseconds(_ startedAt: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - startedAt) / 1_000_000
    }

    private static var version: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "2.3.0"
    }
}

// MARK: - SwamaDiagnosticFileWriter

final class SwamaDiagnosticFileWriter: @unchecked Sendable {
    init(url: URL, maximumBytes: Int = 8 * 1024 * 1024) {
        self.url = url
        self.maximumBytes = maximumBytes
    }

    func write(_ data: Data) throws {
        guard data.count <= maximumBytes else {
            throw SwamaDiagnosticFileWriterError.eventTooLarge
        }

        try prepareDirectory()
        try withFileLock {
            try rotateIfNeeded(incomingBytes: data.count)
            let descriptor = Darwin.open(url.path, O_CREAT | O_WRONLY | O_APPEND, S_IRUSR | S_IWUSR)
            guard descriptor >= 0 else {
                throw POSIXError(.init(rawValue: errno) ?? .EIO)
            }

            defer { Darwin.close(descriptor) }

            try data.withUnsafeBytes { rawBuffer in
                guard var pointer = rawBuffer.baseAddress else {
                    return
                }

                var remaining = rawBuffer.count
                while remaining > 0 {
                    let written = Darwin.write(descriptor, pointer, remaining)
                    guard written > 0 else {
                        throw POSIXError(.init(rawValue: errno) ?? .EIO)
                    }

                    remaining -= written
                    pointer = pointer.advanced(by: written)
                }
            }
        }
    }

    func readSnapshot() throws -> Data {
        try prepareDirectory()
        return try withFileLock {
            let rotatedURL = url.appendingPathExtension("1")
            var snapshot = (try? Data(contentsOf: rotatedURL)) ?? Data()
            snapshot.append((try? Data(contentsOf: url)) ?? Data())
            return snapshot
        }
    }

    private let url: URL
    private let maximumBytes: Int

    private func prepareDirectory() throws {
        let directory = url.deletingLastPathComponent()
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            return
        }

        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
    }

    private func withFileLock<T>(_ operation: () throws -> T) throws -> T {
        let lockURL = url.appendingPathExtension("lock")
        let lockDescriptor = Darwin.open(lockURL.path, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
        guard lockDescriptor >= 0 else {
            throw POSIXError(.init(rawValue: errno) ?? .EIO)
        }

        defer { Darwin.close(lockDescriptor) }
        guard flock(lockDescriptor, LOCK_EX) == 0 else {
            throw POSIXError(.init(rawValue: errno) ?? .EIO)
        }

        defer { flock(lockDescriptor, LOCK_UN) }
        return try operation()
    }

    private func rotateIfNeeded(incomingBytes: Int) throws {
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        guard size + incomingBytes > maximumBytes else {
            return
        }

        let rotatedURL = url.appendingPathExtension("1")
        try? FileManager.default.removeItem(at: rotatedURL)
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.moveItem(at: url, to: rotatedURL)
        }
    }
}

// MARK: - SwamaDiagnosticFileWriterError

private enum SwamaDiagnosticFileWriterError: Error {
    case eventTooLarge
}

// MARK: - SwamaDiagnostics

package enum SwamaDiagnostics {
    package static var logFileURL: URL {
        if let override = ProcessInfo.processInfo.environment["SWAMA_DIAGNOSTICS_PATH"], !override.isEmpty {
            return URL(fileURLWithPath: override)
        }
        return FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent(".swama/logs/events.jsonl")
    }

    package static func startSession(mode: SwamaDiagnosticMode) {
        recorder.start(mode: mode)
    }

    package static func stopSession(outcome: SwamaDiagnosticOutcome, error: SwamaDiagnosticError? = nil) {
        recorder.stop(outcome: outcome, error: error)
    }

    package static func withSession<T>(
        mode: SwamaDiagnosticMode,
        operation: () async throws -> T
    ) async throws -> T {
        startSession(mode: mode)
        do {
            let value = try await operation()
            stopSession(outcome: .ok)
            return value
        }
        catch is CancellationError {
            stopSession(outcome: .cancelled)
            throw CancellationError()
        }
        catch {
            stopSession(
                outcome: .error,
                error: .init(code: .sessionFailed, transient: false)
            )
            throw error
        }
    }

    package static func startModelLoad(model: String) -> SwamaDiagnosticOperation {
        let operation = recorder.makeOperation()
        recorder.record(
            level: .info,
            subsystem: "model",
            event: .modelLoadStarted,
            operation: operation,
            model: model,
            data: [:]
        )
        return operation
    }

    package static func completeModelLoad(
        _ operation: SwamaDiagnosticOperation,
        model: String,
        phases: SwamaModelLoadPhases
    ) {
        recorder.record(
            level: .info,
            subsystem: "model",
            event: .modelLoadCompleted,
            operation: operation,
            model: model,
            durationMs: operation.elapsedMilliseconds(),
            outcome: .ok,
            data: ["phases": .object(phases.values)]
        )
    }

    package static func failModelLoad(
        _ operation: SwamaDiagnosticOperation,
        model: String,
        code: SwamaDiagnosticErrorCode = .modelLoadFailed,
        transient: Bool = false,
        phases: SwamaModelLoadPhases = .init()
    ) {
        recorder.record(
            level: .error,
            subsystem: "model",
            event: .modelLoadFailed,
            operation: operation,
            model: model,
            durationMs: operation.elapsedMilliseconds(),
            outcome: .error,
            error: .init(code: code, transient: transient),
            data: ["phases": .object(phases.values)]
        )
    }

    package static func cancelModelLoad(
        _ operation: SwamaDiagnosticOperation,
        model: String,
        phases: SwamaModelLoadPhases
    ) {
        recorder.record(
            level: .info,
            subsystem: "model",
            event: .modelLoadCancelled,
            operation: operation,
            model: model,
            durationMs: operation.elapsedMilliseconds(),
            outcome: .cancelled,
            data: ["phases": .object(phases.values)]
        )
    }

    package static func startEviction(model: String) -> SwamaDiagnosticOperation {
        let operation = recorder.makeOperation()
        recorder.record(
            level: .info,
            subsystem: "model",
            event: .modelEvictionStarted,
            operation: operation,
            model: model
        )
        return operation
    }

    package static func completeEviction(
        _ operation: SwamaDiagnosticOperation,
        model: String,
        generation: UInt64,
        residentBytesAfter: Int
    ) {
        recorder.record(
            level: .info,
            subsystem: "model",
            event: .modelEvictionCompleted,
            operation: operation,
            model: model,
            durationMs: operation.elapsedMilliseconds(),
            outcome: .ok,
            data: [
                "generation": .integer(Int64(generation)),
                "resident_bytes_after": .integer(Int64(residentBytesAfter))
            ]
        )
    }

    package static func startGeneration(model: String) -> SwamaDiagnosticOperation {
        let operation = recorder.makeOperation()
        recorder.record(
            level: .info,
            subsystem: "generation",
            event: .generationStarted,
            operation: operation,
            model: model
        )
        return operation
    }

    package static func firstToken(_ operation: SwamaDiagnosticOperation, model: String) {
        recorder.record(
            level: .info,
            subsystem: "generation",
            event: .generationFirstToken,
            operation: operation,
            model: model,
            data: ["ttft_ms": .number(operation.elapsedMilliseconds())]
        )
    }

    package static func completeGeneration(
        _ operation: SwamaDiagnosticOperation,
        model: String,
        inputTokens: Int,
        outputTokens: Int
    ) {
        recorder.record(
            level: .info,
            subsystem: "generation",
            event: .generationCompleted,
            operation: operation,
            model: model,
            durationMs: operation.elapsedMilliseconds(),
            outcome: .ok,
            data: [
                "input_tokens": .integer(Int64(inputTokens)),
                "output_tokens": .integer(Int64(outputTokens))
            ]
        )
    }

    package static func cancelGeneration(
        _ operation: SwamaDiagnosticOperation,
        model: String,
        outputTokens: Int
    ) {
        recorder.record(
            level: .info,
            subsystem: "generation",
            event: .generationCancelled,
            operation: operation,
            model: model,
            durationMs: operation.elapsedMilliseconds(),
            outcome: .cancelled,
            data: ["output_tokens": .integer(Int64(outputTokens))]
        )
    }

    package static func failGeneration(
        _ operation: SwamaDiagnosticOperation,
        model: String,
        transient: Bool = false
    ) {
        recorder.record(
            level: .error,
            subsystem: "generation",
            event: .generationFailed,
            operation: operation,
            model: model,
            durationMs: operation.elapsedMilliseconds(),
            outcome: .error,
            error: .init(code: .generationFailed, transient: transient)
        )
    }

    package static func flush() {
        recorder.flush()
    }

    package static func isValidSnapshot(_ data: Data) -> Bool {
        switch SwamaDiagnosticTimeline.parse(data) {
        case .degraded,
             .valid:
            true
        case .unknown:
            false
        }
    }

    package static func readLogSnapshot() throws -> Data {
        try SwamaDiagnosticFileWriter(url: logFileURL).readSnapshot()
    }

    package static func residentBytes() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<natural_t>
            .size
        )
        let result = withUnsafeMutablePointer(to: &info) { infoPointer in
            infoPointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { integerPointer in
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    integerPointer,
                    &count
                )
            }
        }
        guard result == KERN_SUCCESS else {
            return 0
        }

        return Int(info.resident_size)
    }

    #if DEBUG
        @discardableResult
        static func installRecorderForTesting(_ replacement: SwamaDiagnosticRecorder) -> SwamaDiagnosticRecorder {
            recorderHolder.swap(replacement)
        }

        static func restoreRecorderForTesting(_ previous: SwamaDiagnosticRecorder) {
            _ = recorderHolder.swap(previous)
        }
    #endif

    private static var recorder: SwamaDiagnosticRecorder {
        recorderHolder.current
    }

    private static let recorderHolder: SwamaDiagnosticRecorderHolder = .init(liveRecorder)

    private static let liveRecorder: SwamaDiagnosticRecorder = {
        let environment = ProcessInfo.processInfo.environment
        let processName = ProcessInfo.processInfo.processName.lowercased()
        let isTestProcess = environment["XCTestConfigurationFilePath"] != nil
            || processName.contains("xctest")
            || processName.contains("packagetests")
        let enabled = environment["SWAMA_DIAGNOSTICS_DISABLED"] != "1" && !isTestProcess
        let writer = SwamaDiagnosticFileWriter(url: logFileURL)
        return .init(
            enabled: enabled,
            primaryWrite: { data in try writer.write(data) },
            fallbackWrite: { data in try? FileHandle.standardError.write(contentsOf: data) }
        )
    }()
}

// MARK: - SwamaDiagnosticRecorderHolder

private final class SwamaDiagnosticRecorderHolder: @unchecked Sendable {
    init(_ recorder: SwamaDiagnosticRecorder) {
        currentRecorder = recorder
    }

    var current: SwamaDiagnosticRecorder {
        lock.withLock { currentRecorder }
    }

    func swap(_ replacement: SwamaDiagnosticRecorder) -> SwamaDiagnosticRecorder {
        lock.withLock {
            let previous = currentRecorder
            currentRecorder = replacement
            return previous
        }
    }

    private let lock: NSLock = .init()
    private var currentRecorder: SwamaDiagnosticRecorder
}

private extension SwamaModelLoadPhases {
    var values: [String: SwamaDiagnosticValue] {
        [
            "config_read": configReadMs.map(SwamaDiagnosticValue.number) ?? .null,
            "config_decode": configDecodeMs.map(SwamaDiagnosticValue.number) ?? .null,
            "model_graph": modelGraphMs.map(SwamaDiagnosticValue.number) ?? .null,
            "tokenizer": tokenizerMs.map(SwamaDiagnosticValue.number) ?? .null,
            "weights": weightsMs.map(SwamaDiagnosticValue.number) ?? .null
        ]
    }
}

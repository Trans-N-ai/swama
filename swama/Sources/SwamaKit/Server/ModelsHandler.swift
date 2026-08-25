//
//  ModelsHandler.swift
//  SwamaKit
//

import Foundation
import NIOCore
import NIOHTTP1

// MARK: - ModelsHandler

public enum ModelsHandler {
    // MARK: Public

    public static func handle(requestHead: HTTPRequestHead, channel: Channel) async {
        do {
            var entries = [ModelEntry]()
            for rootDirectory in ModelPaths.allModelsDirectories {
                if let rootEntries = await scanRoot(at: rootDirectory) {
                    entries.append(contentsOf: rootEntries)
                }
                else {
                    NSLog(
                        "SwamaKit.ModelsHandler: models scan of \(rootDirectory.path) did not complete within \(scanTimeoutSeconds)s; skipping this root (likely inaccessible, e.g. awaiting a folder-access consent this process cannot present)"
                    )
                }
            }

            let responsePayload: [String: Any] = [
                "object": "list",
                "data": entries.map { entry -> [String: Any] in
                    [
                        "id": entry.id,
                        "object": "model",
                        "created": entry.created,
                        "owned_by": "swama",
                        "size_in_bytes": entry.sizeInBytes
                    ]
                }
            ]

            let jsonData = try JSONSerialization.data(withJSONObject: responsePayload)
            try await sendJSONResponse(channel: channel, data: jsonData, requestVersion: requestHead.version)
        }
        catch {
            NSLog("SwamaKit.ModelsHandler Error: Failed to process models request - \(error)")
            try? await sendErrorResponse(
                channel: channel,
                status: .internalServerError,
                error: "Internal Server Error: Could not process model list.",
                requestVersion: requestHead.version
            )
        }
    }

    // MARK: Internal

    /// The subset of `ModelInfo` the endpoint responds with; `Sendable` so scan
    /// results can cross from the scan queue back into the handler's task.
    struct ModelEntry {
        let id: String
        let created: Int
        let sizeInBytes: Int64
    }

    /// Runs `operation` for `key`, giving up after `seconds` and returning `nil`
    /// to this particular caller. Unlike a plain per-call timeout, concurrent
    /// callers sharing a `key` do not each start (or enqueue) their own
    /// operation: only the first caller to find `key` idle actually runs one —
    /// on a dedicated per-key queue, so a stuck operation costs one thread in
    /// total, however many callers arrive behind it — and every other caller,
    /// including ones that arrive after this caller has already timed out,
    /// attaches to that same in-flight run and is resumed with its result when
    /// it finishes. A caller's timeout detaches only that caller; it never
    /// cancels the run and never starts a replacement, so at most one
    /// operation per key is ever in flight, memory does not grow with however
    /// many callers time out while it runs, and the key returns to idle by
    /// itself the moment that operation returns — even if every waiter had
    /// already given up on it — so the next caller gets a fresh run rather
    /// than inheriting a stale one.
    static func coalescedRun<T: Sendable>(
        key: String,
        seconds: TimeInterval,
        operation: @escaping @Sendable () -> T
    ) async -> T? {
        await withCheckedContinuation { (continuation: CheckedContinuation<T?, Never>) in
            let pending = PendingResult(continuation)
            let waiterID = ObjectIdentifier(pending)

            let shouldStartRun = scanCoalescer.attach(key: key, waiterID: waiterID) { anyResult in
                pending.resume(anyResult as? T)
            }
            if shouldStartRun {
                scanQueues.queue(for: key).async {
                    scanCoalescer.complete(key: key, result: operation())
                }
            }

            DispatchQueue.global().asyncAfter(deadline: .now() + seconds) {
                scanCoalescer.detach(key: key, waiterID: waiterID)
                pending.resume(nil)
            }
        }
    }

    // MARK: Private

    private static let scanTimeoutSeconds: TimeInterval = 3

    private static let scanQueues: ScanQueueRegistry = .init()

    private static let scanCoalescer: ScanCoalescer = .init()

    /// Scans one models root with a bounded, coalesced wait: a root stuck
    /// behind a TCC consent prompt a headless process can never answer costs
    /// one thread in total no matter how many `/v1/models` requests land
    /// while it is stuck, leaves the other roots serving, and starts
    /// answering again by itself if the blocked call eventually completes.
    private static func scanRoot(at rootDirectory: URL) async -> [ModelEntry]? {
        await coalescedRun(
            key: rootDirectory.path,
            seconds: scanTimeoutSeconds
        ) {
            ModelManager.scanModelsDirectory(at: rootDirectory).map { info in
                ModelEntry(id: info.id, created: info.created, sizeInBytes: info.sizeInBytes)
            }
        }
    }

    /// Resumes a continuation exactly once, whichever of completion or timeout
    /// gets there first.
    private final class PendingResult<T: Sendable>: @unchecked Sendable {
        init(_ continuation: CheckedContinuation<T?, Never>) {
            self.continuation = continuation
        }

        func resume(_ value: T?) {
            lock.lock()
            let pendingContinuation = continuation
            continuation = nil
            lock.unlock()
            pendingContinuation?.resume(returning: value)
        }

        private let lock: NSLock = .init()
        private var continuation: CheckedContinuation<T?, Never>?
    }

    private final class ScanQueueRegistry: @unchecked Sendable {
        func queue(for path: String) -> DispatchQueue {
            lock.lock()
            defer { lock.unlock() }
            if let existing = queues[path] {
                return existing
            }
            let created = DispatchQueue(label: "swama.models-scan")
            queues[path] = created
            return created
        }

        private let lock: NSLock = .init()
        private var queues: [String: DispatchQueue] = [:]
    }

    /// Coalesces every concurrent caller for a given key onto a single
    /// in-flight run, so `coalescedRun` never enqueues more than one
    /// operation per key regardless of how many callers arrive — or time
    /// out — while it is running. `coalescedRun` is generic per call site,
    /// but this registry backs every one of its instantiations (Swift does
    /// not allow stored `static` state inside a generic type), so waiters
    /// are held as type-erased `(Any) -> Void` resume closures; each closure
    /// is created by a `coalescedRun<T>` call that already knows how to cast
    /// the delivered value back to its own `T`.
    private final class ScanCoalescer: @unchecked Sendable {
        /// Attaches a waiter for `key` and reports whether it must also run
        /// the operation. Returns `true` exactly once per run — for the
        /// first caller to arrive while `key` is idle, who becomes
        /// responsible for calling `complete(key:result:)` — and `false` for
        /// every caller that joins that same run before it does.
        func attach(key: String, waiterID: ObjectIdentifier, resume: @escaping @Sendable (Any) -> Void) -> Bool {
            lock.lock()
            defer { lock.unlock() }
            if var run = runs[key] {
                run.waiters[waiterID] = resume
                runs[key] = run
                return false
            }
            runs[key] = Run(waiters: [waiterID: resume])
            return true
        }

        /// Drops one waiter without touching the run itself. Used by a
        /// caller's timeout: the underlying operation is never cancelled, so
        /// this only stops that one caller from being resumed a second time
        /// and keeps a permanently stuck key's memory bounded — each timed
        /// out caller's captured state is released on its own timeout
        /// instead of accumulating for as long as the key stays stuck.
        func detach(key: String, waiterID: ObjectIdentifier) {
            lock.lock()
            defer { lock.unlock() }
            runs[key]?.waiters.removeValue(forKey: waiterID)
        }

        /// Delivers `result` to every waiter still attached to `key` and
        /// returns the key to idle, so the next caller starts a fresh run
        /// instead of joining this one after the fact.
        func complete(key: String, result: Any) {
            lock.lock()
            let waiters = runs.removeValue(forKey: key)?.waiters ?? [:]
            lock.unlock()
            for resume in waiters.values {
                resume(result)
            }
        }

        private struct Run {
            var waiters: [ObjectIdentifier: @Sendable (Any) -> Void]
        }

        private let lock: NSLock = .init()
        private var runs: [String: Run] = [:]
    }

    private static func sendJSONResponse(
        channel: Channel,
        data: Data,
        requestVersion: HTTPVersion
    ) async throws {
        var buffer = channel.allocator.buffer(capacity: data.count)
        buffer.writeBytes(data)

        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json")
        headers.add(name: "Content-Length", value: "\(buffer.readableBytes)")
        headers.add(name: "Connection", value: "close")
        HTTPHandler.applyCORSHeaders(&headers)

        try await channel.writeAndFlush(
            HTTPServerResponsePart.head(HTTPResponseHead(
                version: requestVersion,
                status: .ok,
                headers: headers
            ))
        )

        try await channel.writeAndFlush(
            HTTPServerResponsePart.body(.byteBuffer(buffer))
        )

        try await channel.writeAndFlush(
            HTTPServerResponsePart.end(nil)
        )
    }

    private static func sendErrorResponse(
        channel: Channel,
        status: HTTPResponseStatus,
        error: String,
        requestVersion: HTTPVersion
    ) async throws {
        var buffer = channel.allocator.buffer(capacity: error.utf8.count)
        buffer.writeString(error)

        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "text/plain")
        headers.add(name: "Content-Length", value: "\(buffer.readableBytes)")
        headers.add(name: "Connection", value: "close")
        HTTPHandler.applyCORSHeaders(&headers)

        try await channel.writeAndFlush(
            HTTPServerResponsePart.head(HTTPResponseHead(
                version: requestVersion,
                status: status,
                headers: headers
            ))
        )

        try await channel.writeAndFlush(
            HTTPServerResponsePart.body(.byteBuffer(buffer))
        )

        try await channel.writeAndFlush(
            HTTPServerResponsePart.end(nil)
        )
    }
}

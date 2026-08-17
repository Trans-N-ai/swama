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
            var entries: [ModelEntry] = []
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
    struct ModelEntry: Sendable {
        let id: String
        let created: Int
        let sizeInBytes: Int64
    }

    /// Runs `operation` on `queue`, giving up after `seconds` and returning `nil`.
    /// The operation itself is not interrupted by the timeout: if it is stuck it
    /// keeps its queue (and one thread) until it returns, and a completion after
    /// the deadline is discarded.
    static func runWithTimeout<T: Sendable>(
        on queue: DispatchQueue,
        seconds: TimeInterval,
        operation: @escaping @Sendable () -> T
    ) async -> T? {
        await withCheckedContinuation { (continuation: CheckedContinuation<T?, Never>) in
            let pending = PendingResult(continuation)
            queue.async {
                pending.resume(operation())
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + seconds) {
                pending.resume(nil)
            }
        }
    }

    // MARK: Private

    private static let scanTimeoutSeconds: TimeInterval = 3

    private static let scanQueues = ScanQueueRegistry()

    /// Scans one models root with a bounded wait. Each root gets its own serial
    /// queue, so a scan that never returns — e.g. `~/Documents/...` blocked on a
    /// TCC consent prompt that a headless process can never answer — costs one
    /// thread in total, leaves the other roots serving, and starts answering
    /// again by itself if the blocked call eventually completes. Requests that
    /// arrive while a root is stuck only enqueue a closure behind it; they do
    /// not stack up additional blocked threads.
    private static func scanRoot(at rootDirectory: URL) async -> [ModelEntry]? {
        await runWithTimeout(
            on: scanQueues.queue(for: rootDirectory.path),
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

        private let lock = NSLock()
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

        private let lock = NSLock()
        private var queues: [String: DispatchQueue] = [:]
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

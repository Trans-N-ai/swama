import Foundation
import NIOCore
import NIOHTTP1
import SwamaCore

// MARK: - ResponsesHandler

/// OpenAI **Responses API** (`POST /v1/responses`) as a thin HTTP <-> SwamaCore adapter.
///
/// This is an honest, minimal-but-real subset. It deliberately does NOT pretend to
/// support server-side state, hosted tools, or structured output: every unsupported
/// input is rejected with an explicit `400`, never silently downgraded to Chat
/// Completions behaviour. See ``RejectionReason`` for the exact refusal set.
///
/// The wire contract follows the official OpenAI Responses references:
///   - create:  https://developers.openai.com/api/reference/resources/responses/methods/create
///   - streaming events: https://developers.openai.com/api/reference/resources/responses/streaming-events
public enum ResponsesHandler {
    // MARK: Entry points

    public static func handle(
        requestHead: HTTPRequestHead,
        body: ByteBuffer?,
        channel: Channel
    ) async {
        await handle(
            requestHead: requestHead,
            body: body,
            channel: channel,
            engine: ServerCoreEngine.shared
        )
    }

    static func handle(
        requestHead _: HTTPRequestHead,
        body: ByteBuffer?,
        channel: Channel,
        engine: SwamaEngine
    ) async {
        let parsed: ParsedRequest
        do {
            parsed = try parse(body)
        }
        catch let rejection as RejectionReason {
            try? await respondError(channel: channel, status: .badRequest, reason: rejection)
            return
        }
        catch {
            try? await respondError(
                channel: channel,
                status: .badRequest,
                reason: .malformed(error.localizedDescription)
            )
            return
        }

        do {
            if parsed.stream {
                try await respondStreaming(channel: channel, engine: engine, parsed: parsed)
            }
            else {
                try await respondNonStreaming(channel: channel, engine: engine, parsed: parsed)
            }
        }
        catch let coreError as SwamaError {
            try? await respondError(channel: channel, status: status(for: coreError), reason: .core(coreError))
        }
        catch {
            guard channel.isActive else {
                return
            }

            try? await respondError(
                channel: channel,
                status: .internalServerError,
                reason: .malformed(error.localizedDescription)
            )
        }
    }

    // MARK: Non-streaming

    private static func respondNonStreaming(
        channel: Channel,
        engine: SwamaEngine,
        parsed: ParsedRequest
    ) async throws {
        let responseID = Self.newResponseID()
        let createdAt = Self.now()

        let result = try await engine.generate(parsed.request)

        let object = responseObject(
            id: responseID,
            createdAt: createdAt,
            model: parsed.request.model.rawValue,
            result: result
        )
        try await writeJSON(channel: channel, status: .ok, payload: object)
    }

    // MARK: Streaming (typed SSE, monotonic sequence_number from 0)

    private static func respondStreaming(
        channel: Channel,
        engine: SwamaEngine,
        parsed: ParsedRequest
    ) async throws {
        let responseID = Self.newResponseID()
        let createdAt = Self.now()
        let model = parsed.request.model.rawValue
        let sequence = SequenceCounter()

        try await startSSE(channel: channel)

        // response.created + response.in_progress
        try await emit(channel, sequence, "response.created", [
            "response": inProgressResponse(id: responseID, createdAt: createdAt, model: model)
        ])
        try await emit(channel, sequence, "response.in_progress", [
            "response": inProgressResponse(id: responseID, createdAt: createdAt, model: model)
        ])

        let messageItemID = Self.newItemID(prefix: "msg")
        let outputIndex = 0
        let contentIndex = 0
        let assembledText = TextAccumulator()
        let toolItems = ToolCallItemState()

        do {
            let result = try await CompletionsHandler.runCancellingOnClose(channel: channel) {
                try await engine.generate(parsed.request) { event in
                    switch event {
                    case let .textDelta(chunk):
                        if await assembledText.isEmpty {
                            try await emit(channel, sequence, "response.output_item.added", [
                                "output_index": outputIndex,
                                "item": messageItemStub(id: messageItemID),
                            ])
                            try await emit(channel, sequence, "response.content_part.added", [
                                "item_id": messageItemID,
                                "output_index": outputIndex,
                                "content_index": contentIndex,
                                "part": ["type": "output_text", "text": "", "annotations": []],
                            ])
                        }
                        await assembledText.append(chunk)
                        try await emit(channel, sequence, "response.output_text.delta", [
                            "item_id": messageItemID,
                            "output_index": outputIndex,
                            "content_index": contentIndex,
                            "delta": chunk,
                        ])

                    case let .toolCall(toolCall):
                        let itemID = await toolItems.add(toolCall)
                        let index = await toolItems.outputIndex(after: outputIndex)
                        try await emit(channel, sequence, "response.output_item.added", [
                            "output_index": index,
                            "item": functionCallStub(id: itemID, toolCall: toolCall),
                        ])
                        let arguments = Self.encodedArguments(toolCall.arguments)
                        try await emit(channel, sequence, "response.function_call_arguments.delta", [
                            "item_id": itemID,
                            "output_index": index,
                            "delta": arguments,
                        ])
                        try await emit(channel, sequence, "response.function_call_arguments.done", [
                            "item_id": itemID,
                            "output_index": index,
                            "arguments": arguments,
                        ])
                        try await emit(channel, sequence, "response.output_item.done", [
                            "output_index": index,
                            "item": functionCallItem(id: itemID, toolCall: toolCall, status: "completed"),
                        ])
                    }
                }
            }

            // Close out the assistant message item, if any text was produced.
            let finalText = await assembledText.value
            if await assembledText.started {
                try await emit(channel, sequence, "response.output_text.done", [
                    "item_id": messageItemID,
                    "output_index": outputIndex,
                    "content_index": contentIndex,
                    "text": finalText,
                ])
                try await emit(channel, sequence, "response.content_part.done", [
                    "item_id": messageItemID,
                    "output_index": outputIndex,
                    "content_index": contentIndex,
                    "part": ["type": "output_text", "text": finalText, "annotations": []],
                ])
                try await emit(channel, sequence, "response.output_item.done", [
                    "output_index": outputIndex,
                    "item": messageItem(id: messageItemID, text: finalText, status: "completed"),
                ])
            }

            let terminalStatus = (result.finishReason == .length) ? "incomplete" : "completed"
            let object = responseObject(
                id: responseID,
                createdAt: createdAt,
                model: model,
                result: result,
                status: terminalStatus
            )
            let terminalEvent = terminalStatus == "incomplete" ? "response.incomplete" : "response.completed"
            try await emit(channel, sequence, terminalEvent, ["response": object])
        }
        catch {
            guard channel.isActive else {
                return
            }

            try await emit(channel, sequence, "response.failed", [
                "response": failedResponse(
                    id: responseID,
                    createdAt: createdAt,
                    model: model,
                    message: error.localizedDescription
                ),
            ])
        }

        try await finishSSE(channel: channel)
    }
}

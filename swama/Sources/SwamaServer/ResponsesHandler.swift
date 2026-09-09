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

    /// Full typed lifecycle of one function-call output item; the assembler
    /// slot guarantees the terminal response reuses the same id and index.
    private static func emitFunctionCallLifecycle(
        channel: Channel,
        sequence: SequenceCounter,
        outputs: OutputAssembler,
        toolCall: ToolCall
    ) async throws {
        let slot = await outputs.addFunctionCall(toolCall)
        try await emit(channel, sequence, "response.output_item.added", [
            "output_index": slot.index,
            "item": functionCallStub(id: slot.id, toolCall: toolCall),
        ])
        let arguments = Self.encodedArguments(toolCall.arguments)
        try await emit(channel, sequence, "response.function_call_arguments.delta", [
            "item_id": slot.id,
            "output_index": slot.index,
            "delta": arguments,
        ])
        try await emit(channel, sequence, "response.function_call_arguments.done", [
            "item_id": slot.id,
            "output_index": slot.index,
            "arguments": arguments,
        ])
        try await emit(channel, sequence, "response.output_item.done", [
            "output_index": slot.index,
            "item": functionCallItem(id: slot.id, toolCall: toolCall, status: "completed"),
        ])
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
            parsed: parsed,
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
            "response": inProgressResponse(id: responseID, createdAt: createdAt, model: model, parsed: parsed)
        ])
        try await emit(channel, sequence, "response.in_progress", [
            "response": inProgressResponse(id: responseID, createdAt: createdAt, model: model, parsed: parsed)
        ])

        let contentIndex = 0
        let outputs = OutputAssembler()

        do {
            let result = try await CompletionsHandler.runCancellingOnClose(channel: channel) {
                try await engine.generate(parsed.request) { event in
                    switch event {
                    case let .textDelta(chunk):
                        let (slot, isFirst) = await outputs.startTextIfNeeded()
                        if isFirst {
                            try await emit(channel, sequence, "response.output_item.added", [
                                "output_index": slot.index,
                                "item": messageItemStub(id: slot.id),
                            ])
                            try await emit(channel, sequence, "response.content_part.added", [
                                "item_id": slot.id,
                                "output_index": slot.index,
                                "content_index": contentIndex,
                                "part": ["type": "output_text", "text": "", "annotations": []],
                            ])
                        }
                        await outputs.appendText(chunk)
                        try await emit(channel, sequence, "response.output_text.delta", [
                            "item_id": slot.id,
                            "output_index": slot.index,
                            "content_index": contentIndex,
                            "logprobs": [Any](),
                            "delta": chunk,
                        ])

                    case let .toolCall(toolCall):
                        try await emitFunctionCallLifecycle(
                            channel: channel,
                            sequence: sequence,
                            outputs: outputs,
                            toolCall: toolCall
                        )
                    }
                }
            }

            // The final Core result is authoritative for tool calls too: a call
            // that produced no callback event must still get its full item
            // lifecycle and terminal presence (deduplicated by call_id).
            let announcedCallIDs = await Set(outputs.records.compactMap { record -> String? in
                if case let .functionCall(_, toolCall) = record {
                    return toolCall.id
                }
                return nil
            })
            for toolCall in result.toolCalls {
                if let callID = toolCall.id, announcedCallIDs.contains(callID) {
                    continue
                }
                try await emitFunctionCallLifecycle(
                    channel: channel,
                    sequence: sequence,
                    outputs: outputs,
                    toolCall: toolCall
                )
            }

            let terminalStatus = (result.finishReason == .length) ? "incomplete" : "completed"
            // Close out the assistant message item with the same item id and
            // output_index the deltas carried. Core's final output is
            // authoritative over accumulated deltas; when Core produced final
            // text without any delta, the item lifecycle is synthesized here so
            // every terminal output item was announced as an event. A truncated
            // (`length`) response marks its message item incomplete, matching
            // the response status.
            let textStarted = await outputs.textSlotIfStarted != nil
            if !result.output.isEmpty || textStarted {
                let (slot, isFirst) = await outputs.startTextIfNeeded()
                if isFirst {
                    try await emit(channel, sequence, "response.output_item.added", [
                        "output_index": slot.index,
                        "item": messageItemStub(id: slot.id),
                    ])
                    try await emit(channel, sequence, "response.content_part.added", [
                        "item_id": slot.id,
                        "output_index": slot.index,
                        "content_index": contentIndex,
                        "part": ["type": "output_text", "text": "", "annotations": []],
                    ])
                }
                if !result.output.isEmpty {
                    await outputs.setFinalText(result.output)
                }
                let finalText = await outputs.assembledText
                let itemStatus = terminalStatus == "incomplete" ? "incomplete" : "completed"
                try await emit(channel, sequence, "response.output_text.done", [
                    "item_id": slot.id,
                    "output_index": slot.index,
                    "content_index": contentIndex,
                    "logprobs": [Any](),
                    "text": finalText,
                ])
                try await emit(channel, sequence, "response.content_part.done", [
                    "item_id": slot.id,
                    "output_index": slot.index,
                    "content_index": contentIndex,
                    "part": ["type": "output_text", "text": finalText, "annotations": []],
                ])
                try await emit(channel, sequence, "response.output_item.done", [
                    "output_index": slot.index,
                    "item": messageItem(id: slot.id, text: finalText, status: itemStatus),
                ])
            }
            let object = await responseObject(
                id: responseID,
                createdAt: createdAt,
                model: model,
                parsed: parsed,
                result: result,
                status: terminalStatus,
                streamedOutput: streamedOutput(
                    records: outputs.records,
                    text: outputs.assembledText,
                    textStatus: terminalStatus == "incomplete" ? "incomplete" : "completed"
                )
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
                    parsed: parsed,
                    message: error.localizedDescription
                ),
            ])
        }

        try await finishSSE(channel: channel)
    }
}

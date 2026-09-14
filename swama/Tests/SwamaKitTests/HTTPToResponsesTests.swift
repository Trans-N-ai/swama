import Foundation
import NIOCore
import NIOEmbedded
import NIOHTTP1
import SwamaCore
@testable import SwamaServer
import Testing

// MARK: - HTTPToResponsesTests

@MainActor @Suite("OpenAI Responses adapter", .serialized)
struct HTTPToResponsesTests {
    // MARK: Parsing / mapping

    @Test func stringInputBecomesUserMessageAndInstructionsBecomeSystem() throws {
        let parsed = try ResponsesHandler.parse(bytes(#"""
        {"model":"org/model","instructions":"be terse","input":"hello",
         "temperature":0.25,"top_p":0.7,"max_output_tokens":42}
        """#))
        #expect(parsed.stream == false)
        #expect(parsed.request.model == ModelID("org/model"))
        #expect(parsed.request.messages.count == 2)
        #expect(parsed.request.messages[0].role == .system)
        #expect(parsed.request.messages[0].content == [.text("be terse")])
        #expect(parsed.request.messages[1].role == .user)
        #expect(parsed.request.messages[1].content == [.text("hello")])
        #expect(parsed.request.options.maxTokens == 42)
        #expect(parsed.request.options.temperature == 0.25)
        #expect(parsed.request.options.topP == 0.7)
    }

    @Test func messageArrayInputWithImageAndFunctionToolMap() throws {
        let parsed = try ResponsesHandler.parse(bytes(#"""
        {"model":"m","stream":true,
         "input":[{"type":"message","role":"user","content":[
            {"type":"input_text","text":"describe"},
            {"type":"input_image","image_url":"https://example.com/a.png"}]}],
         "tools":[{"type":"function","name":"lookup","description":"d",
                   "parameters":{"type":"object","properties":{"k":{"type":"string"}}}}]}
        """#))
        #expect(parsed.stream == true)
        #expect(parsed.request.messages.count == 1)
        #expect(parsed.request.messages[0].content == [
            .text("describe"),
            .imageURL(URL(string: "https://example.com/a.png")!),
        ])
        #expect(parsed.request.tools.count == 1)
        #expect(parsed.request.tools[0].name == "lookup")
    }

    @Test func everyUnsupportedFeatureIsRejectedWith400NotSilentlyIgnored() {
        let rejected: [String] = [
            #"{"model":"m","input":"x","store":true}"#,
            #"{"model":"m","input":"x","previous_response_id":"resp_1"}"#,
            #"{"model":"m","input":"x","conversation":"conv_1"}"#,
            #"{"model":"m","input":"x","background":true}"#,
            #"{"model":"m","input":"x","truncation":"auto"}"#,
            #"{"model":"m","input":"x","include":["message.output_text.logprobs"]}"#,
            #"{"model":"m","input":"x","response_format":{"type":"json_schema"}}"#,
            #"{"model":"m","input":"x","text":{"format":{"type":"json_schema"}}}"#,
            #"{"model":"m","input":"x","tool_choice":"required"}"#,
            #"{"model":"m","input":"x","tool_choice":{"type":"function","name":"f"}}"#,
            #"{"model":"m","input":"x","tools":[{"type":"web_search_preview"}]}"#,
            #"{"model":"m","input":"x","tools":[{"type":"mcp","server_label":"s"}]}"#,
        ]
        for payload in rejected {
            #expect(throws: RejectionReason.self, "must reject: \(payload)") {
                _ = try ResponsesHandler.parse(bytes(payload))
            }
        }
    }

    @Test func supportedToolChoiceAndMissingModelBehaveCorrectly() throws {
        // auto / none are honourable and must NOT throw.
        _ = try ResponsesHandler.parse(bytes(#"{"model":"m","input":"x","tool_choice":"auto"}"#))
        _ = try ResponsesHandler.parse(bytes(#"{"model":"m","input":"x","tool_choice":"none"}"#))
        // missing model / empty input are 400s.
        #expect(throws: RejectionReason.self) {
            _ = try ResponsesHandler.parse(bytes(#"{"input":"x"}"#))
        }
        #expect(throws: RejectionReason.self) {
            _ = try ResponsesHandler.parse(bytes(#"{"model":"m","input":""}"#))
        }
        #expect(throws: RejectionReason.self) {
            _ = try ResponsesHandler.parse(bytes(#"{"model":"m","input":[]}"#))
        }
    }

    @Test func functionCallRoundTripItemsMapToCoreToolMessages() throws {
        let parsed = try ResponsesHandler.parse(bytes(#"""
        {"model":"m","input":[
           {"type":"message","role":"user","content":"look it up"},
           {"type":"function_call","call_id":"call-9","name":"lookup",
            "arguments":"{\"k\":\"v\",\"n\":3}"},
           {"type":"function_call_output","call_id":"call-9","output":"{\"answer\":42}"}]}
        """#))
        #expect(parsed.request.messages.count == 3)
        let callTurn = parsed.request.messages[1]
        #expect(callTurn.role == .assistant)
        #expect(callTurn.toolCalls == [
            ToolCall(id: "call-9", name: "lookup", arguments: ["k": .string("v"), "n": .int(3)]),
        ])
        let resultTurn = parsed.request.messages[2]
        #expect(resultTurn.role == .tool)
        #expect(resultTurn.toolCallID == "call-9")
        #expect(resultTurn.content == [.text(#"{"answer":42}"#)])
        // Malformed round-trip items stay hard 400s.
        #expect(throws: RejectionReason.self) {
            _ = try ResponsesHandler.parse(bytes(#"""
            {"model":"m","input":[{"type":"function_call","name":"lookup"}]}
            """#))
        }
        #expect(throws: RejectionReason.self) {
            _ = try ResponsesHandler.parse(bytes(#"""
            {"model":"m","input":[{"type":"function_call_output","call_id":"c"}]}
            """#))
        }
    }

    @Test func toolChoiceNoneActuallyWithholdsToolsFromTheCore() async throws {
        let backend = ResponsesStubBackend(response: .init(
            output: "plain answer", toolCalls: [],
            usage: .init(promptTokens: 1, completionTokens: 1), finishReason: .completed
        ))
        _ = try await run(backend: backend, body: #"""
        {"model":"m","input":"hi","tool_choice":"none",
         "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}
        """#)
        let requests = await backend.requests
        #expect(requests.count == 1)
        #expect(requests[0].tools.isEmpty)
    }

    @Test func knownFieldsOfTheWrongTypeAreRejectedNotCoerced() {
        let rejected: [String] = [
            #"{"model":"m","input":"x","stream":"true"}"#,
            #"{"model":"m","input":"x","max_output_tokens":1.5}"#,
            #"{"model":"m","input":"x","max_output_tokens":true}"#,
            #"{"model":"m","input":"x","temperature":"hot"}"#,
            #"{"model":"m","input":"x","top_p":false}"#,
            #"{"model":"m","input":"x","tools":{"type":"function"}}"#,
            #"{"model":"m","input":"x","store":"false"}"#,
            #"{"model":"m","input":"x","background":"no"}"#,
            #"{"model":"m","input":"x","instructions":42}"#,
            #"{"model":"m","input":"x","truncation":true}"#,
            #"{"model":"m","input":"x","include":"logprobs"}"#,
        ]
        for payload in rejected {
            #expect(throws: RejectionReason.self, "must reject wrong type: \(payload)") {
                _ = try ResponsesHandler.parse(bytes(payload))
            }
        }
    }

    @Test func unimplementedMeaningfulFieldsAreRejectedAndPlainTextFormatIsNot() {
        let rejected: [String] = [
            #"{"model":"m","input":"x","reasoning":{"effort":"low"}}"#,
            #"{"model":"m","input":"x","max_tool_calls":2}"#,
            #"{"model":"m","input":"x","service_tier":"flex"}"#,
            // `parallel_tool_calls:false` is deliberately NOT here: the Codex
            // profile accepts it and honours it as sequential emission, covered
            // by `parallelToolCallsFalseIsHonouredAsSequentialEmission`.
            #"{"model":"m","input":"x","text":{"verbosity":"low"}}"#,
            #"{"model":"m","input":"x","text":{"format":{"type":"json_object"}}}"#,
            #"{"model":"m","input":"x","tools":[{"type":"function","name":"f","strict":true,"parameters":{}}]}"#,
        ]
        for payload in rejected {
            #expect(throws: RejectionReason.self, "must reject: \(payload)") {
                _ = try ResponsesHandler.parse(bytes(payload))
            }
        }
        // Plain text format is the default behaviour, not Structured Outputs.
        #expect(throws: Never.self) {
            _ = try ResponsesHandler.parse(bytes(#"""
            {"model":"m","input":"x","text":{"format":{"type":"text"}}}
            """#))
        }
        #expect(throws: Never.self) {
            _ = try ResponsesHandler.parse(bytes(#"""
            {"model":"m","input":"x","tools":[{"type":"function","name":"f","strict":false,"parameters":{}}]}
            """#))
        }
    }

    @Test func topLevelSurfaceIsAnExactAllowlist() {
        let rejected: [String] = [
            #"{"model":"m","input":"x","context_management":{"mode":"auto"}}"#,
            #"{"model":"m","input":"x","metadata":{"k":"v"}}"#,
            #"{"model":"m","input":"x","moderation":"auto"}"#,
            #"{"model":"m","input":"x","prompt_cache_options":{"ttl":60}}"#,
            #"{"model":"m","input":"x","prompt_cache_retention":"24h"}"#,
            #"{"model":"m","input":"x","safety_identifier":"abc"}"#,
            #"{"model":"m","input":"x","stream_options":{"include_obfuscation":false}}"#,
            #"{"model":"m","input":"x","top_logprobs":3}"#,
            #"{"model":"m","input":"x","user":"u-1"}"#,
            #"{"model":"m","input":"x","totally_made_up_field":1}"#,
        ]
        for payload in rejected {
            #expect(throws: RejectionReason.self, "must reject: \(payload)") {
                _ = try ResponsesHandler.parse(bytes(payload))
            }
        }
    }

    @Test func nestedTypesAndEnumsAreStructurallyValidated() throws {
        let rejected: [String] = [
            // numeric item type / role must not default to message / user.
            #"{"model":"m","input":[{"type":7,"role":"user","content":"x"}]}"#,
            #"{"model":"m","input":[{"type":"message","role":7,"content":"x"}]}"#,
            // a wrong-typed text part must not be silently dropped from the array.
            #"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":7},{"type":"input_text","text":"kept"}]}]}"#,
            // unknown keys inside items are refused, not ignored.
            #"{"model":"m","input":[{"type":"message","role":"user","content":"x","attachments":[]}]}"#,
            // non-default image detail would be silently downgraded.
            #"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"https://e.com/a.png","detail":"high"}]}]}"#,
            // tools: explicit string type required, description typed, parameters an object.
            #"{"model":"m","input":"x","tools":[{"name":"f","parameters":{}}]}"#,
            #"{"model":"m","input":"x","tools":[{"type":"function","name":"f","description":7,"parameters":{}}]}"#,
            #"{"model":"m","input":"x","tools":[{"type":"function","name":"f","parameters":"schema"}]}"#,
            #"{"model":"m","input":"x","tools":[{"type":"function","name":"f","parameters":{},"handler":"local"}]}"#,
            // truncation is a closed enum.
            #"{"model":"m","input":"x","truncation":"banana"}"#,
        ]
        for payload in rejected {
            #expect(throws: RejectionReason.self, "must reject: \(payload)") {
                _ = try ResponsesHandler.parse(bytes(payload))
            }
        }
        // The supported spellings stay accepted.
        _ = try ResponsesHandler.parse(bytes(#"{"model":"m","input":"x","truncation":"disabled"}"#))
        _ = try ResponsesHandler.parse(bytes(#"""
        {"model":"m","input":[{"type":"message","role":"user","content":[
           {"type":"input_image","image_url":"https://e.com/a.png","detail":"auto"}]}]}
        """#))
    }

    @Test func functionCallArgumentsAreRequiredJSONObjectStrings() {
        let rejected: [String] = [
            #"{"model":"m","input":[{"type":"function_call","call_id":"c","name":"f"}]}"#,
            #"{"model":"m","input":[{"type":"function_call","call_id":"c","name":"f","arguments":""}]}"#,
            #"{"model":"m","input":[{"type":"function_call","call_id":"c","name":"f","arguments":"[1,2]"}]}"#,
            #"{"model":"m","input":[{"type":"function_call","call_id":"c","name":"f","arguments":{"k":1}}]}"#,
        ]
        for payload in rejected {
            #expect(throws: RejectionReason.self, "must reject: \(payload)") {
                _ = try ResponsesHandler.parse(bytes(payload))
            }
        }
    }

    @Test func acceptedMetadataValuesAreTypedAndIntegersMustFitInInt() {
        let rejected: [String] = [
            // 1e100 is integral but saturates through NSNumber.intValue.
            #"{"model":"m","input":"x","max_output_tokens":1e100}"#,
            // text.format is accepted only as a bare {"type":"text"}.
            #"{"model":"m","input":"x","text":{"format":{"type":"text","schema":{}}}}"#,
            // item envelope fields we accept but ignore still have to be typed.
            #"{"model":"m","input":[{"type":"message","role":"user","content":"x","status":7}]}"#,
            #"{"model":"m","input":[{"type":"message","role":"user","content":"x","id":7}]}"#,
            #"""
            {"model":"m","input":[{"type":"function_call","call_id":"c","name":"f",
             "arguments":"{}","status":7}]}
            """#,
        ]
        for payload in rejected {
            #expect(throws: RejectionReason.self, "must reject: \(payload)") {
                _ = try ResponsesHandler.parse(bytes(payload))
            }
        }
        // A plain in-range integer and a bare text format still pass.
        #expect(throws: Never.self) {
            _ = try ResponsesHandler.parse(bytes(#"""
            {"model":"m","input":"x","max_output_tokens":128,"text":{"format":{"type":"text"}}}
            """#))
        }
    }

    @Test func malformedImageWireFormsAreRejectedLikeTheChatAdapter() {
        // Both HTTP adapters share ImageInputParser; these are the exact inputs
        // that previously reached Core through the weaker Responses-only check.
        func payload(_ url: String) -> String {
            #"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"\#(url)"}]}]}"#
        }
        let rejected: [String] = [
            payload("https://example.com/%zz.png"), // invalid percent encoding
            payload("https://example.com/a.png\n"), // raw trailing newline
            payload("https://example.com:99999/a.png"), // port parses, out of range
            payload("https://[::1]:99999/a.png"), // bracketed IPv6, out of range
            // Overflowing ports do NOT parse, so `components.port` is nil and the
            // range check passes by default; only the explicit-port scan rejects
            // these. Without them this suite cannot police that guard.
            payload("https://example.com:999999999999999999999999/a.png"),
            payload("https://[::1]:999999999999999999999999/a.png"),
            payload("data:image/pn/g;base64,QQ=="), // extra slash in subtype
            payload("data:image/p g;base64,QQ=="), // space in subtype
            payload("data:image/png;base64,!!!!"), // invalid base64
            payload("data:image/png;base64,") // empty base64
        ]
        for body in rejected {
            #expect(throws: RejectionReason.self, "must reject: \(body)") {
                _ = try ResponsesHandler.parse(bytes(body))
            }
        }
        // Control: the supported forms still pass, so this is not a blanket refusal.
        #expect(throws: Never.self) {
            _ = try ResponsesHandler.parse(bytes(payload("https://example.com/a.png")))
            _ = try ResponsesHandler.parse(bytes(payload("data:image/png;base64,QQ==")))
        }
    }

    @Test func itemMetadataIsAClosedSchema() {
        let rejected: [String] = [
            // status must be the official enum, not any string
            #"{"model":"m","input":[{"type":"message","role":"user","content":"x","status":"banana"}]}"#,
            #"{"model":"m","input":[{"type":"function_call","call_id":"c","name":"n","arguments":"{}","status":"banana"}]}"#,
            #"{"model":"m","input":[{"type":"function_call_output","call_id":"c","output":"o","status":"banana"}]}"#,
            // an empty id is not an id
            #"{"model":"m","input":[{"type":"message","role":"user","content":"x","id":""}]}"#,
            // accepted-and-erased annotations/logprobs must fail closed
            #"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"t","annotations":5}]}]}"#,
            #"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"t","logprobs":"x"}]}]}"#,
            #"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"t","annotations":[{"a":1}]}]}]}"#
        ]
        for body in rejected {
            #expect(throws: RejectionReason.self, "must reject: \(body)") {
                _ = try ResponsesHandler.parse(bytes(body))
            }
        }
        // Controls: valid status, non-empty id and empty arrays are accepted.
        #expect(throws: Never.self) {
            _ = try ResponsesHandler.parse(bytes(
                #"{"model":"m","input":[{"type":"message","role":"user","content":"x","status":"completed","id":"msg_1"}]}"#
            ))
            _ = try ResponsesHandler.parse(bytes(
                #"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"t","annotations":[],"logprobs":[]}]}]}"#
            ))
        }
    }

    @Test func parallelToolCallsIsEchoedFromTheRequestNotHardcoded() throws {
        let asked = try ResponsesHandler.parse(bytes(
            #"{"model":"m","input":"x","parallel_tool_calls":false}"#
        ))
        #expect(asked.parallelToolCalls == false)
        let scaffold = ResponsesHandler.baseResponse(
            id: "resp_1", createdAt: 0, model: "m", parsed: asked, status: "completed"
        )
        #expect(scaffold["parallel_tool_calls"] as? Bool == false)

        let allowed = try ResponsesHandler.parse(bytes(
            #"{"model":"m","input":"x","parallel_tool_calls":true}"#
        ))
        #expect(allowed.parallelToolCalls == true)
        #expect(ResponsesHandler.baseResponse(
            id: "r", createdAt: 0, model: "m", parsed: allowed, status: "completed"
        )["parallel_tool_calls"] as? Bool ==
            true
        )

        // Absent: this server never issues calls concurrently, so it reports false.
        let absent = try ResponsesHandler.parse(bytes(#"{"model":"m","input":"x"}"#))
        #expect(absent.parallelToolCalls == false)
    }

    @Test func imageURLsUseOnlySupportedWireForms() {
        let rejected: [String] = [
            // Only the wire forms the hosted API accepts are supported; every
            // other scheme or shape is an unsupported input.
            #"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"file:///a/b.png"}]}]}"#,
            #"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"/relative/a.png"}]}]}"#,
            #"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"ftp://example.com/a.png"}]}]}"#,
            #"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"data:text/plain;base64,QQ=="}]}]}"#,
        ]
        for payload in rejected {
            #expect(throws: RejectionReason.self, "must reject: \(payload)") {
                _ = try ResponsesHandler.parse(bytes(payload))
            }
        }
        #expect(throws: Never.self) {
            _ = try ResponsesHandler.parse(bytes(#"""
            {"model":"m","input":[{"type":"message","role":"user","content":[
               {"type":"input_image","image_url":"https://example.com/a.png"},
               {"type":"input_image","image_url":"data:image/png;base64,QQ=="}]}]}
            """#))
        }
    }

    @Test func nilCallIDToolCallIsNotEmittedTwice() async throws {
        // Core's ToolCall.id is optional: the same call arriving via callback and
        // again in the final result must produce exactly one output item.
        let call = ToolCall(id: nil, name: "lookup", arguments: ["k": .string("v")])
        let backend = ResponsesStubBackend(
            response: .init(
                output: "", toolCalls: [call],
                usage: .init(promptTokens: 1, completionTokens: 1), finishReason: .toolCall
            ),
            events: [.toolCall(call)]
        )
        let events = try await runStream(backend: backend, body: #"""
        {"model":"m","input":"x","stream":true,
         "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}
        """#)
        let added = events.filter { $0["type"] as? String == "response.output_item.added" }
        #expect(added.count == 1)
        let terminal = try #require(events.last?["response"] as? [String: Any])
        let output = try #require(terminal["output"] as? [[String: Any]])
        #expect(output.count == 1)
    }

    @Test func genuinelyRepeatedNilCallIDToolCallsAreBothEmitted() async throws {
        // Two identical nil-id calls really made twice must not collapse into one.
        let call = ToolCall(id: nil, name: "lookup", arguments: ["k": .string("v")])
        let backend = ResponsesStubBackend(
            response: .init(
                output: "", toolCalls: [call, call],
                usage: .init(promptTokens: 1, completionTokens: 1), finishReason: .toolCall
            ),
            events: [.toolCall(call)]
        )
        let events = try await runStream(backend: backend, body: #"""
        {"model":"m","input":"x","stream":true,
         "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}
        """#)
        let terminal = try #require(events.last?["response"] as? [String: Any])
        let output = try #require(terminal["output"] as? [[String: Any]])
        #expect(output.count == 2)
    }

    @Test func codexEnvelopeIsAcceptedWithExplicitLocalSemantics() throws {
        // The fixed envelope Codex CLI always sends must get through; each field
        // has a documented local meaning rather than being silently honoured.
        let parsed = try ResponsesHandler.parse(bytes(#"""
        {"model":"m","input":"hi","stream":true,
         "client_metadata":{"session_id":"s-1","cli_version":"0.147.0"},
         "prompt_cache_key":"codex-session-1",
         "reasoning":{"summary":"auto"},
         "include":["reasoning.encrypted_content"],
         "parallel_tool_calls":false,
         "tools":[{"type":"function","name":"shell","parameters":{"type":"object"}}]}
        """#))
        #expect(parsed.request.model == ModelID("m"))
        #expect(parsed.request.tools.count == 1)
        #expect(parsed.stream == true)
    }

    @Test func adjacentSystemTurnsAreMergedForTheLocalChatTemplate() throws {
        // `instructions` and `developer` items both lower to system turns, which
        // every real Codex request contains together. Two adjacent system turns
        // make the local backend fail, so they are concatenated in order.
        let parsed = try ResponsesHandler.parse(bytes(#"""
        {"model":"m","instructions":"top-level guidance",
         "input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"dev one"}]},
                  {"type":"message","role":"developer","content":[{"type":"input_text","text":"dev two"}]},
                  {"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]}
        """#))
        #expect(parsed.request.messages.count == 2)
        #expect(parsed.request.messages[0].role == .system)
        #expect(parsed.request.messages[0].content == [
            .text("top-level guidance"), .text("dev one"), .text("dev two")
        ])
        #expect(parsed.request.messages[1].role == .user)
        // No two system turns may remain adjacent anywhere in the sequence.
        for pair in zip(parsed.request.messages, parsed.request.messages.dropFirst()) {
            #expect(!(pair.0.role == .system && pair.1.role == .system))
        }
    }

    @Test func mergingSystemTurnsPreservesOrderAndDoesNotTouchOtherRoles() {
        let input: [Message] = [
            Message(role: .system, text: "a"),
            Message(role: .system, text: "b"),
            Message(role: .user, text: "u1"),
            Message(role: .system, text: "c"),
            Message(role: .assistant, text: "r"),
            Message(role: .user, text: "u2")
        ]
        let merged = ResponsesHandler.mergingAdjacentSystemTurns(input)
        #expect(merged.map(\.role) == [.system, .user, .system, .assistant, .user])
        #expect(merged[0].content == [.text("a"), .text("b")])
        // A lone system turn later in the conversation is left exactly as it was.
        #expect(merged[2].content == [.text("c")])
        #expect(merged[4].content == [.text("u2")])
    }

    @Test func codexWebSearchToolIsAcceptedButNeverReachesTheModel() throws {
        // Codex CLI 0.147.0 always sends this tool and no client configuration
        // removes it. We accept the exact offline-only shape and DROP it, so the
        // model is never offered a capability this server cannot perform.
        let parsed = try ResponsesHandler.parse(bytes(#"""
        {"model":"m","input":"hi",
         "tools":[{"type":"function","name":"exec_command","parameters":{"type":"object"}},
                  {"type":"web_search","external_web_access":false},
                  {"type":"function","name":"update_plan","parameters":{"type":"object"}}]}
        """#))
        // The request is accepted...
        #expect(parsed.request.tools.count == 2)
        // ...and what the model can see contains ONLY the real function tools.
        #expect(parsed.request.tools.map(\.name).sorted() == ["exec_command", "update_plan"])
        #expect(!parsed.request.tools.contains { $0.name.contains("web_search") })
    }

    @Test func onlyTheExactOfflineWebSearchShapeIsAccepted() {
        let rejected: [String] = [
            // Real web access is a capability we do not have; silence would lie.
            #"{"model":"m","input":"x","tools":[{"type":"web_search","external_web_access":true}]}"#,
            // Absent flag is not the offline shape; it must not default to accepted.
            #"{"model":"m","input":"x","tools":[{"type":"web_search"}]}"#,
            #"{"model":"m","input":"x","tools":[{"type":"web_search","external_web_access":null}]}"#,
            #"{"model":"m","input":"x","tools":[{"type":"web_search","external_web_access":"false"}]}"#,
            // Any richer configuration (filters, domains, ...) is a request for
            // behaviour we cannot honour.
            #"{"model":"m","input":"x","tools":[{"type":"web_search","external_web_access":false,"filters":{"allowed_domains":["a.com"]}}]}"#,
            #"{"model":"m","input":"x","tools":[{"type":"web_search","external_web_access":false,"search_context_size":"low"}]}"#,
            // Other hosted kinds stay refused outright.
            #"{"model":"m","input":"x","tools":[{"type":"file_search"}]}"#,
            #"{"model":"m","input":"x","tools":[{"type":"namespace","name":"multi_agent_v1","tools":[]}]}"#,
        ]
        for payload in rejected {
            #expect(throws: RejectionReason.self, "must reject: \(payload)") {
                _ = try ResponsesHandler.parse(bytes(payload))
            }
        }
    }

    @Test func toolChoiceCannotForceTheStrippedWebSearchTool() {
        // The tool is gone from the model's view, so a forced choice naming it
        // could never be satisfied; it must fail loudly rather than silently.
        let rejected: [String] = [
            #"{"model":"m","input":"x","tool_choice":"web_search","tools":[{"type":"web_search","external_web_access":false}]}"#,
            #"{"model":"m","input":"x","tool_choice":{"type":"web_search"},"tools":[{"type":"web_search","external_web_access":false}]}"#,
        ]
        for payload in rejected {
            #expect(throws: RejectionReason.self, "must reject: \(payload)") {
                _ = try ResponsesHandler.parse(bytes(payload))
            }
        }
    }

    @Test func anythingRicherThanTheCodexEnvelopeStillFailsClosed() {
        let rejected: [String] = [
            // Reasoning we cannot perform must not look honoured.
            #"{"model":"m","input":"x","reasoning":{"effort":"high"}}"#,
            #"{"model":"m","input":"x","reasoning":{"summary":"detailed"}}"#,
            #"{"model":"m","input":"x","reasoning":{"summary":7}}"#,
            #"{"model":"m","input":"x","reasoning":{"summary":"auto","effort":"low"}}"#,
            // Only the one include entry Codex sends; nothing else exists locally.
            #"{"model":"m","input":"x","include":["message.output_text.logprobs"]}"#,
            #"{"model":"m","input":"x","include":["reasoning.encrypted_content","file_search_call.results"]}"#,
            // Envelope fields are still strictly typed.
            #"{"model":"m","input":"x","client_metadata":"session-1"}"#,
            #"{"model":"m","input":"x","prompt_cache_key":42}"#,
            #"{"model":"m","input":"x","reasoning":"high"}"#,
            #"{"model":"m","input":"x","include":"reasoning.encrypted_content"}"#,
            // Hosted tool types Codex offers by default remain refused.
            #"{"model":"m","input":"x","tools":[{"type":"web_search"}]}"#,
        ]
        for payload in rejected {
            #expect(throws: RejectionReason.self, "must reject: \(payload)") {
                _ = try ResponsesHandler.parse(bytes(payload))
            }
        }
    }

    @Test func parallelToolCallsFalseIsHonouredAsSequentialEmission() async throws {
        // We never execute tools; `false` asks that calls not be issued at once,
        // and the stream emits each function item's lifecycle in order.
        let first = ToolCall(id: "call-a", name: "shell", arguments: ["cmd": .string("ls")])
        let second = ToolCall(id: "call-b", name: "shell", arguments: ["cmd": .string("pwd")])
        let backend = ResponsesStubBackend(
            response: .init(
                output: "", toolCalls: [first, second],
                usage: .init(promptTokens: 1, completionTokens: 1), finishReason: .toolCall
            ),
            events: [.toolCall(first), .toolCall(second)]
        )
        let events = try await runStream(backend: backend, body: #"""
        {"model":"m","input":"x","stream":true,"parallel_tool_calls":false,
         "tools":[{"type":"function","name":"shell","parameters":{"type":"object"}}]}
        """#)
        let indexes = events
            .filter { $0["type"] as? String == "response.output_item.added" }
            .compactMap { $0["output_index"] as? Int }
        #expect(indexes == [0, 1])
        let terminal = try #require(events.last?["response"] as? [String: Any])
        let output = try #require(terminal["output"] as? [[String: Any]])
        #expect(output.count == 2)
        #expect(output.compactMap { $0["call_id"] as? String } == ["call-a", "call-b"])
    }

    // MARK: Non-streaming response object

    @Test func nonStreamingBuildsResponseObjectFromInjectedCore() async throws {
        let backend = ResponsesStubBackend(response: .init(
            output: "the answer",
            toolCalls: [.init(id: "call-1", name: "lookup", arguments: ["k": .string("v")])],
            usage: .init(promptTokens: 5, completionTokens: 3),
            finishReason: .toolCall
        ))
        let json = try await run(backend: backend, body: #"""
        {"model":"org/m","input":"hi","stream":false}
        """#)
        #expect(json.status == .ok)
        #expect(json.body["object"] as? String == "response")
        #expect(json.body["status"] as? String == "completed")
        let output = try #require(json.body["output"] as? [[String: Any]])
        #expect(output.contains { $0["type"] as? String == "message" })
        #expect(output.contains { $0["type"] as? String == "function_call" })
        let usage = try #require(json.body["usage"] as? [String: Any])
        #expect(usage["input_tokens"] as? Int == 5)
        #expect(usage["output_tokens"] as? Int == 3)
        #expect(usage["total_tokens"] as? Int == 8)
        #expect(await backend.requests.map(\.model) == [ModelID("org/m")])
    }

    @Test func lengthFinishReasonYieldsIncompleteStatus() async throws {
        let backend = ResponsesStubBackend(response: .init(
            output: "partial", toolCalls: [],
            usage: .init(promptTokens: 1, completionTokens: 1), finishReason: .length
        ))
        let json = try await run(backend: backend, body: #"{"model":"m","input":"hi"}"#)
        #expect(json.body["status"] as? String == "incomplete")
        // The truncated message item is incomplete too, not just the response.
        let output = try #require(json.body["output"] as? [[String: Any]])
        #expect(output.first?["status"] as? String == "incomplete")
    }

    @Test func silentFinalToolCallStillGetsLifecycleAndTerminalPresence() async throws {
        // Core returns a tool call in the final result without ever streaming
        // a .toolCall event: the lifecycle must be synthesized and the terminal
        // output must contain the call.
        let call = ToolCall(id: "call-silent", name: "lookup", arguments: ["k": .string("v")])
        let backend = ResponsesStubBackend(
            response: .init(
                output: "", toolCalls: [call],
                usage: .init(promptTokens: 1, completionTokens: 1), finishReason: .toolCall
            ),
            events: []
        )
        let events = try await runStream(backend: backend, body: #"""
        {"model":"m","input":"x","stream":true,
         "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}
        """#)
        let types = events.compactMap { $0["type"] as? String }
        #expect(types.contains("response.output_item.added"))
        #expect(types.contains("response.function_call_arguments.done"))
        let terminal = try #require(events.last?["response"] as? [String: Any])
        let output = try #require(terminal["output"] as? [[String: Any]])
        #expect(output.count == 1)
        #expect(output[0]["type"] as? String == "function_call")
        #expect(output[0]["call_id"] as? String == "call-silent")
        // And an already-announced call is not duplicated.
        let announced = ResponsesStubBackend(
            response: .init(
                output: "", toolCalls: [call],
                usage: .init(promptTokens: 1, completionTokens: 1), finishReason: .toolCall
            ),
            events: [.toolCall(call)]
        )
        let dedupedEvents = try await runStream(backend: announced, body: #"""
        {"model":"m","input":"x","stream":true,
         "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}
        """#)
        let dedupedTerminal = try #require(dedupedEvents.last?["response"] as? [String: Any])
        let dedupedOutput = try #require(dedupedTerminal["output"] as? [[String: Any]])
        #expect(dedupedOutput.count == 1)
    }

    // MARK: Streaming typed SSE

    @Test func streamingEmitsMonotonicSequenceAndTypedEvents() async throws {
        let backend = ResponsesStubBackend(
            response: .init(
                output: "hello world", toolCalls: [],
                usage: .init(promptTokens: 2, completionTokens: 2), finishReason: .completed
            ),
            events: [.textDelta("hello "), .textDelta("world")]
        )
        let events = try await runStream(backend: backend, body: #"""
        {"model":"m","input":"hi","stream":true}
        """#)
        // sequence_number strictly increasing from 0.
        let sequences = events.compactMap { $0["sequence_number"] as? Int }
        #expect(sequences == Array(0 ..< sequences.count))
        let types = events.compactMap { $0["type"] as? String }
        #expect(types.first == "response.created")
        #expect(types.contains("response.in_progress"))
        #expect(types.contains("response.output_item.added"))
        #expect(types.contains("response.output_text.delta"))
        #expect(types.contains("response.output_text.done"))
        #expect(types.last == "response.completed")
        // Terminal response reuses the streamed item id at the streamed index.
        let streamedID = try #require(
            events.first { $0["type"] as? String == "response.output_text.delta" }?["item_id"] as? String
        )
        let terminal = try #require(events.last { $0["type"] as? String == "response.completed" })
        let response = try #require(terminal["response"] as? [String: Any])
        let output = try #require(response["output"] as? [[String: Any]])
        #expect(output.count == 1)
        #expect(output[0]["id"] as? String == streamedID)
        let addedIndex = try #require(
            events.first { $0["type"] as? String == "response.output_item.added" }?["output_index"] as? Int
        )
        #expect(addedIndex == 0)
    }

    @Test func functionOnlyStreamStartsAtIndexZeroAndClosesWithSameIdentity() async throws {
        let call = ToolCall(id: "call-1", name: "lookup", arguments: ["k": .string("v")])
        let backend = ResponsesStubBackend(
            response: .init(
                output: "", toolCalls: [call],
                usage: .init(promptTokens: 2, completionTokens: 2), finishReason: .toolCall
            ),
            events: [.toolCall(call)]
        )
        let events = try await runStream(backend: backend, body: #"""
        {"model":"m","input":"hi","stream":true,
         "tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}
        """#)
        let added = try #require(events.first { $0["type"] as? String == "response.output_item.added" })
        // No text item precedes it: the function item owns output_index 0.
        #expect(added["output_index"] as? Int == 0)
        let streamedID = try #require((added["item"] as? [String: Any])?["id"] as? String)
        for event in events where event["type"] as? String == "response.function_call_arguments.done" {
            #expect(event["item_id"] as? String == streamedID)
            #expect(event["output_index"] as? Int == 0)
        }
        let terminal = try #require(events.last { $0["type"] as? String == "response.completed" })
        let output = try #require((terminal["response"] as? [String: Any])?["output"] as? [[String: Any]])
        #expect(output.count == 1)
        #expect(output[0]["id"] as? String == streamedID)
        #expect(output[0]["type"] as? String == "function_call")
        #expect(output[0]["call_id"] as? String == "call-1")
    }

    @Test func responseObjectCarriesRequiredSchemaFieldsAndUsageDetails() async throws {
        let backend = ResponsesStubBackend(response: .init(
            output: "hi", toolCalls: [],
            usage: .init(promptTokens: 3, completionTokens: 4), finishReason: .completed
        ))
        let json = try await run(backend: backend, body: #"""
        {"model":"m","input":"x","tool_choice":"auto",
         "tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]}
        """#)
        #expect(json.body["tool_choice"] as? String == "auto")
        let tools = try #require(json.body["tools"] as? [[String: Any]])
        #expect(tools.count == 1)
        #expect(tools[0]["name"] as? String == "f")
        let usage = try #require(json.body["usage"] as? [String: Any])
        #expect((usage["input_tokens_details"] as? [String: Any])?["cached_tokens"] as? Int == 0)
        #expect((usage["output_tokens_details"] as? [String: Any])?["reasoning_tokens"] as? Int == 0)
        #expect(json.body["error"] is NSNull)
    }

    @Test func errorEnvelopeUsesStringCode() async throws {
        let backend = ResponsesStubBackend(response: .init(
            output: "", toolCalls: [],
            usage: .init(promptTokens: 0, completionTokens: 0), finishReason: .completed
        ))
        let json = try await run(backend: backend, body: #"{"model":"m","input":"x","store":true}"#)
        #expect(json.status == .badRequest)
        let error = try #require(json.body["error"] as? [String: Any])
        #expect(error["code"] is String)
        #expect(error["type"] as? String == "invalid_request_error")
    }

    @Test func streamCarriesLogprobsAndEndsWithTypedTerminalNotDone() async throws {
        let backend = ResponsesStubBackend(
            response: .init(
                output: "hello", toolCalls: [],
                usage: .init(promptTokens: 1, completionTokens: 1), finishReason: .completed
            ),
            events: [.textDelta("hello")]
        )
        let (_, raw) = try await drive(backend: backend, body: #"{"model":"m","input":"x","stream":true}"#)
        #expect(!raw.contains("data: [DONE]"))
        let events = parseSSE(raw)
        let delta = try #require(events.first { $0["type"] as? String == "response.output_text.delta" })
        #expect(delta["logprobs"] as? [Any] != nil)
        let done = try #require(events.first { $0["type"] as? String == "response.output_text.done" })
        #expect(done["logprobs"] as? [Any] != nil)
        #expect(events.last?["type"] as? String == "response.completed")
    }

    @Test func lengthStreamMarksItemAndResponseIncomplete() async throws {
        let backend = ResponsesStubBackend(
            response: .init(
                output: "partial", toolCalls: [],
                usage: .init(promptTokens: 1, completionTokens: 1), finishReason: .length
            ),
            events: [.textDelta("partial")]
        )
        let events = try await runStream(backend: backend, body: #"{"model":"m","input":"x","stream":true}"#)
        #expect(events.last?["type"] as? String == "response.incomplete")
        let itemDone = try #require(events.last { $0["type"] as? String == "response.output_item.done" })
        #expect((itemDone["item"] as? [String: Any])?["status"] as? String == "incomplete")
        let terminal = try #require(events.last?["response"] as? [String: Any])
        let output = try #require(terminal["output"] as? [[String: Any]])
        #expect(output.first?["status"] as? String == "incomplete")
    }

    @Test func finalTextWithoutDeltasSynthesizesLifecycleAndFinalTextIsAuthoritative() async throws {
        // No deltas at all: the lifecycle must still be emitted.
        let silent = ResponsesStubBackend(
            response: .init(
                output: "final only", toolCalls: [],
                usage: .init(promptTokens: 1, completionTokens: 1), finishReason: .completed
            ),
            events: []
        )
        let quietEvents = try await runStream(backend: silent, body: #"{"model":"m","input":"x","stream":true}"#)
        let quietTypes = quietEvents.compactMap { $0["type"] as? String }
        #expect(quietTypes.contains("response.output_item.added"))
        #expect(quietTypes.contains("response.content_part.added"))
        #expect(quietTypes.contains("response.output_text.done"))
        let quietDone = try #require(quietEvents.first { $0["type"] as? String == "response.output_text.done" })
        #expect(quietDone["text"] as? String == "final only")

        // Deltas diverge from the final output: Core's final output wins everywhere.
        let divergent = ResponsesStubBackend(
            response: .init(
                output: "hello world", toolCalls: [],
                usage: .init(promptTokens: 1, completionTokens: 1), finishReason: .completed
            ),
            events: [.textDelta("hel")]
        )
        let events = try await runStream(backend: divergent, body: #"{"model":"m","input":"x","stream":true}"#)
        let done = try #require(events.first { $0["type"] as? String == "response.output_text.done" })
        #expect(done["text"] as? String == "hello world")
        let terminal = try #require(events.last?["response"] as? [String: Any])
        let output = try #require(terminal["output"] as? [[String: Any]])
        let content = try #require(output.first?["content"] as? [[String: Any]])
        #expect(content.first?["text"] as? String == "hello world")
    }

    // MARK: - Harness

    private func bytes(_ json: String) -> ByteBuffer { ByteBuffer(bytes: Data(json.utf8)) }

    private func run(
        backend: ResponsesStubBackend,
        body: String
    ) async throws -> (status: HTTPResponseStatus, body: [String: Any]) {
        let (status, raw) = try await drive(backend: backend, body: body)
        let object = try #require(JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any])
        return (status, object)
    }

    private func runStream(backend: ResponsesStubBackend, body: String) async throws -> [[String: Any]] {
        let (_, raw) = try await drive(backend: backend, body: body)
        return parseSSE(raw)
    }

    private func parseSSE(_ raw: String) -> [[String: Any]] {
        var events: [[String: Any]] = []
        for line in raw.split(separator: "\n") {
            guard line.hasPrefix("data: ") else { continue }

            let payload = line.dropFirst("data: ".count)
            if payload == "[DONE]" { continue }
            if let object = try? JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any] {
                events.append(object)
            }
        }
        return events
    }

    private func drive(
        backend: ResponsesStubBackend,
        body: String
    ) async throws -> (status: HTTPResponseStatus, body: String) {
        let engine = SwamaEngine(backend: backend)
        let channel = NIOAsyncTestingChannel()
        try await channel.connect(to: .init(ipAddress: "127.0.0.1", port: 28100))
        await channel.testingEventLoop.run()
        let done = ResponsesAsyncFlag()
        let buffer = ByteBuffer(bytes: Data(body.utf8))
        let task = Task {
            await ResponsesHandler.handle(
                requestHead: .init(version: .http1_1, method: .POST, uri: "/v1/responses"),
                body: buffer,
                channel: channel,
                engine: engine
            )
            await done.set()
        }
        for _ in 0 ..< 2000 {
            await channel.testingEventLoop.run()
            if await done.value {
                break
            }
            await Task.yield()
        }
        await task.value
        await channel.testingEventLoop.run()

        var status: HTTPResponseStatus?
        var out = ByteBuffer()
        while let part = try await channel.readOutbound(as: HTTPServerResponsePart.self) {
            switch part {
            case let .head(head): status = head.status
            case var .body(.byteBuffer(chunk)): out.writeBuffer(&chunk)
            default: break
            }
        }
        return (status ?? .internalServerError, String(decoding: out.readableBytesView, as: UTF8.self))
    }
}

// MARK: - ResponsesStubBackend

private actor ResponsesStubBackend: SwamaEngineBackend {
    init(response: GenerationResponse, events: [GenerationEvent] = []) {
        self.response = response
        self.events = events
    }

    func generate(
        _ request: GenerationRequest,
        onEvent: (@Sendable (GenerationEvent) async throws -> Void)?
    ) async throws -> GenerationResponse {
        requests.append(request)
        for event in events {
            try await onEvent?(event)
        }
        return response
    }

    func embed(_: EmbeddingRequest) async throws -> SwamaCore.EmbeddingResponse {
        .init(embeddings: [], usage: .init(promptTokens: 0, completionTokens: 0))
    }

    func models() async throws -> [SwamaCore.ModelInfo] { [] }
    func fetch(_ model: ModelID) async throws -> ModelID { model }
    func remove(_: ModelID) async throws {}
    func clearCache(for _: ModelID) async {}
    func clearCache() async {}

    private let response: GenerationResponse
    private let events: [GenerationEvent]
    private(set) var requests: [GenerationRequest] = []
}

// MARK: - ResponsesAsyncFlag

private actor ResponsesAsyncFlag {
    func set() { value = true }
    private(set) var value = false
}

import Foundation
import NIOCore
import NIOHTTP1
import SwamaCore

// MARK: - RejectionReason

/// Every input this honest subset refuses. Each maps to an explicit `400` so an
/// unsupported feature is never silently downgraded to Chat-Completions behaviour.
enum RejectionReason: Error, LocalizedError, Equatable {
    case malformed(String)
    case missingModel
    case emptyInput
    case unsupportedField(String)
    case unsupportedToolType(String)
    case unforceableToolChoice(String)
    case core(SwamaError)

    var errorDescription: String? {
        switch self {
        case let .malformed(detail):
            "Malformed request: \(detail)"
        case .missingModel:
            "`model` is required"
        case .emptyInput:
            "`input` must be a non-empty string or a non-empty array of input items"
        case let .unsupportedField(field):
            "`\(field)` is not supported by this server; it is rejected rather than silently ignored"
        case let .unsupportedToolType(type):
            "tool type `\(type)` is not supported; only custom `function` tools run locally"
        case let .unforceableToolChoice(choice):
            "tool_choice `\(choice)` cannot be enforced by the local runtime and is rejected"
        case let .core(error):
            error.message
        }
    }

    var wireType: String {
        switch self {
        case .core: "invalid_request_error"
        default: "invalid_request_error"
        }
    }

    /// Bounded string error code (official `code` is string-or-null).
    var wireCode: String {
        switch self {
        case .malformed: "invalid_value"
        case .missingModel: "missing_required_parameter"
        case .emptyInput: "missing_required_parameter"
        case .unsupportedField: "unsupported_parameter"
        case .unsupportedToolType: "unsupported_parameter"
        case .unforceableToolChoice: "unsupported_parameter"
        case let .core(error): error.code.rawValue
        }
    }
}

// MARK: - Parsing

extension ResponsesHandler {
    struct ParsedRequest: Sendable {
        var request: GenerationRequest
        var stream: Bool
        /// Effective tool_choice ("auto" or "none"), echoed into Response objects:
        /// the official schema requires `tool_choice` and `tools` on every response.
        var toolChoice: String
        /// Effective `parallel_tool_calls`, echoed into every Response object.
        /// Absent means `false`: this server never issues tool calls
        /// concurrently, so reporting the hosted default of `true` would
        /// describe behaviour it does not have. An explicit `true` is echoed
        /// back because sequential emission also satisfies "may be parallel".
        var parallelToolCalls: Bool
        var maxOutputTokens: Int?
        var temperature: Float
        var topP: Float
        var instructions: String?
    }

    static func parse(_ body: ByteBuffer?) throws -> ParsedRequest {
        guard let body, body.readableBytes > 0 else {
            throw RejectionReason.malformed("empty body")
        }

        let data = Data(body.readableBytesView)
        let root: [String: Any]
        do {
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                throw RejectionReason.malformed("top-level JSON must be an object")
            }

            root = object
        }
        catch let rejection as RejectionReason {
            throw rejection
        }
        catch {
            throw RejectionReason.malformed(error.localizedDescription)
        }

        guard let model = root["model"] as? String, !model.isEmpty else {
            throw RejectionReason.missingModel
        }

        try rejectUnsupported(root)

        let messages = try mergingAdjacentSystemTurns(buildMessages(root))
        guard !messages.isEmpty else {
            throw RejectionReason.emptyInput
        }

        let options = try buildOptions(root)
        var tools = try buildTools(root)
        // `tool_choice: "none"` must actually prevent tool use, not merely parse:
        // the local runtime enforces it by never offering the tools to the model.
        if (root["tool_choice"] as? String) == "none" {
            tools = []
        }
        let stream = try requireBool(root, "stream") ?? false

        return ParsedRequest(
            request: GenerationRequest(
                model: ModelID(model),
                messages: messages,
                options: options,
                tools: tools
            ),
            stream: stream,
            toolChoice: (root["tool_choice"] as? String) ?? "auto",
            parallelToolCalls: (try? requireBool(root, "parallel_tool_calls")).flatMap { $0 } ?? false,
            maxOutputTokens: options.maxTokens,
            temperature: options.temperature,
            topP: options.topP,
            instructions: root["instructions"] as? String
        )
    }

    /// The complete top-level request surface this server implements. Anything
    /// else — official-but-unimplemented (`metadata`, `context_management`,
    /// `stream_options`, ...) or plain unknown — is a hard 400: an exact
    /// allowlist, not an enumerated blocklist.
    private static let supportedTopLevelFields: Set<String> = [
        "model",
        "input",
        "instructions",
        "stream",
        "temperature",
        "top_p",
        "max_output_tokens",
        "tools",
        "tool_choice",
        "text",
        "truncation",
        "store",
        "background",
        "parallel_tool_calls",
        // Codex CLI interop envelope: accepted with strict typing and an
        // explicit local meaning (see `acceptCodexEnvelope`), never silently
        // honoured. Anything richer than this fixed envelope still fails closed.
        "client_metadata",
        "prompt_cache_key",
        "reasoning",
        "include",
    ]

    private static func rejectUnsupported(_ root: [String: Any]) throws {
        for key in root.keys.sorted() where !supportedTopLevelFields.contains(key) {
            throw RejectionReason.unsupportedField(key)
        }
        if let store = try requireBool(root, "store"), store {
            throw RejectionReason.unsupportedField("store:true")
        }
        if let background = try requireBool(root, "background"), background {
            throw RejectionReason.unsupportedField("background:true")
        }
        // `parallel_tool_calls` asks how tool calls may be issued. This server
        // never executes tools itself and always emits function-call items one
        // after another, which is exactly what `false` asks for; `true` is never
        // exceeded either. So only its type is enforced.
        _ = try requireBool(root, "parallel_tool_calls")
        try acceptCodexEnvelope(root)
        // Structured output. Plain `text.format.type == "text"` is the default
        // behaviour, not Structured Outputs, and must remain accepted.
        if let text = root["text"] {
            guard let object = text as? [String: Any] else {
                throw RejectionReason.malformed("`text` must be a JSON object")
            }

            for key in object.keys.sorted() where key != "format" {
                throw RejectionReason.unsupportedField("text.\(key)")
            }
            if let format = object["format"] {
                guard let formatObject = format as? [String: Any],
                      let formatType = formatObject["type"] as? String
                else {
                    throw RejectionReason.malformed("`text.format` must be an object with a string `type`")
                }

                // Accepting the object but ignoring its other keys would let a
                // schema request look honoured; plain text carries no options.
                try requireKeys(formatObject, within: ["type"], of: "text.format")
                if formatType != "text" {
                    throw RejectionReason.unsupportedField("text.format")
                }
            }
        }
        // Truncation is a closed enum; only the local default is supported.
        if let truncation = root["truncation"] {
            guard let mode = truncation as? String else {
                throw RejectionReason.malformed("`truncation` must be a JSON string")
            }

            switch mode {
            case "disabled": break
            case "auto": throw RejectionReason.unsupportedField("truncation:auto")
            default: throw RejectionReason.malformed("`truncation` must be \"disabled\" or \"auto\"")
            }
        }
        if let instructions = root["instructions"], !(instructions is String) {
            throw RejectionReason.malformed("`instructions` must be a JSON string")
        }
        // tool_choice: only "auto"/"none" (and the default) are honourable locally.
        if let choice = root["tool_choice"] {
            try rejectUnforceableToolChoice(choice)
        }
    }

    /// The fixed, non-reasoning request envelope Codex CLI always sends. Each
    /// field is strictly typed and then given an explicit local meaning; none is
    /// honoured silently:
    ///
    /// - `client_metadata`: client-side tracing only, carries no request
    ///   semantics, so it is validated and ignored.
    /// - `prompt_cache_key`: names OpenAI's server-side prompt cache. This
    ///   server keeps its own local cache keyed by prompt content, so the key is
    ///   accepted and superseded rather than obeyed.
    /// - `reasoning`: the empty object, or exactly `summary: "auto"` (what Codex
    ///   sends). A local model produces no reasoning items, so `effort` and any
    ///   other summary style could not be honoured and are refused.
    /// - `include`: only the reasoning-content entry Codex always sends; with no
    ///   reasoning items there is nothing to return. Any other entry is refused.
    private static func acceptCodexEnvelope(_ root: [String: Any]) throws {
        if let metadata = root["client_metadata"], !(metadata is [String: Any]) {
            throw RejectionReason.malformed("`client_metadata` must be a JSON object")
        }
        if let key = root["prompt_cache_key"], !(key is String) {
            throw RejectionReason.malformed("`prompt_cache_key` must be a JSON string")
        }
        if let reasoning = root["reasoning"] {
            guard let object = reasoning as? [String: Any] else {
                throw RejectionReason.malformed("`reasoning` must be a JSON object")
            }
            // Empty, or only `summary: "auto"` — which is what Codex actually
            // sends. `auto` leaves the choice to the server, and a local model
            // emits no reasoning items, so producing none satisfies it. An
            // explicit `effort`, or a summary style we cannot produce, is refused.
            try requireKeys(object, within: ["summary"], of: "reasoning")
            if let summary = object["summary"] {
                guard let mode = summary as? String else {
                    throw RejectionReason.malformed("`reasoning.summary` must be a JSON string")
                }

                guard mode == "auto" else {
                    throw RejectionReason.unsupportedField("reasoning.summary")
                }
            }
        }
        if let include = root["include"] {
            guard let items = include as? [Any] else {
                throw RejectionReason.malformed("`include` must be a JSON array")
            }

            for item in items {
                guard let entry = item as? String, entry == "reasoning.encrypted_content" else {
                    throw RejectionReason.unsupportedField("include")
                }
            }
        }
    }

    private static func rejectUnforceableToolChoice(_ choice: Any) throws {
        if let string = choice as? String {
            switch string {
            case "auto",
                 "none": return
            case "required": throw RejectionReason.unforceableToolChoice("required")
            default: throw RejectionReason.unforceableToolChoice(string)
            }
        }
        if let object = choice as? [String: Any] {
            let type = (object["type"] as? String) ?? "object"
            throw RejectionReason.unforceableToolChoice(type)
        }
        throw RejectionReason.unforceableToolChoice("unknown")
    }

    // MARK: input -> [Message]

    /// The local chat template cannot consume two adjacent `system` turns — the
    /// backend fails outright (reproducible on `/v1/chat/completions` too, so
    /// this is a runtime limitation rather than something this adapter invents).
    ///
    /// The Responses API produces exactly that shape for every real client:
    /// top-level `instructions` lowers to a system turn, and `developer` role
    /// items lower to system turns as well, so Codex CLI always yields two or
    /// more in a row. Concatenating their parts is a faithful lowering — layered
    /// guidance in order, nothing dropped or reordered — and is what makes the
    /// endpoint usable at all. The underlying runtime limitation is tracked
    /// separately; it also affects the chat endpoint, which this cannot fix.
    static func mergingAdjacentSystemTurns(_ messages: [Message]) -> [Message] {
        var merged: [Message] = []
        for message in messages {
            if message.role == .system,
               let previous = merged.last,
               previous.role == .system,
               previous.toolCalls.isEmpty, message.toolCalls.isEmpty
            {
                merged[merged.count - 1] = Message(
                    role: .system,
                    content: previous.content + message.content
                )
                continue
            }

            merged.append(message)
        }

        return merged
    }

    private static func buildMessages(_ root: [String: Any]) throws -> [Message] {
        var messages: [Message] = []
        if let instructions = root["instructions"] as? String, !instructions.isEmpty {
            messages.append(Message(role: .system, text: instructions))
        }

        guard let input = root["input"] else { throw RejectionReason.emptyInput }

        if let text = input as? String {
            guard !text.isEmpty else {
                throw RejectionReason.emptyInput
            }

            messages.append(Message(role: .user, text: text))
            return messages
        }

        guard let items = input as? [Any], !items.isEmpty else {
            throw RejectionReason.emptyInput
        }

        for item in items {
            guard let object = item as? [String: Any] else {
                throw RejectionReason.malformed("input items must be objects")
            }

            if let rawType = object["type"], !(rawType is String) {
                throw RejectionReason.malformed("input item `type` must be a JSON string")
            }
            let type = (object["type"] as? String) ?? "message"
            switch type {
            case "message":
                try messages.append(buildMessageItem(object))
            case "function_call":
                try messages.append(buildFunctionCallItem(object))
            case "function_call_output":
                try messages.append(buildFunctionCallOutputItem(object))
            default:
                throw RejectionReason.unsupportedField("input item type `\(type)`")
            }
        }
        return messages
    }

    private static func buildMessageItem(_ object: [String: Any]) throws -> Message {
        try requireKeys(object, within: ["type", "role", "content", "id", "status"], of: "message item")
        try requireItemMetadata(object, of: "message item", statuses: ["in_progress", "completed", "incomplete"])
        if let rawRole = object["role"], !(rawRole is String) {
            throw RejectionReason.malformed("message item `role` must be a JSON string")
        }
        let role = try mapRole(object["role"] as? String)
        guard let rawContent = object["content"] else {
            throw RejectionReason.malformed("message item requires `content`")
        }

        if let text = rawContent as? String {
            return Message(role: role, text: text)
        }
        guard let parts = rawContent as? [Any] else {
            throw RejectionReason.malformed("message content must be a string or array")
        }

        var content: [ContentPart] = []
        for part in parts {
            guard let partObject = part as? [String: Any],
                  let partType = partObject["type"] as? String
            else {
                throw RejectionReason.malformed("content parts must be typed objects")
            }

            switch partType {
            case "input_text",
                 "output_text",
                 "text":
                try requireKeys(
                    partObject,
                    within: ["type", "text", "annotations", "logprobs"],
                    of: "text content part"
                )
                // Echoed by clients replaying our own output. We produce
                // neither, so only the empty forms are meaningful; anything
                // else would be accepted and then silently erased.
                for key in ["annotations", "logprobs"] {
                    guard let value = partObject[key] else {
                        continue
                    }
                    guard let entries = value as? [Any] else {
                        throw RejectionReason.malformed("text content part `\(key)` must be a JSON array")
                    }
                    guard entries.isEmpty else {
                        throw RejectionReason.unsupportedField("text content part `\(key)`")
                    }
                }
                guard let text = partObject["text"] as? String else {
                    throw RejectionReason.malformed("text content part requires a string `text`")
                }

                content.append(.text(text))

            case "input_image":
                try requireKeys(partObject, within: ["type", "image_url", "detail"], of: "input_image part")
                // The local runtime has one image path; a non-default detail
                // request would be silently downgraded, so it is refused.
                if let detail = partObject["detail"] {
                    guard let mode = detail as? String, mode == "auto" else {
                        throw RejectionReason.unsupportedField("input_image.detail")
                    }
                }
                guard let url = partObject["image_url"] as? String else {
                    throw RejectionReason.malformed("input_image requires a valid image_url")
                }
                // Same validator as /v1/chat/completions - see ImageInputParser.
                // Responses previously checked only the scheme and so accepted
                // malformed inputs that Chat already refused.
                guard let part = ImageInputParser.contentPart(url) else {
                    throw RejectionReason.unsupportedField("input_image.image_url")
                }

                content.append(part)

            default:
                throw RejectionReason.unsupportedField("content part `\(partType)`")
            }
        }
        guard !content.isEmpty else {
            throw RejectionReason.malformed("empty message content")
        }

        return Message(role: role, content: content)
    }

    /// Item envelope fields we accept but do not act on are still a CLOSED
    /// schema. Typing them was not enough: an arbitrary `status: "banana"` or an
    /// empty `id` was accepted and then erased, which is indistinguishable from
    /// support - the same fail-closed rule already applied to `text.format`.
    private static func requireItemMetadata(
        _ object: [String: Any],
        of surface: String,
        statuses: Set<String>
    ) throws {
        if let value = object["id"] {
            guard let id = value as? String else {
                throw RejectionReason.malformed("\(surface) `id` must be a JSON string")
            }
            guard !id.isEmpty else {
                throw RejectionReason.malformed("\(surface) `id` must not be empty")
            }
        }
        if let value = object["status"] {
            guard let status = value as? String else {
                throw RejectionReason.malformed("\(surface) `status` must be a JSON string")
            }
            guard statuses.contains(status) else {
                throw RejectionReason.unsupportedField("\(surface) status `\(status)`")
            }
        }
    }

    /// Structural allowlist for one object: any key outside the schema is a 400.
    private static func requireKeys(
        _ object: [String: Any],
        within allowed: Set<String>,
        of surface: String
    ) throws {
        for key in object.keys.sorted() where !allowed.contains(key) {
            throw RejectionReason.unsupportedField("\(surface) `\(key)`")
        }
    }

    /// A prior model turn's tool call, replayed by the client as conversation
    /// context for the follow-up turn of the local tool loop.
    private static func buildFunctionCallItem(_ object: [String: Any]) throws -> Message {
        try requireKeys(
            object,
            within: ["type", "call_id", "name", "arguments", "id", "status"],
            of: "function_call item"
        )
        try requireItemMetadata(object, of: "function_call item", statuses: ["in_progress", "completed", "incomplete"])
        guard let callID = object["call_id"] as? String, !callID.isEmpty,
              let name = object["name"] as? String, !name.isEmpty
        else {
            throw RejectionReason.malformed("function_call item requires `call_id` and `name`")
        }

        // `arguments` is a required JSON-object string in the official item
        // type; missing or empty must not be forged into `{}`.
        guard let text = object["arguments"] as? String,
              let parsed = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
        else {
            throw RejectionReason.malformed("function_call requires `arguments` as a JSON object string")
        }

        return Message(
            role: .assistant,
            content: [],
            toolCalls: [ToolCall(id: callID, name: name, arguments: parsed.mapValues(jsonValue))]
        )
    }

    /// The client-executed tool result for a prior `function_call`.
    private static func buildFunctionCallOutputItem(_ object: [String: Any]) throws -> Message {
        try requireKeys(object, within: ["type", "call_id", "output", "id", "status"], of: "function_call_output item")
        try requireItemMetadata(object, of: "function_call_output item", statuses: ["in_progress", "completed", "incomplete"])
        guard let callID = object["call_id"] as? String, !callID.isEmpty else {
            throw RejectionReason.malformed("function_call_output item requires `call_id`")
        }
        guard let output = object["output"] as? String else {
            throw RejectionReason.malformed("function_call_output item requires a string `output`")
        }

        return Message(role: .tool, content: [.text(output)], toolCallID: callID)
    }

    private static func mapRole(_ raw: String?) throws -> Message.Role {
        switch raw ?? "user" {
        case "user": .user
        case "developer",
             "system": .system
        case "assistant": .assistant
        case "tool": .tool
        case let other: throw RejectionReason.malformed("unsupported role `\(other)`")
        }
    }

    // MARK: Strict field typing

    // JSONSerialization coerces aggressively (bools are NSNumbers, "1.5" stays a
    // string, ...). A known field of the wrong JSON type must be a 400, never a
    // silent default or truncation — that would change request meaning.

    private static func requireBool(_ root: [String: Any], _ key: String) throws -> Bool? {
        guard let value = root[key] else {
            return nil
        }
        guard let number = value as? NSNumber, CFGetTypeID(number as CFTypeRef) == CFBooleanGetTypeID() else {
            throw RejectionReason.malformed("`\(key)` must be a JSON boolean")
        }

        return number.boolValue
    }

    private static func requireNumber(_ root: [String: Any], _ key: String) throws -> Float? {
        guard let value = root[key] else {
            return nil
        }
        guard let number = value as? NSNumber, CFGetTypeID(number as CFTypeRef) != CFBooleanGetTypeID() else {
            throw RejectionReason.malformed("`\(key)` must be a JSON number")
        }

        return number.floatValue
    }

    private static func requireInt(_ root: [String: Any], _ key: String) throws -> Int? {
        guard let value = root[key] else {
            return nil
        }

        // `1e100` is integral but saturates through `intValue`; require a finite
        // value that actually fits in Int before converting.
        guard let number = value as? NSNumber,
              CFGetTypeID(number as CFTypeRef) != CFBooleanGetTypeID()
        else {
            throw RejectionReason.malformed("`\(key)` must be a JSON integer")
        }

        let double = number.doubleValue
        guard double.isFinite,
              double == double.rounded(),
              double >= -9_007_199_254_740_992, double <= 9_007_199_254_740_992
        else {
            throw RejectionReason.malformed("`\(key)` must be an integer within the supported range")
        }

        return number.intValue
    }

    // MARK: sampling -> GenerationOptions

    private static func buildOptions(_ root: [String: Any]) throws -> GenerationOptions {
        try GenerationOptions(
            maxTokens: requireInt(root, "max_output_tokens"),
            temperature: requireNumber(root, "temperature") ?? 0.6,
            topP: requireNumber(root, "top_p") ?? 1
        )
    }

    // MARK: tools -> [ToolDefinition]

    private static func buildTools(_ root: [String: Any]) throws -> [ToolDefinition] {
        guard let value = root["tools"] else { return [] }
        guard let rawTools = value as? [Any] else {
            throw RejectionReason.malformed("`tools` must be a JSON array")
        }

        var tools: [ToolDefinition] = []
        for raw in rawTools {
            guard let object = raw as? [String: Any] else {
                throw RejectionReason.malformed("tools must be objects")
            }

            // Check the tool type before its key allowlist: a hosted tool kind
            // carries its own fields, and reporting an unexpected key there
            // hides the real reason (the kind itself is unsupported).
            guard let type = object["type"] as? String else {
                throw RejectionReason.malformed("tool requires a string `type`")
            }
            if type == "web_search" {
                // Codex CLI 0.147.0 advertises this unconditionally; no client
                // configuration removes it (`tools.web_search=false` only flips
                // `external_web_access`). Refusing it outright would make the
                // CLI unusable against this server, so the exact offline-only
                // shape is accepted and then DROPPED before the model sees it.
                //
                // This is a deliberate, narrow compatibility DEGRADATION, not
                // support: per OpenAI's reference, `external_web_access:false`
                // does not disable search — it runs web search in an
                // offline/cache-only mode over OpenAI's own index. Swama has no
                // such index, so it performs no search at all and never emits a
                // `web_search_call`. Any request that actually needs search
                // capability is unsupported here.
                try requireKeys(object, within: ["type", "external_web_access"], of: "web_search tool")
                guard let offline = try requireBool(object, "external_web_access"), offline == false else {
                    throw RejectionReason.unsupportedToolType("web_search")
                }

                continue
            }
            guard type == "function" else {
                throw RejectionReason.unsupportedToolType(type)
            }

            try requireKeys(
                object,
                within: ["type", "name", "description", "parameters", "strict"],
                of: "function tool"
            )
            guard let name = object["name"] as? String, !name.isEmpty else {
                throw RejectionReason.malformed("function tool requires a name")
            }

            // The local runtime does not enforce strict JSON-schema adherence,
            // so accepting `strict: true` would promise validation it can't do.
            if let strict = try requireBool(object, "strict"), strict {
                throw RejectionReason.unsupportedField("tools[].strict:true")
            }

            let description: String?
            switch object["description"] {
            case nil,
                 is NSNull:
                description = nil
            case let text as String:
                description = text
            default:
                throw RejectionReason.malformed("tool `description` must be a JSON string")
            }
            if let parameters = object["parameters"], !(parameters is [String: Any]) {
                throw RejectionReason.malformed("tool `parameters` must be a JSON object")
            }
            let parameters = jsonValue(object["parameters"] ?? [String: Any]())
            tools.append(ToolDefinition(name: name, description: description, parameters: parameters))
        }
        return tools
    }
}

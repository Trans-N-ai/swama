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

        let messages = try buildMessages(root)
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
            maxOutputTokens: options.maxTokens,
            temperature: options.temperature,
            topP: options.topP,
            instructions: root["instructions"] as? String
        )
    }

    /// The complete top-level request surface this server implements. Anything
    /// else — official-but-unimplemented (`reasoning`, `metadata`,
    /// `context_management`, `stream_options`, ...) or plain unknown — is a
    /// hard 400: an exact allowlist, not an enumerated blocklist.
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
        if let parallel = try requireBool(root, "parallel_tool_calls"), !parallel {
            throw RejectionReason.unsupportedField("parallel_tool_calls:false")
        }
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
                if let url = partObject["image_url"] as? String, let parsed = URL(string: url) {
                    content.append(.imageURL(parsed))
                }
                else {
                    throw RejectionReason.malformed("input_image requires a valid image_url")
                }

            default:
                throw RejectionReason.unsupportedField("content part `\(partType)`")
            }
        }
        guard !content.isEmpty else {
            throw RejectionReason.malformed("empty message content")
        }

        return Message(role: role, content: content)
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
        guard let number = value as? NSNumber,
              CFGetTypeID(number as CFTypeRef) != CFBooleanGetTypeID(),
              number.doubleValue == number.doubleValue.rounded()
        else {
            throw RejectionReason.malformed("`\(key)` must be a JSON integer")
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

            try requireKeys(
                object,
                within: ["type", "name", "description", "parameters", "strict"],
                of: "tool"
            )
            guard let type = object["type"] as? String else {
                throw RejectionReason.malformed("tool requires a string `type`")
            }
            guard type == "function" else {
                throw RejectionReason.unsupportedToolType(type)
            }
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

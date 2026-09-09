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
}

// MARK: - Parsing

extension ResponsesHandler {
    struct ParsedRequest: Sendable {
        var request: GenerationRequest
        var stream: Bool
    }

    /// Fields that carry server-side state or hosted capabilities this runtime does
    /// not provide. Their mere presence (with a meaningful value) is a hard `400`.
    private static let unsupportedStatefulFields: [String] = [
        "previous_response_id",
        "conversation",
        "prompt",
        "prompt_cache_key",
    ]

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

        let options = buildOptions(root)
        let tools = try buildTools(root)
        let stream = (root["stream"] as? Bool) ?? false

        return ParsedRequest(
            request: GenerationRequest(
                model: ModelID(model),
                messages: messages,
                options: options,
                tools: tools
            ),
            stream: stream
        )
    }

    private static func rejectUnsupported(_ root: [String: Any]) throws {
        // Stateful / hosted-capability fields.
        for field in unsupportedStatefulFields where root[field] != nil {
            throw RejectionReason.unsupportedField(field)
        }
        if let store = root["store"] as? Bool, store {
            throw RejectionReason.unsupportedField("store:true")
        }
        if let background = root["background"] as? Bool, background {
            throw RejectionReason.unsupportedField("background:true")
        }
        // Structured output.
        if root["text"] is [String: Any], (root["text"] as? [String: Any])?["format"] != nil {
            throw RejectionReason.unsupportedField("text.format")
        }
        if root["response_format"] != nil {
            throw RejectionReason.unsupportedField("response_format")
        }
        // Auto truncation cannot be honoured without server-side context management.
        if let truncation = root["truncation"] as? String, truncation == "auto" {
            throw RejectionReason.unsupportedField("truncation:auto")
        }
        // include[] pulls hosted-only artifacts (logprobs, file search results, ...).
        if let include = root["include"] as? [Any], !include.isEmpty {
            throw RejectionReason.unsupportedField("include")
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

            let type = (object["type"] as? String) ?? "message"
            switch type {
            case "message":
                try messages.append(buildMessageItem(object))
            case "function_call",
                 "function_call_output":
                throw RejectionReason.unsupportedField("input item type `\(type)`")
            default:
                throw RejectionReason.unsupportedField("input item type `\(type)`")
            }
        }
        return messages
    }

    private static func buildMessageItem(_ object: [String: Any]) throws -> Message {
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
                if let text = partObject["text"] as? String { content.append(.text(text)) }

            case "input_image":
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

    // MARK: sampling -> GenerationOptions

    private static func buildOptions(_ root: [String: Any]) -> GenerationOptions {
        func float(_ key: String) -> Float? { (root[key] as? NSNumber)?.floatValue }
        func int(_ key: String) -> Int? { (root[key] as? NSNumber)?.intValue }
        return GenerationOptions(
            maxTokens: int("max_output_tokens"),
            temperature: float("temperature") ?? 0.6,
            topP: float("top_p") ?? 1
        )
    }

    // MARK: tools -> [ToolDefinition]

    private static func buildTools(_ root: [String: Any]) throws -> [ToolDefinition] {
        guard let rawTools = root["tools"] as? [Any] else { return [] }

        var tools: [ToolDefinition] = []
        for raw in rawTools {
            guard let object = raw as? [String: Any] else {
                throw RejectionReason.malformed("tools must be objects")
            }

            let type = (object["type"] as? String) ?? "function"
            guard type == "function" else {
                throw RejectionReason.unsupportedToolType(type)
            }
            guard let name = object["name"] as? String, !name.isEmpty else {
                throw RejectionReason.malformed("function tool requires a name")
            }

            let description = object["description"] as? String
            let parameters = jsonValue(object["parameters"] ?? [String: Any]())
            tools.append(ToolDefinition(name: name, description: description, parameters: parameters))
        }
        return tools
    }
}

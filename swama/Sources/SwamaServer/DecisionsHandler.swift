import Foundation
import NIOCore
import NIOHTTP1
import SwamaCore

// MARK: - DecisionWireError

enum DecisionWireError: Error, LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        if case let .invalid(message) = self {
            return message
        }
        return nil
    }
}

// MARK: - DecisionsHandler

/// SGLang prompt format 1 wire adapter. A future OpenAI Decisions adapter can map to the same
/// Core request without making its evolving JSON contract part of SwamaCore.
enum DecisionsHandler {
    static func handle(requestHead: HTTPRequestHead, body: ByteBuffer, channel: Channel) async {
        await handle(requestHead: requestHead, body: body, channel: channel, engine: ServerCoreEngine.shared)
    }

    static func handle(requestHead: HTTPRequestHead, body: ByteBuffer, channel: Channel, engine: SwamaEngine) async {
        do {
            var readable = body
            guard let bytes = readable.readBytes(length: body.readableBytes) else {
                throw DecisionWireError.invalid("Invalid request body.")
            }

            let object = try JSONDecoder().decode([String: JSONValue].self, from: Data(bytes))
            let request = try parse(object)
            let result = try await engine.decide(request)
            try Task.checkCancellation()
            var answers = [String: [String: Any]]()
            for (id, answer) in result.answers {
                var value: [String: Any] = [
                    "type": answer.type.rawValue,
                    "probabilities": answer.probabilities,
                    "label_mass": answer.labelMass
                ]
                if let choice = answer.choice {
                    value["choice"] = choice
                }
                if let score = answer.score {
                    value["score"] = score
                }
                if let ids = answer.promptTokenIDs {
                    value["prompt_token_ids"] = ids
                }
                if let ids = answer.labelTokenIDs {
                    value["label_token_ids"] = ids
                }
                answers[id] = value
            }
            let response: [String: Any] = [
                "object": "decisions",
                "model": result.model.rawValue,
                "prompt_format_version": result.promptFormatVersion,
                "answers": answers,
                "usage": [
                    "prompt_tokens": result.usage.promptTokens,
                    "completion_tokens": 0,
                    "total_tokens": result.usage.totalTokens
                ]
            ]
            try await send(response, status: .ok, version: requestHead.version, channel: channel)
        }
        catch is CancellationError {
            // The peer has disconnected; there is no response to deliver.
        }
        catch let error as SwamaError {
            let status: HTTPResponseStatus =
                switch error.code {
                case .contextLimitExceeded,
                     .invalidImage,
                     .invalidRequest: .badRequest
                case .modelNotFound: .notFound
                default: .internalServerError
                }
            try? await sendError(error.message, status: status, version: requestHead.version, channel: channel)
        }
        catch {
            let message = error is DecisionWireError ? error.localizedDescription : "Invalid decision request."
            try? await sendError(message, status: .badRequest, version: requestHead.version, channel: channel)
        }
    }

    static func parse(_ body: [String: JSONValue]) throws -> DecisionRequest {
        try allow(
            body,
            [
                "model",
                "input",
                "questions",
                "temperature",
                "chat_template_kwargs",
                "prompt_format_version",
                "return_prompt_token_ids"
            ]
        )
        let model = try string(body, "model")
        let input = try renderedText(body["input"], field: "input")
        guard case let .array(rawQuestions)? = body["questions"], !rawQuestions.isEmpty else {
            throw DecisionWireError.invalid("questions must be a non-empty array.")
        }

        let questions = try rawQuestions.map(parseQuestion)
        let temperature: Double
        if let value = body["temperature"] {
            switch value {
            case let .double(number): temperature = number
            case let .int(number): temperature = Double(number)
            default: throw DecisionWireError.invalid("temperature must be a positive finite number.")
            }
        }
        else {
            temperature = 1
        }
        guard temperature.isFinite, temperature > 0 else {
            throw DecisionWireError.invalid("temperature must be a positive finite number.")
        }

        if let value = body["prompt_format_version"], value != .null {
            guard case let .int(version) = value, version == 1 else {
                throw DecisionWireError.invalid("Only prompt_format_version 1 is served.")
            }
        }
        if let value = body["chat_template_kwargs"] {
            guard case let .object(kwargs) = value else {
                throw DecisionWireError.invalid("chat_template_kwargs must be an object.")
            }

            try allow(kwargs, ["enable_thinking"])
            if let setting = kwargs["enable_thinking"] {
                guard case .bool(false) = setting else {
                    throw DecisionWireError.invalid("Decisions require enable_thinking=false.")
                }
            }
        }
        let returnIDs: Bool
        if let value = body["return_prompt_token_ids"] {
            guard case let .bool(enabled) = value else {
                throw DecisionWireError.invalid("return_prompt_token_ids must be a boolean.")
            }

            returnIDs = enabled
        }
        else {
            returnIDs = false
        }
        return .init(
            model: .init(model),
            input: input,
            questions: questions,
            temperature: temperature,
            returnPromptTokenIDs: returnIDs
        )
    }

    private static func parseQuestion(_ value: JSONValue) throws -> DecisionQuestion {
        guard case let .object(question) = value else {
            throw DecisionWireError.invalid("Each question must be an object.")
        }

        let id = try string(question, "id")
        let type = try string(question, "type")
        let text = try renderedText(question["question"], field: "question")
        switch type {
        case "choice":
            try allow(question, ["id", "type", "question", "options"])
            guard case let .array(rawOptions)? = question["options"], (2 ... 26).contains(rawOptions.count) else {
                throw DecisionWireError.invalid("choice options must contain 2–26 entries.")
            }

            let options = try rawOptions.map { raw -> DecisionOption in
                guard case let .object(option) = raw else {
                    throw DecisionWireError.invalid("Each option must be an object.")
                }

                try allow(option, ["name", "description"])
                let name = try string(option, "name")
                let description = try optionalText(option["description"], field: "description")
                return .init(name: name, description: description)
            }
            return .choice(id: id, question: text, options: options)

        case "score":
            try allow(question, ["id", "type", "question", "levels"])
            guard case let .array(rawLevels)? = question["levels"], (2 ... 10).contains(rawLevels.count) else {
                throw DecisionWireError.invalid("score levels must contain 2–10 entries.")
            }

            return try .score(id: id, question: text, levels: rawLevels.map { try renderedText($0, field: "level") })

        case "yes_no":
            try allow(question, ["id", "type", "question", "yes", "no"])
            return try .yesNo(
                id: id,
                question: text,
                yes: optionalText(question["yes"], field: "yes"),
                no: optionalText(question["no"], field: "no")
            )

        default:
            throw DecisionWireError.invalid("Question type must be choice, score, or yes_no.")
        }
    }

    private static func allow(_ object: [String: JSONValue], _ keys: Set<String>) throws {
        if let unknown = object.keys.first(where: { !keys.contains($0) }) {
            throw DecisionWireError.invalid("Unknown decision field '\(unknown)'.")
        }
    }

    private static func string(_ object: [String: JSONValue], _ key: String) throws -> String {
        guard case let .string(value)? = object[key],
              !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw DecisionWireError.invalid("\(key) must be a non-empty string.")
        }

        return value
    }

    private static func optionalText(_ value: JSONValue?, field: String) throws -> String? {
        guard let value, value != .null else {
            return nil
        }

        return try renderedText(value, field: field, allowBlank: true)
    }

    private static func renderedText(_ value: JSONValue?, field: String, allowBlank: Bool = false) throws -> String {
        guard let value else {
            throw DecisionWireError.invalid("\(field) is required.")
        }

        let text: String
        switch value {
        case let .string(raw): text = raw

        case let .object(object):
            guard allowBlank || !object.isEmpty else {
                throw DecisionWireError.invalid("\(field) must not be blank.")
            }

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            let encoded = try encoder.encode(value)
            text = String(decoding: encoded, as: UTF8.self)

        case let .array(array):
            guard allowBlank || !array.isEmpty else {
                throw DecisionWireError.invalid("\(field) must not be blank.")
            }

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
            let encoded = try encoder.encode(value)
            text = String(decoding: encoded, as: UTF8.self)

        default:
            throw DecisionWireError.invalid("\(field) must be text, an object, or an array.")
        }
        guard allowBlank || !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw DecisionWireError.invalid("\(field) must not be blank.")
        }

        return text
    }

    private static func sendError(
        _ message: String,
        status: HTTPResponseStatus,
        version: HTTPVersion,
        channel: Channel
    ) async throws {
        try await send(
            ["error": ["message": message, "type": status.code >= 500 ? "server_error" : "invalid_request_error"]],
            status: status,
            version: version,
            channel: channel
        )
    }

    private static func send(
        _ value: [String: Any],
        status: HTTPResponseStatus,
        version: HTTPVersion,
        channel: Channel
    ) async throws {
        let data = try JSONSerialization.data(withJSONObject: value)
        var buffer = channel.allocator.buffer(capacity: data.count)
        buffer.writeBytes(data)
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json")
        headers.add(name: "Content-Length", value: "\(buffer.readableBytes)")
        headers.add(name: "Connection", value: "close")
        HTTPHandler.applyCORSHeaders(&headers)
        try await channel.writeAndFlush(HTTPServerResponsePart.head(.init(
            version: version,
            status: status,
            headers: headers
        )))
        try await channel.writeAndFlush(HTTPServerResponsePart.body(.byteBuffer(buffer)))
        try await channel.writeAndFlush(HTTPServerResponsePart.end(nil))
    }
}

import CoreGraphics
import Foundation
import ImageIO
import NIOCore
import NIOEmbedded
import NIOHTTP1
import SwamaCore
@testable import SwamaServer
import Testing
import UniformTypeIdentifiers

// MARK: - OpenAIDecisionsAPITests

/// The OpenAI Decisions wire format on /v1/decisions (openai-openapi 4a4020d8), without a model.
@Suite("OpenAI Decisions wire format")
struct OpenAIDecisionsAPITests {
    private static let predicate = #"{"type":"predicate","instructions":"Is it damaged?"}"#

    private func object(_ text: String) throws -> [String: JSONValue] {
        try JSONDecoder().decode([String: JSONValue].self, from: Data(text.utf8))
    }

    private func parse(_ text: String) throws -> OpenAIDecisionRequest {
        try OpenAIDecisionRequest.parse(object(text))
    }

    private func body(input: String = #""The screen is cracked.""#, questions: String, extra: String = "") -> String {
        #"{"model":"org/model","input":\#(input),"questions":[\#(questions)]\#(extra)}"#
    }

    /// The request must be refused with a message naming the check that refused it.
    private func refusal(_ text: String, contains expected: String, sourceLocation: SourceLocation = #_sourceLocation) {
        do {
            _ = try parse(text)
            Issue.record("Accepted: \(text.prefix(160))", sourceLocation: sourceLocation)
        }
        catch let DecisionWireError.invalid(message) {
            #expect(message.contains(expected), "\(message)", sourceLocation: sourceLocation)
        }
        catch {
            Issue.record("Unexpected error \(error)", sourceLocation: sourceLocation)
        }
    }

    // MARK: Request mapping

    @Test func mapsEachQuestionTypeInOrderWithInternalIDs() throws {
        let request = try parse(body(questions: [
            #"{"type":"choice","name":"team","instructions":"Which team?","choices":[{"value":"billing","description":"money"},{"value":true},{"value":false,"description":""}]}"#,
            Self.predicate,
            #"{"type":"score","name":"urgency","instructions":"How urgent?","levels":[{"label":"low"},{"label":"high","description":"today"},{"label":"none","description":""}]}"#
        ].joined(separator: ","), extra: #","safety_identifier":"user-1""#))
        #expect(request.decision.model == .init("org/model"))
        #expect(request.decision.input == "The screen is cracked.")
        #expect(request.decision.temperature == 1)
        #expect(request.decision.returnPromptTokenIDs == false)
        #expect(request.decision.allowsBlankInput)
        #expect(request.decision.allowsPairLabels == false)
        #expect(request.decision.images.isEmpty)
        #expect(request.decision.questions == [
            .choice(id: "q0", question: "Which team?", options: [
                .init(name: "billing", description: "money"),
                .init(name: "true"),
                .init(name: "false", description: "")
            ]),
            .yesNo(id: "q1", question: "Is it damaged?"),
            .score(id: "q2", question: "How urgent?", levels: ["low", "high: today", "none"])
        ])
        #expect(request.questions.map(\.id) == ["q0", "q1", "q2"])
        #expect(request.questions.map(\.name) == ["team", nil, "urgency"])
        #expect(request.questions.map(\.kind) == [.choice, .yesNo, .score])
        #expect(request.questions[0].values == [.string("billing"), .bool(true), .bool(false)])
        #expect(request.questions[2].labels == ["low", "high", "none"])
        // The decision prompt Core renders for the mapped question.
        #expect(request.decision.questions[0].decisionPrompt(input: "x") ==
            "x\n\nQuestion: Which team?\nA: billing - money\nB: true\nC: false\nAnswer with the letter of one option only."
        )
    }

    @Test func moreThan26ChoicesUsePairLabels() throws {
        func choices(_ count: Int) -> String {
            let values = (0 ..< count).map { #"{"value":"option \#($0)"}"# }.joined(separator: ",")
            return #"{"type":"choice","instructions":"Pick","choices":[\#(values)]}"#
        }
        #expect(try parse(body(questions: choices(26))).decision.allowsPairLabels == false)
        #expect(try parse(body(questions: choices(27))).decision.allowsPairLabels)
        #expect(try parse(body(questions: [Self.predicate, choices(255)].joined(separator: ",")))
            .decision
            .allowsPairLabels
        )
    }

    @Test func messagesJoinTextAndKeepImagesInOrder() throws {
        let png = try image(.png), jpeg = try image(.jpeg)
        let pngText = png.base64EncodedString(), jpegText = jpeg.base64EncodedString()
        let input = """
        [{"role":"user","content":"first"},\
        {"type":"message","role":"user","content":[{"type":"input_text","text":"second"},\
        {"type":"input_image","image_url":"data:image/png;base64,\(pngText)","detail":"low"},\
        {"type":"input_text","text":"third"}]},\
        {"role":"user","content":[{"type":"input_image","image_url":"DATA:image/JPEG;base64,\(jpegText
        )","detail":null}]}]
        """
        let request = try parse(body(input: input, questions: Self.predicate))
        #expect(request.decision.input == "first\nsecond\nthird")
        #expect(request.decision.images == [
            DecisionImage(data: png, mediaType: "image/png"),
            DecisionImage(data: jpeg, mediaType: "image/jpeg")
        ])
        // An image alone, an empty message list and an empty string are all accepted, rendered as is.
        let only =
            #"[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,\#(pngText)"}]}]"#
        #expect(try parse(body(input: only, questions: Self.predicate)).decision.input == "")
        #expect(try parse(body(input: "[]", questions: Self.predicate)).decision.input == "")
        #expect(try parse(body(input: #""""#, questions: Self.predicate)).decision.input == "")
    }

    @Test func safetyIdentifierIsAcceptedAndIgnored() throws {
        let plain = try parse(body(questions: Self.predicate)).decision
        for value in ["null", #""""#, #""\#(String(repeating: "é", count: 128))""#] {
            #expect(try parse(body(questions: Self.predicate, extra: #","safety_identifier":\#(value)"#))
                .decision == plain
            )
        }
    }

    // MARK: Refusals

    @Test func refusesUnknownFieldsAtEveryLevel() throws {
        let png = try image(.png).base64EncodedString()
        refusal(
            body(questions: Self.predicate, extra: #","temperature":1"#),
            contains: "Unknown decision field 'temperature'"
        )
        refusal(
            body(questions: #"{"type":"predicate","instructions":"x","criteria":{}}"#),
            contains: "Unknown decision field 'criteria'"
        )
        refusal(
            body(questions: #"{"type":"predicate","instructions":"x","choices":[]}"#),
            contains: "Unknown decision field 'choices'"
        )
        refusal(
            body(questions: #"{"type":"choice","instructions":"x","choices":[{"value":"a"},{"value":"b"}],"levels":[]}"#
            ),
            contains: "Unknown decision field 'levels'"
        )
        refusal(
            body(questions: #"{"type":"score","instructions":"x","levels":[{"label":"a"},{"label":"b"}],"choices":[]}"#
            ),
            contains: "Unknown decision field 'choices'"
        )
        refusal(
            body(questions: #"{"type":"choice","instructions":"x","choices":[{"value":"a","name":"a"},{"value":"b"}]}"#
            ),
            contains: "Unknown decision field 'name'"
        )
        refusal(
            body(questions: #"{"type":"score","instructions":"x","levels":[{"label":"a","value":0},{"label":"b"}]}"#),
            contains: "Unknown decision field 'value'"
        )
        refusal(
            body(input: #"[{"role":"user","content":"x","name":"n"}]"#, questions: Self.predicate),
            contains: "Unknown decision field 'name'"
        )
        refusal(
            body(
                input: #"[{"role":"user","content":[{"type":"input_text","text":"x","extra":1}]}]"#,
                questions: Self.predicate
            ),
            contains: "Unknown decision field 'extra'"
        )
        refusal(
            body(
                input: #"[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,\#(png)","file_id":"f"}]}]"#,
                questions: Self.predicate
            ),
            contains: "Unknown decision field 'file_id'"
        )
    }

    @Test func refusesUnsupportedInput() throws {
        let gif = try image(.gif).base64EncodedString(), jpeg = try image(.jpeg).base64EncodedString()
        let png = try image(.png)
        func parts(_ parts: String) -> String { #"[{"role":"user","content":[\#(parts)]}]"# }
        func imagePart(_ url: String) -> String { #"{"type":"input_image","image_url":"\#(url)"}"# }
        for role in ["assistant", "system", "developer"] {
            refusal(
                body(input: #"[{"role":"\#(role)","content":"x"}]"#, questions: Self.predicate),
                contains: "input[0] role must be user"
            )
        }
        refusal(body(input: #"[{"content":"x"}]"#, questions: Self.predicate), contains: "role must be user")
        refusal(
            body(input: #"[{"role":"user","content":"x","type":"function_call"}]"#, questions: Self.predicate),
            contains: "input[0] type must be message"
        )
        refusal(body(input: #"[{"role":"user"}]"#, questions: Self.predicate), contains: "content must be a string")
        refusal(body(input: #"["x"]"#, questions: Self.predicate), contains: "input[0] must be a message object")
        refusal(body(input: "42", questions: Self.predicate), contains: "input must be a string or an array")
        refusal(#"{"model":"m","questions":[\#(Self.predicate)]}"#, contains: "input must be a string or an array")
        refusal(
            body(input: parts(#""x""#), questions: Self.predicate),
            contains: "input[0].content[0] must be an object"
        )
        for type in ["input_file", "input_audio", "output_text"] {
            refusal(
                body(input: parts(#"{"type":"\#(type)","text":"x"}"#), questions: Self.predicate),
                contains: "input[0].content[0] type must be input_text or input_image"
            )
        }
        refusal(
            body(input: parts(#"{"type":"input_text"}"#), questions: Self.predicate),
            contains: "text must be a string"
        )
        refusal(
            body(input: parts(#"{"type":"input_image"}"#), questions: Self.predicate),
            contains: "image_url must be a string"
        )
        refusal(
            body(input: parts(imagePart("https://example.com/a.png")), questions: Self.predicate),
            contains: "input[0].content[0] must be a base64 data URL; remote URLs are not accepted"
        )
        refusal(
            body(
                input: parts(
                    #"{"type":"input_image","image_url":"data:image/png;base64,\#(png.base64EncodedString())","detail":"ultra"}"#
                ),
                questions: Self.predicate
            ),
            contains: "detail must be low, high, auto, original, or null"
        )
        // The shared SystemOne image checks, reported with the part's path.
        refusal(
            body(input: parts(imagePart("data:image/gif;base64,\(gif)")), questions: Self.predicate),
            contains: "input[0].content[0] data URL must be image/png"
        )
        refusal(
            body(input: parts(imagePart("data:image/png;base64,\(jpeg)")), questions: Self.predicate),
            contains: "input[0].content[0] is declared image/png but contains image/jpeg"
        )
        refusal(
            body(
                input: parts(imagePart("data:image/png;base64,\(png.prefix(png.count - 1).base64EncodedString())")),
                questions: Self.predicate
            ),
            contains: "input[0].content[0] is truncated or incomplete"
        )
        let five = Array(repeating: imagePart("data:image/png;base64,\(png.base64EncodedString())"), count: 5)
        refusal(
            body(input: parts(five.joined(separator: ",")), questions: Self.predicate),
            contains: "at most 4 images"
        )
        // Four images across messages are accepted.
        let four = Array(repeating: parts(imagePart("data:image/png;base64,\(png.base64EncodedString())")), count: 4)
            .map { String($0.dropFirst().dropLast()) }
            .joined(separator: ",")
        #expect(try parse(body(input: "[\(four)]", questions: Self.predicate)).decision.images.count == 4)
    }

    @Test func refusesMalformedRequestsAndQuestions() throws {
        refusal(#"{"input":"x","questions":[\#(Self.predicate)]}"#, contains: "model must be a non-empty string")
        refusal(
            #"{"model":" ","input":"x","questions":[\#(Self.predicate)]}"#,
            contains: "model must be a non-empty string"
        )
        refusal(
            body(questions: Self.predicate, extra: #","safety_identifier":"\#(String(repeating: "a", count: 129))""#),
            contains: "safety_identifier must be a string of at most 128 characters"
        )
        refusal(body(questions: Self.predicate, extra: #","safety_identifier":7"#), contains: "safety_identifier")
        refusal(body(questions: ""), contains: "questions must be an array of 1–200 questions")
        refusal(#"{"model":"m","input":"x","questions":{}}"#, contains: "questions must be an array of 1–200 questions")
        refusal(
            body(questions: Array(repeating: Self.predicate, count: 201).joined(separator: ",")),
            contains: "questions must be an array of 1–200 questions"
        )
        #expect(try parse(body(questions: Array(repeating: Self.predicate, count: 200).joined(separator: ",")))
            .questions
            .count == 200
        )
        refusal(body(questions: #""x""#), contains: "questions[0] must be an object")
        refusal(
            body(questions: #"{"type":"yes_no","instructions":"x"}"#),
            contains: "questions[0] type must be predicate, choice, or score"
        )
        refusal(body(questions: #"{"instructions":"x"}"#), contains: "questions[0] type must be predicate")
        for instructions in [#""""#, #"" \n ""#, "null", "7"] {
            refusal(
                body(questions: #"{"type":"predicate","instructions":\#(instructions)}"#),
                contains: "questions[0] instructions must be a non-empty string"
            )
        }
        refusal(body(questions: #"{"type":"predicate"}"#), contains: "instructions must be a non-empty string")
        for name in ["null", "1", "[]"] {
            refusal(
                body(questions: #"{"type":"predicate","name":\#(name),"instructions":"x"}"#),
                contains: "questions[0] name must be a string"
            )
        }
        refusal(
            body(questions: [
                #"{"type":"predicate","name":"a","instructions":"x"}"#,
                #"{"type":"predicate","name":"b","instructions":"y"}"#,
                #"{"type":"predicate","name":"a","instructions":"z"}"#
            ].joined(separator: ",")),
            contains: "Duplicate question name 'a'"
        )
        // Unnamed questions may repeat, and names are case-sensitive.
        #expect(try parse(body(questions: [
            Self.predicate, Self.predicate,
            #"{"type":"predicate","name":"a","instructions":"x"}"#,
            #"{"type":"predicate","name":"A","instructions":"x"}"#
        ].joined(separator: ",")))
            .questions
            .count == 4
        )
    }

    @Test func refusesTheRemovedSGLangShape() throws {
        // SGLang prompt format 1 is no longer served here; its fields fail strict parsing.
        refusal(
            #"{"model":"m","input":"x","questions":[{"id":"q","type":"yes_no","question":"True?"}]}"#,
            contains: "questions[0] type must be predicate, choice, or score"
        )
        refusal(
            body(questions: #"{"id":"q","type":"predicate","instructions":"True?"}"#),
            contains: "Unknown decision field 'id'"
        )
        refusal(
            body(questions: #"{"type":"choice","question":"Pick","options":[{"name":"a"},{"name":"b"}]}"#),
            contains: "Unknown decision field 'options'"
        )
        refusal(
            body(questions: #"{"type":"score","instructions":"Rate","levels":["low","high"]}"#),
            contains: "Each score level must be an object"
        )
        for field in [
            #""temperature":1"#,
            #""chat_template_kwargs":{"enable_thinking":false}"#,
            #""prompt_format_version":1"#,
            #""return_prompt_token_ids":true"#
        ] {
            let name = String(field.dropFirst().prefix { $0 != "\"" })
            refusal(body(questions: Self.predicate, extra: "," + field), contains: "Unknown decision field '\(name)'")
        }
    }

    @Test func refusesOutOfRangeChoicesAndLevels() throws {
        func choice(_ values: [String]) -> String {
            let choices = values.map { #"{"value":\#($0)}"# }.joined(separator: ",")
            return body(questions: #"{"type":"choice","instructions":"Pick","choices":[\#(choices)]}"#)
        }
        func score(_ count: Int) -> String {
            let levels = (0 ..< count).map { #"{"label":"level \#($0)"}"# }.joined(separator: ",")
            return body(questions: #"{"type":"score","instructions":"Rate","levels":[\#(levels)]}"#)
        }
        refusal(choice([#""only""#]), contains: "questions[0] choices must contain 2–255 entries")
        refusal(choice((0 ..< 256).map { #""v\#($0)""# }), contains: "choices must contain 2–255 entries")
        refusal(body(questions: #"{"type":"choice","instructions":"Pick"}"#), contains: "choices must contain 2–255")
        #expect(try parse(choice([#""a""#, #""b""#])).questions[0].values.count == 2)
        refusal(score(1), contains: "questions[0] levels must contain 2–10 entries")
        refusal(score(11), contains: "levels must contain 2–10 entries")
        refusal(body(questions: #"{"type":"score","instructions":"Rate"}"#), contains: "levels must contain 2–10")
        #expect(try parse(score(2)).questions[0].labels.count == 2)
        #expect(try parse(score(10)).questions[0].labels.count == 10)

        refusal(choice([#""a""#, #""b""#, #""a""#]), contains: #"questions[0] has the duplicate choice value "a""#)
        refusal(choice(["true", #""b""#, "true"]), contains: "has the duplicate choice value true")
        refusal(choice(["true", #""true""#]), contains: "both the string and the boolean true")
        refusal(choice([#""false""#, "false"]), contains: "both the string and the boolean false")
        #expect(try parse(choice(["true", "false"])).questions[0].values == [.bool(true), .bool(false)])
        for value in ["1", "null", "{}", #"["a"]"#] {
            refusal(choice([value, #""b""#]), contains: "choice value must be a string or a boolean")
        }
        refusal(
            body(questions: #"{"type":"choice","instructions":"Pick","choices":["a","b"]}"#),
            contains: "Each choice must be an object"
        )
        refusal(
            body(
                questions: #"{"type":"choice","instructions":"Pick","choices":[{"value":"a","description":1},{"value":"b"}]}"#
            ),
            contains: "choice description must be a string"
        )
        refusal(
            body(questions: #"{"type":"score","instructions":"Rate","levels":["low","high"]}"#),
            contains: "Each score level must be an object"
        )
        for label in [#"" ""#, "1", "null"] {
            refusal(
                body(questions: #"{"type":"score","instructions":"Rate","levels":[{"label":\#(label)},{"label":"b"}]}"#
                ),
                contains: "level label must be a non-empty string"
            )
        }
        refusal(
            body(
                questions: #"{"type":"score","instructions":"Rate","levels":[{"label":"a","description":1},{"label":"b"}]}"#
            ),
            contains: "level description must be a string"
        )
    }

    // MARK: Response

    @Test func encodesAnswersInOrderWithTypedValues() throws {
        let request = try parse(body(questions: [
            Self.predicate,
            #"{"type":"choice","name":"c","instructions":"Pick","choices":[{"value":true},{"value":false},{"value":"maybe"}]}"#,
            #"{"type":"score","name":"s","instructions":"Rate","levels":[{"label":"low"},{"label":"high","description":"now"}]}"#
        ].joined(separator: ",")))
        let result = DecisionResponse(
            model: .init("org/model"),
            answers: [
                "q2": .init(
                    type: .score, probabilities: ["0": 0.75, "1": 0.25], labelMass: 0.5, score: 1, confidence: 0.5
                ),
                "q0": .init(type: .yesNo, probabilities: ["yes": 0.8, "no": 0.2], labelMass: 0.25),
                "q1": .init(
                    type: .choice, probabilities: ["true": 0.25, "false": 0.5, "maybe": 0.25], labelMass: 0.9,
                    choice: "false", confidence: 0.25
                )
            ],
            usage: .init(promptTokens: 42, completionTokens: 0)
        )
        let expected = #"{"model":"org/model","answers":["#
            + #"{"type":"predicate","name":null,"probability":0.8},"#
            + #"{"type":"choice","name":"c","choice":false,"probabilities":[{"value":true,"probability":0.25},"#
            + #"{"value":false,"probability":0.5},{"value":"maybe","probability":0.25}],"confidence":0.25},"#
            + #"{"type":"score","name":"s","score":1.0,"probabilities":[{"value":0,"label":"low","probability":0.75},"#
            + #"{"value":1,"label":"high","probability":0.25}],"confidence":0.5}],"#
            + #""usage":{"input_tokens":42,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"#
            + #""output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":42}}"#
        #expect(try request.response(result).json == expected)
    }

    @Test func refusesAnswersThatDoNotMatchTheQuestions() throws {
        let request =
            try parse(
                body(questions: #"{"type":"choice","instructions":"Pick","choices":[{"value":"a"},{"value":"b"}]}"#)
            )
        for answer in [
            DecisionAnswer(
                type: .choice,
                probabilities: ["a": 0.5, "c": 0.5],
                labelMass: 1,
                choice: "a",
                confidence: 0
            ),
            DecisionAnswer(
                type: .choice,
                probabilities: ["a": 0.5, "b": 0.5],
                labelMass: 1,
                choice: "c",
                confidence: 0
            ),
            DecisionAnswer(type: .score, probabilities: ["a": 0.5, "b": 0.5], labelMass: 1, score: 0, confidence: 0)
        ] {
            #expect(throws: SwamaError.self) {
                try request.response(.init(model: .init("m"), answers: ["q0": answer], usage: .init(
                    promptTokens: 1,
                    completionTokens: 0
                )))
            }
        }
    }

    // MARK: Handler

    @Test func handlerReturnsCoreAnswersInQuestionOrder() async throws {
        let request = body(questions: [
            #"{"type":"score","name":"s","instructions":"How urgent?","levels":[{"label":"low"},{"label":"mid"},{"label":"high"}]}"#,
            #"{"type":"predicate","name":"p","instructions":"Is it damaged?"}"#,
            #"{"type":"choice","instructions":"Which team?","choices":[{"value":"billing","description":"money"},{"value":false}]}"#
        ].joined(separator: ","))
        let backend = OpenAIDecisionTestBackend()
        let (status, text) = try await handle(request, backend: backend)
        #expect(status == .ok)
        let core = try #require(await backend.responses.first)

        let wire = try #require(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        #expect(Set(wire.keys) == ["model", "answers", "usage"])
        #expect(wire["model"] as? String == "org/model")
        let answers = try #require(wire["answers"] as? [[String: Any]])
        #expect(answers.map { $0["type"] as? String } == ["score", "predicate", "choice"])
        #expect(answers.map { $0["name"] as? String } == ["s", "p", nil])
        #expect(answers[2]["name"] is NSNull)
        // Exactly the numbers Core reports, not renormalized.
        let score = try #require(core.answers["q0"])
        #expect(answers[0]["score"] as? Double == score.score)
        #expect(answers[0]["confidence"] as? Double == score.confidence)
        let levels = try #require(answers[0]["probabilities"] as? [[String: Any]])
        #expect(levels.map { $0["label"] as? String } == ["low", "mid", "high"])
        for (index, level) in levels.enumerated() {
            #expect(level["probability"] as? Double == score.probabilities[String(index)])
        }
        #expect(answers[1]["probability"] as? Double == core.answers["q1"]?.probabilities["yes"])
        let choice = try #require(core.answers["q2"])
        let choices = try #require(answers[2]["probabilities"] as? [[String: Any]])
        #expect(choices[0]["probability"] as? Double == choice.probabilities["billing"])
        #expect(choices[1]["probability"] as? Double == choice.probabilities["false"])
        #expect(answers[2]["choice"] as? String == "billing")
        #expect(answers[2]["confidence"] as? Double == choice.confidence)
        // Typed on the wire: the boolean choice is a JSON boolean and score values are integers.
        #expect(text.contains(#"{"value":false,"probability":"#))
        #expect(text.contains(#"{"value":0,"label":"low","probability":"#))
        #expect(text.hasSuffix(
            #""usage":{"input_tokens":17,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"#
                + #""output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":17}}"#
        ))
        for removed in ["label_mass", "prompt_format_version", "\"object\"", "prompt_token_ids", "refusal"] {
            #expect(!text.contains(removed))
        }
    }

    @Test func handlerRefusesOldShapeAndInvalidRequestsWith400() async throws {
        let backend = OpenAIDecisionTestBackend()
        for (request, message) in [
            (
                #"{"model":"org/model","input":"x","questions":[{"id":"team","type":"choice","question":"Which?","options":[{"name":"a"},{"name":"b"}]}],"prompt_format_version":1}"#,
                "Unknown decision field 'prompt_format_version'."
            ),
            (
                #"{"model":"org/model","input":"x","questions":[{"id":"q","type":"yes_no","question":"True?"}]}"#,
                "questions[0] type must be predicate, choice, or score."
            ),
            (body(questions: Self.predicate, extra: #","temperature":1"#), "Unknown decision field 'temperature'."),
            (body(questions: ""), "questions must be an array of 1–200 questions."),
            ("[]", "Invalid decision request.")
        ] {
            let (status, text) = try await handle(request, backend: backend)
            #expect(status == .badRequest)
            #expect(text == #"{"error":{"message":"\#(message)","type":"invalid_request_error"}}"#)
        }
        #expect(await backend.responses.isEmpty)
    }

    private func handle(
        _ text: String,
        backend: OpenAIDecisionTestBackend
    ) async throws -> (HTTPResponseStatus?, String) {
        let channel = NIOAsyncTestingChannel()
        var buffer = channel.allocator.buffer(capacity: text.utf8.count)
        buffer.writeString(text)
        let head = HTTPRequestHead(version: .http1_1, method: .POST, uri: "/v1/decisions")
        await DecisionsHandler.handle(
            requestHead: head, body: buffer, channel: channel, engine: SwamaEngine(backend: backend)
        )
        var status: HTTPResponseStatus?
        var bytes = ByteBuffer()
        while let part = try await channel.readOutbound(as: HTTPServerResponsePart.self) {
            switch part {
            case let .head(response): status = response.status
            case var .body(.byteBuffer(chunk)): bytes.writeBuffer(&chunk)
            default: break
            }
        }
        return (status, String(decoding: bytes.readableBytesView, as: UTF8.self))
    }

    /// A small PNG, JPEG or GIF.
    private func image(_ type: UTType) throws -> Data {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let context = CGContext(
            data: nil, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 0, space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ),
            let cgImage = context.makeImage()
        else {
            throw DecisionWireError.invalid("image")
        }

        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, type.identifier as CFString, 1, nil) else {
            throw DecisionWireError.invalid("destination")
        }

        CGImageDestinationAddImage(destination, cgImage, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw DecisionWireError.invalid("finalize")
        }

        return output as Data
    }
}

// MARK: - OpenAIDecisionTestBackend

/// Scores every question from fixed, unequal log-probabilities and records the Core responses it returns.
private actor OpenAIDecisionTestBackend: SwamaEngineBackend {
    private(set) var responses: [DecisionResponse] = .init()

    func decide(_ request: DecisionRequest) async throws -> DecisionResponse {
        var answers = [String: DecisionAnswer]()
        for question in request.questions {
            answers[question.id] = try question.decisionAnswer(
                logProbs: question.decisionNames.indices.map { log(0.6 / Double($0 + 2)) },
                temperature: request.temperature,
                promptTokenIDs: nil,
                labelTokenIDs: nil
            )
        }
        let response = DecisionResponse(
            model: request.model, answers: answers, usage: .init(promptTokens: 17, completionTokens: 0)
        )
        responses.append(response)
        return response
    }

    func generate(
        _: GenerationRequest,
        onEvent _: (@Sendable (GenerationEvent) async throws -> Void)?
    ) async throws -> GenerationResponse {
        throw SwamaError(code: .backendFailure, message: "Unexpected generation")
    }

    func embed(_: EmbeddingRequest) async throws -> SwamaCore.EmbeddingResponse {
        throw SwamaError(code: .backendFailure, message: "Unexpected embedding")
    }

    func models() async throws -> [SwamaCore.ModelInfo] { [] }
    func fetch(_ model: ModelID) async throws -> ModelID { model }
    func remove(_: ModelID) async throws {}
    func clearCache(for _: ModelID) async {}
    func clearCache() async {}
}

import Foundation
import SwamaCore
@testable import SwamaServer
import Testing

@Suite("SystemOne wire adapter")
struct SystemOneAPITests {
    private func parse(_ text: String) throws -> SystemOneRequest {
        try .parse(Data(text.utf8))
    }

    private func number(_ value: JSONValue?) -> Double? {
        switch value {
        case let .double(value): value
        case let .int(value): Double(value)
        default: nil
        }
    }

    @Test func preservesRequestAndCriteriaOrderIncludingEscapedNames() throws {
        let request =
            try parse(
                #"{"model":"m","state":{"z":1.0,"a":[true,null,"中文"]},"questions":{"z":{"type":"choice","instructions":"Pick","criteria":{"zebra":null,"\u0061nt":"small"}},"a":{"type":"noul","instructions":"True?","criteria":{"true":{"yes":"affirm"},"false":"deny"}},"s":{"type":"score","instructions":"Rate","criteria":[{"b":2,"a":1},["urgent"]]}}}"#
            )
        #expect(request.decision.input == #"{"z":1.0,"a":[true,null,"中文"]}"#)
        #expect(request.decision.questions.map(\.id) == ["z", "a", "s"])
        #expect(request.decision.questions[0].decisionNames == ["zebra", "ant"])
        #expect(request.decision.questions[0].decisionLabels == ["A", "B"])
        #expect(request.decision
            .questions[0]
            .decisionPrompt(input: "state") ==
            "state\n\nQuestion: Pick\nA: zebra\nB: ant - small\nAnswer with the letter of one option only."
        )
        #expect(request.decision.questions[1] == .yesNo(
            id: "a",
            question: "True?",
            yes: #"{"yes":"affirm"}"#,
            no: "deny"
        ))
        #expect(request.decision.questions[2] == .score(
            id: "s",
            question: "Rate",
            levels: [#"{"b":2,"a":1}"#, #"["urgent"]"#]
        ))
    }

    @Test func jsonRenderingMatchesCompactUnicodePythonFixtures() throws {
        let cases: [(String, String)] = [
            (#"{"b":1.00,"a":-0,"c":1e-7}"#, #"{"b":1.0,"a":0,"c":1e-07}"#),
            (#"["a/b","\u0000","\uD83D\uDE00",-0.0]"#, "[\"a/b\",\"\\u0000\",\"😀\",-0.0]"),
            (#"{"large":123456789012345678901234567890}"#, #"{"large":123456789012345678901234567890}"#),
            ("{\"u\":\"é\u{2028}\u{2029}\"}", "{\"u\":\"é\u{2028}\u{2029}\"}")
        ]
        for (source, expected) in cases {
            #expect(try SystemOneJSON.parse(Data(source.utf8)).json == expected)
        }
        #expect(try SystemOneJSON.parse(Data(#""\u0000tail""#.utf8)).text == "\0tail")
        #expect(throws: (any Error).self) { try SystemOneJSON.parse(Data(#"{"a":1,"\u0061":2}"#.utf8)) }
        #expect(throws: (any Error).self) { try SystemOneJSON.parse(Data(#"{"a":1,}"#.utf8)) }
        #expect(throws: (any Error).self) { try SystemOneJSON.parse(Data(#"{"a":"\uD800"}"#.utf8)) }
    }

    @Test func usesExactlyTheExistingScoredValuesAndOriginalLegend() throws {
        let request =
            try parse(
                #"{"model":"m","state":"context","questions":{"c":{"type":"choice","instructions":"Pick","criteria":{"second":null,"first":"desc"}},"s":{"type":"score","instructions":"Rate","criteria":[{"urgent":false},["now"]]},"n":{"type":"noul","instructions":"True?"}}}"#
            )
        var answers = [String: DecisionAnswer]()
        for question in request.decision.questions {
            answers[question.id] = try question.decisionAnswer(
                logProbs: [log(0.2), log(0.2)],
                temperature: 1,
                promptTokenIDs: nil,
                labelTokenIDs: nil
            )
        }
        let result = DecisionResponse(
            model: .init("m"),
            answers: answers,
            usage: .init(promptTokens: 123, completionTokens: 0)
        )
        let response = try request.response(result)
        let root = try JSONDecoder().decode([String: JSONValue].self, from: Data(response.json.utf8))
        guard case let .object(wireAnswers)? = root["answers"], case let .object(usage)? = root["usage"] else {
            Issue.record("Missing SystemOne fields"); return
        }

        #expect(usage == ["input_tokens": .int(123), "output_tokens": .int(0)])
        guard case let .object(choice)? = wireAnswers["c"], case let .object(score)? = wireAnswers["s"],
              case let .object(noul)? = wireAnswers["n"]
        else {
            Issue.record("Missing typed answers"); return
        }

        #expect(choice["choice"] == .string("second")) // First maximum in the original criteria order.
        #expect(number(choice["confidence"]) == answers["c"]!.confidence)
        #expect(number(score["score"]) == answers["s"]!.score)
        #expect(score["legend"] == .object(["0": .object(["urgent": .bool(false)]), "1": .array([.string("now")])]))
        #expect(noul["type"] == .string("noul"))
        #expect(number(noul["noul"]) == answers["n"]!.probabilities["yes"])
        #expect(noul["confidence"] == nil)
        #expect(number(noul["x_label_mass"]) == answers["n"]!.labelMass)
        for id in ["c", "s"] {
            guard case let .object(answer)? = wireAnswers[id],
                  case let .object(probabilities)? = answer["probabilities"]
            else {
                Issue.record("Missing probabilities"); continue
            }

            for (name, value) in answers[id]!.probabilities {
                #expect(number(probabilities[name]) == value)
            }
        }
    }

    @Test func refusalsKeepTheNativeCapacityAndReasoningBounds() throws {
        for source in [
            #"{"model":"m","state":"x","questions":{"c":{"type":"choice","instructions":"Pick","criteria":{"only":null}}}}"#,
            #"{"model":"m","state":"x","questions":{"s":{"type":"score","instructions":"Rate","criteria":["only"]}}}"#,
            #"{"model":"m","state":"x","questions":{"n":{"type":"noul"}}}"#,
            #"{"model":"m","state":"x","questions":{"n":{"type":"noul","instructions":[]}}}"#,
            #"{"model":"m","state":"x","questions":{"n":{"type":"noul","instructions":"True?"}},"chat_template_kwargs":{"enable_thinking":true}}"#,
            #"{"model":"m","state":"x","questions":{"s":{"type":"score","instructions":"Rate","criteria":[null,"good"]}}}"#
        ] {
            #expect(throws: DecisionWireError.self) { try parse(source) }
        }
        do {
            _ =
                try parse(
                    #"{"model":"m","state":"x","questions":{"c":{"type":"choice","instructions":"Pick","criteria":{"only":null}}}}"#
                )
        }
        catch { #expect(error.localizedDescription.contains("a choice needs at least two options")) }
        do {
            _ =
                try parse(
                    #"{"model":"m","state":"x","questions":{"s":{"type":"score","instructions":"Rate","criteria":["only"]}}}"#
                )
        }
        catch { #expect(error.localizedDescription.contains("a score takes 2 to 10 levels")) }
    }

    @Test func extensionsFollowTheSGLangPolicy() throws {
        let valid =
            #"{"model":"m","state":"x","questions":{"n":{"type":"noul","instructions":"True?"}},"custom_extension":1,"temperature":null,"chat_template_kwargs":{"enable_thinking":false}}"#
        #expect(try parse(valid).decision.temperature == 1)
        #expect(throws: DecisionWireError.self) { try parse(valid.replacingOccurrences(
            of: #""temperature":null"#,
            with: #""temperature":1"#
        )) }
        #expect(throws: DecisionWireError.self) { try parse(valid.replacingOccurrences(
            of: #""instructions":"True?""#,
            with: #""instructions":"True?","typo":1"#
        )) }
    }

    @Test func blankStateIsRenderedAsIsLikeSGLang() throws {
        let questions = #""questions":{"n":{"type":"noul","instructions":"Is it?"},"#
            + #""c":{"type":"choice","instructions":"Which?","criteria":{"a":null,"b":"second"}},"#
            + #""s":{"type":"score","instructions":"How much?","criteria":["low","high"]}}"#
        let request = try parse(#"{"model":"m","state":"","# + questions + "}")
        #expect(request.decision.input == "")
        #expect(request.decision.allowsBlankInput)
        // SGLang: "\n".join([render_text(""), "", *lines]).
        #expect(request.decision.questions.map { $0.decisionPrompt(input: request.decision.input) } == [
            "\n\nIs the following true? Is it?\nAnswer with yes or no only.",
            "\n\nQuestion: Which?\nA: a\nB: b - second\nAnswer with the letter of one option only.",
            "\n\nQuestion: How much?\n0: low\n1: high\nAnswer with the number of one level only."
        ])
        for (state, input) in [(#"" \n ""#, " \n "), ("{}", "{}"), ("[]", "[]")] {
            #expect(try parse(#"{"model":"m","state":"# + state + "," + questions + "}").decision.input == input)
        }
        #expect(throws: DecisionWireError.self) { try parse(#"{"model":"m","state":null,"# + questions + "}") }
    }
}

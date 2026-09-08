import ArgumentParser
import Foundation
@testable import Swama
import SwamaCore
import Testing

@Suite("swama run Core adapter")
struct RunCommandTests {
    @Test func inProcessCoreIsTheDefaultAndDirectRemainsCompatible() throws {
        let defaultCommand = try Run.parse(["org/model", "hello"])
        #expect(try defaultCommand.executionRoute() == .core)

        let directCommand = try Run.parse(["org/model", "hello", "--direct"])
        #expect(try directCommand.executionRoute() == .core)

        let serverCommand = try Run.parse(["org/model", "hello", "--server"])
        #expect(try serverCommand.executionRoute() == .server)

        let conflictingCommand = try Run.parse(["org/model", "hello", "--direct", "--server"])
        #expect(throws: ValidationError.self) {
            _ = try conflictingCommand.executionRoute()
        }
    }

    @Test func everyCoreOptionMapsWithoutAnUpstreamType() throws {
        let command = try Run.parse([
            "org/model", "describe", "--temperature", "0.25", "--top-p", "0.75",
            "--max-tokens", "42", "--repetition-penalty", "1.1", "--context-limit", "4096",
            "--image-paths", "/tmp/example.png", "--no-stream"
        ])

        let request = command.makeCoreRequest(modelName: "resolved/model")
        #expect(request.model == ModelID("resolved/model"))
        #expect(request.options.maxTokens == 42)
        #expect(request.options.temperature == 0.25)
        #expect(request.options.topP == 0.75)
        #expect(request.options.repetitionPenalty == 1.1)
        #expect(request.options.contextLimit == 4096)
        #expect(request.messages.count == 1)
        #expect(request.messages[0].role == .user)
        #expect(request.messages[0].content == [
            .text("describe"),
            .imageURL(URL(fileURLWithPath: "/tmp/example.png"))
        ])
    }

    @Test func explicitServerModeMapsSupportedOptionsAndRejectsContextLimit() throws {
        let command = try Run.parse([
            "org/model", "hello", "--server", "--temperature", "0.2", "--top-p", "0.8",
            "--max-tokens", "17", "--repetition-penalty", "1.05", "--no-stream"
        ])
        try command.validateServerOptions()

        let object = try #require(
            JSONSerialization.jsonObject(with: command.encodedServerRequest(modelName: "resolved/model"))
                as? [String: Any]
        )
        #expect(object["model"] as? String == "resolved/model")
        #expect((object["temperature"] as? NSNumber)?.floatValue == 0.2)
        #expect((object["top_p"] as? NSNumber)?.floatValue == 0.8)
        #expect((object["max_tokens"] as? NSNumber)?.intValue == 17)
        #expect((object["repetition_penalty"] as? NSNumber)?.floatValue == 1.05)
        #expect((object["stream"] as? NSNumber)?.boolValue == false)

        let unsupported = try Run.parse([
            "org/model", "hello", "--server", "--context-limit", "4096"
        ])
        #expect(throws: ValidationError.self) {
            try unsupported.validateServerOptions()
        }
    }
}

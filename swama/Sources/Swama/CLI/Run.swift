import AppKit
import ArgumentParser
import Foundation
import SwamaCore

// MARK: - CompletionRequest

private struct CompletionRequest: Codable {
    let model: String
    let messages: [Message]
    let temperature: Float?
    let top_p: Float?
    let max_tokens: Int?
    let repetition_penalty: Float?
    let stream: Bool?
}

// MARK: - Message

private struct Message: Codable {
    let role: String
    let content: MessageContent
}

// MARK: - MessageContent

private enum MessageContent: Codable {
    case text(String)
    case multimodal([ContentPartValue])

    func encode(to encoder: Encoder) throws {
        switch self {
        case let .text(text):
            try text.encode(to: encoder)
        case let .multimodal(parts):
            try parts.encode(to: encoder)
        }
    }

    init(from decoder: Decoder) throws {
        if let text = try? String(from: decoder) {
            self = .text(text)
        }
        else {
            let parts = try [ContentPartValue](from: decoder)
            self = .multimodal(parts)
        }
    }
}

// MARK: - ContentPartValue

private enum ContentPartValue: Codable {
    case text(String)
    case imageURL(ImageURL)

    private enum CodingKeys: String, CodingKey {
        case type
        case text
        case image_url
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case let .text(text):
            try container.encode("text", forKey: .type)
            try container.encode(text, forKey: .text)

        case let .imageURL(imageURL):
            try container.encode("image_url", forKey: .type)
            try container.encode(imageURL, forKey: .image_url)
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "text":
            let text = try container.decode(String.self, forKey: .text)
            self = .text(text)

        case "image_url":
            let imageURL = try container.decode(ImageURL.self, forKey: .image_url)
            self = .imageURL(imageURL)

        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: container, debugDescription: "Invalid type")
        }
    }
}

// MARK: - ImageURL

private struct ImageURL: Codable {
    let url: String
}

// MARK: - RunError

private enum RunError: Error, LocalizedError {
    case serverError(String)
    case fileNotFound(String)

    var errorDescription: String? {
        switch self {
        case let .serverError(message):
            "Server error: \(message)"
        case let .fileNotFound(path):
            "File not found: \(path)"
        }
    }
}

// MARK: - RunRoute

enum RunRoute: Equatable {
    case core
    case server
}

// MARK: - Run

struct Run: AsyncParsableCommand {
    static let configuration: CommandConfiguration = .init(
        abstract: "Run a local model with a prompt using Swama",
        discussion: """
        Supports both text-only LLM models and vision-language models (VLM) with image inputs.

        Features real-time streaming output by default for immediate response feedback.

        For vision models, use the --image-paths option to include image files:

        Examples:
          swama run qwen3 "Hello, AI"
          swama run gemma3 "What's in this image?" --image-paths image.jpg
          swama run llama-vision "Describe these images" -i img1.png -i img2.jpg
          swama run qwen3 "Explain this" --no-stream  # Disable streaming for complete response
          swama run qwen3 "Hello" --server             # Explicitly use the local HTTP server
        """
    )

    @Argument(help: "Model name or alias, e.g. qwen3 or mlx-community/Llama-3.2-1B-Instruct-4bit")
    var modelName: String

    @Argument(help: "Prompt to run")
    var prompt: String

    @Option(name: [.customShort("t"), .long], help: "Sampling temperature")
    var temperature: Float = 0.6

    @Option(name: [.long], help: "Top-p (nucleus sampling)")
    var topP: Float = 1.0

    @Option(name: [.customShort("n"), .long], help: "Maximum number of tokens to generate")
    var maxTokens: Int?

    @Option(name: [.long], help: "Repetition penalty")
    var repetitionPenalty: Float?

    @Option(
        name: [.customShort("i"), .long],
        help: "Path to image file(s) for vision models (can specify multiple times)"
    )
    var imagePaths: [String] = []

    @Flag(name: [.customShort("s"), .long], inversion: .prefixedNo, help: "Enable streaming output (default: true)")
    var stream: Bool = true

    @Flag(name: [.long], help: "Use in-process execution (kept for compatibility; this is now the default)")
    var direct: Bool = false

    @Flag(name: [.long], help: "Run through the local HTTP server instead of in-process SwamaCore")
    var server: Bool = false

    @Option(name: [.long], help: "Server host (default: localhost)")
    var serverHost: String = "localhost"

    @Option(name: [.long], help: "Server port (default: 28100)")
    var serverPort: Int = 28100

    @OptionGroup()
    var commonOptions: CommonRunOptions

    func run() async throws {
        try await SwamaEngine.withCLIDiagnostics {
            let route = try executionRoute()
            if route == .server {
                try validateServerOptions()
            }

            let engine = SwamaEngine()
            let resolvedModel = try await engine.fetchResolved(ModelID(modelName))

            switch route {
            case .core:
                try await runWithCore(engine: engine, model: resolvedModel)

            case .server:
                if await isServerRunning() == false,
                   await startServerAndWait() == false
                {
                    throw RunError.serverError("local server is unavailable")
                }
                try await runViaServer(modelName: resolvedModel.rawValue)
            }
        }
    }

    func executionRoute() throws -> RunRoute {
        guard direct == false || server == false else {
            throw ValidationError("--direct and --server cannot be used together")
        }

        return server ? .server : .core
    }

    func validateServerOptions() throws {
        guard commonOptions.resolvedContextLimit == nil else {
            throw ValidationError("--context-limit/--num-ctx is not supported with --server")
        }
    }

    func makeCoreRequest(modelName: String) -> GenerationRequest {
        let images = imagePaths.map { path in
            ContentPart.imageURL(URL(fileURLWithPath: path))
        }
        return GenerationRequest(
            model: ModelID(modelName),
            messages: [.init(role: .user, content: [.text(prompt)] + images)],
            options: .init(
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                repetitionPenalty: repetitionPenalty,
                contextLimit: commonOptions.resolvedContextLimit
            )
        )
    }

    func encodedServerRequest(modelName: String) throws -> Data {
        try JSONEncoder().encode(makeServerRequest(modelName: modelName))
    }

    // MARK: - Server Detection and Management

    private func isServerRunning() async -> Bool {
        do {
            let url = URL(string: "http://\(serverHost):\(serverPort)/v1/models")!
            var request = URLRequest(url: url)
            request.timeoutInterval = 2.0

            let (data, response) = try await URLSession.shared.data(for: request)

            // Check both status code and that we got valid JSON response
            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200
            else {
                return false
            }

            let _ = try JSONSerialization.jsonObject(with: data)
            return true
        }
        catch {
            return false
        }
    }

    private func startServerAndWait() async -> Bool {
        let success = launchSwamaApp()
        if !success {
            return false
        }

        // Wait for server to be ready
        return await waitForServerReady()
    }

    private func launchSwamaApp() -> Bool {
        let appPath = "/Applications/Swama.app"

        // Check if app exists
        guard FileManager.default.fileExists(atPath: appPath) else {
            print("❌ Swama.app not found at \(appPath)")
            return false
        }

        let workspace = NSWorkspace.shared
        let appURL = URL(fileURLWithPath: appPath)
        let configuration = NSWorkspace.OpenConfiguration()
        configuration.activates = false // Don't bring to front

        workspace.openApplication(at: appURL, configuration: configuration) { _, error in
            if let error {
                print("❌ Failed to launch Swama.app: \(error)")
            }
        }
        return true
    }

    private func waitForServerReady(timeout: TimeInterval = 30) async -> Bool {
        let startTime = Date()
        let checkInterval: TimeInterval = 1.0

        while Date().timeIntervalSince(startTime) < timeout {
            if await isServerRunning() {
                return true
            }
            try? await Task.sleep(nanoseconds: UInt64(checkInterval * 1_000_000_000))
        }

        return false
    }

    // MARK: - Common Processing

    private func validateImageFiles() throws {
        for imagePath in imagePaths {
            guard FileManager.default.fileExists(atPath: imagePath) else {
                fputs("❌ Image file not found: \(imagePath)\n", stderr)
                throw ExitCode.failure
            }
        }
    }

    private func showResponseHeader() {
        fputs("💬 Response:\n", stdout)
        fflush(stdout)
    }

    private func showCompletionIndicator() {
        fputs("\n✨ Generation completed\n", stdout)
        fflush(stdout)
    }

    // MARK: - Server-based Execution

    private func runViaServer(modelName: String) async throws {
        // Validate image files first
        try validateImageFiles()

        // Process image files for HTTP API
        var processedImages: [String] = []
        if !imagePaths.isEmpty {
            for imagePath in imagePaths {
                let imageData = try Data(contentsOf: URL(fileURLWithPath: imagePath))
                let base64String = imageData.base64EncodedString()
                let dataURI = "data:image/jpeg;base64,\(base64String)"
                processedImages.append(dataURI)
            }
        }

        // Create message content
        let messageContent: MessageContent
        if processedImages.isEmpty {
            messageContent = .text(prompt)
        }
        else {
            var contentParts: [ContentPartValue] = [.text(prompt)]
            for imageURL in processedImages {
                contentParts.append(.imageURL(ImageURL(url: imageURL)))
            }
            messageContent = .multimodal(contentParts)
        }

        let request = makeServerRequest(modelName: modelName, content: messageContent)

        showResponseHeader()

        if stream {
            try await sendStreamingRequest(request)
        }
        else {
            try await sendNonStreamingRequest(request)
        }

        showCompletionIndicator()
    }

    private func makeServerRequest(
        modelName: String,
        content: MessageContent? = nil
    ) -> CompletionRequest {
        let message = Message(
            role: "user",
            content: content ?? .text(prompt)
        )
        return CompletionRequest(
            model: modelName,
            messages: [message],
            temperature: temperature,
            top_p: topP,
            max_tokens: maxTokens,
            repetition_penalty: repetitionPenalty,
            stream: stream
        )
    }

    private func sendStreamingRequest(_ request: CompletionRequest) async throws {
        let url = URL(string: "http://\(serverHost):\(serverPort)/v1/chat/completions")!
        var httpRequest = URLRequest(url: url)
        httpRequest.httpMethod = "POST"
        httpRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        httpRequest.setValue("text/event-stream", forHTTPHeaderField: "Accept")

        let encoder = JSONEncoder()
        httpRequest.httpBody = try encoder.encode(request)

        let (asyncBytes, response) = try await URLSession.shared.bytes(for: httpRequest)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200
        else {
            throw RunError.serverError("HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)")
        }

        // Process SSE stream
        for try await line in asyncBytes.lines {
            if line.hasPrefix("data: ") {
                let jsonString = String(line.dropFirst(6)) // Remove "data: "
                if jsonString == "[DONE]" {
                    break
                }

                if let data = jsonString.data(using: .utf8),
                   let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let choices = json["choices"] as? [[String: Any]],
                   let firstChoice = choices.first,
                   let delta = firstChoice["delta"] as? [String: Any],
                   let content = delta["content"] as? String
                {
                    fputs(content, stdout)
                    fflush(stdout)
                }
            }
        }
    }

    private func sendNonStreamingRequest(_ request: CompletionRequest) async throws {
        let url = URL(string: "http://\(serverHost):\(serverPort)/v1/chat/completions")!
        var httpRequest = URLRequest(url: url)
        httpRequest.httpMethod = "POST"
        httpRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let encoder = JSONEncoder()
        httpRequest.httpBody = try encoder.encode(request)

        let (data, response) = try await URLSession.shared.data(for: httpRequest)

        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200
        else {
            throw RunError.serverError("HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)")
        }

        // Parse response and extract content
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let choices = json?["choices"] as? [[String: Any]],
              let firstChoice = choices.first,
              let message = firstChoice["message"] as? [String: Any],
              let content = message["content"] as? String
        else {
            throw RunError.serverError("Invalid response format")
        }

        print(content)
    }

    // MARK: - In-process Core Execution

    private func runWithCore(engine: SwamaEngine, model: ModelID) async throws {
        // Animation for model loading and response generation
        let animatedMessagePrefix = "Generating response"
        let spinnerFrames = ["/", "-", "\\", "|"]
        var animationDisplayTask: Task<Void, Never>?

        // Defer block to ensure animation line is cleared if an error occurs or scope is exited prematurely
        defer {
            if let task = animationDisplayTask {
                task.cancel() // Request cancellation

                // Perform final cleanup of the animation line
                let cleanupMessageSample =
                    "\(animatedMessagePrefix)... \(spinnerFrames[0])  " // Base message + ... + spinner + few spaces
                let lineToClear = String(
                    repeating: " ",
                    count: cleanupMessageSample.utf8.count + 5
                ) // Extra margin for safety
                fputs("\r\(lineToClear)\r", stdout)
                fflush(stdout)
            }
        }

        animationDisplayTask = Task.detached { @Sendable in
            var frameIndex = 0
            let messagePart = "\(animatedMessagePrefix)... " // e.g., "Generating response... "

            while !Task.isCancelled {
                let currentFrameChar = spinnerFrames[frameIndex % spinnerFrames.count]
                // Print: carriage return, message, spinner char, then a space to clear any previous wider char.
                fputs("\r\(messagePart)\(currentFrameChar) ", stdout)
                fflush(stdout)
                frameIndex += 1
                do {
                    // Sleep for a short duration. Task.sleep is cancellation-aware.
                    try await Task.sleep(nanoseconds: 120_000_000) // 120ms
                }
                catch {
                    // If sleep is cancelled (e.g., task is cancelled), break the loop.
                    break
                }
            }
            // The animation task itself doesn't do the final clear;
            // that's handled by the main thread's explicit cleanup or the defer block.
        }

        try validateImageFiles()

        // Stop animation before starting output
        if let task = animationDisplayTask {
            task.cancel()
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms

            // Clear animation line
            let cleanupMessageSample = "\(animatedMessagePrefix)... \(spinnerFrames[0])  "
            let lineToClear = String(repeating: " ", count: cleanupMessageSample.utf8.count + 5)
            fputs("\r\(lineToClear)\r", stdout)
            fflush(stdout)
        }

        showResponseHeader()

        let request = makeCoreRequest(modelName: model.rawValue)
        let response: GenerationResponse
        if stream {
            response = try await engine.generate(request) { event in
                if case let .textDelta(chunk) = event {
                    fputs(chunk, stdout)
                    fflush(stdout)
                }
            }
            fputs("\n", stdout)
            fflush(stdout)
        }
        else {
            response = try await engine.generate(request)
            print(response.output)
        }

        showCompletionIndicator()
    }
}

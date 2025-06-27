import AppKit
import ArgumentParser
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import SwamaKit

// MARK: - CompletionRequest

private struct CompletionRequest: Codable {
    let model: String
    let messages: [Message]
    let temperature: Float?
    let top_p: Float?
    let max_tokens: Int?
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

    @Flag(name: [.long], help: "Force direct model execution (bypass server)")
    var direct: Bool = false

    @Option(name: [.long], help: "Server host (default: localhost)")
    var serverHost: String = "localhost"

    @Option(name: [.long], help: "Server port (default: 28100)")
    var serverPort: Int = 28100

    func run() async throws {
        let resolvedModelName = ModelAliasResolver.resolve(name: modelName)

        if !direct {
            if await isServerRunning() {
                do {
                    try await runViaServer(modelName: resolvedModelName)
                    return
                }
                catch {
                    print("⚠️  Server request failed, falling back to direct execution...")
                }
            }
            else {
                // Try to start server silently
                if await startServerAndWait() {
                    do {
                        try await runViaServer(modelName: resolvedModelName)
                        return
                    }
                    catch {
                        print("⚠️  Server request failed, falling back to direct execution...")
                    }
                }
                else {
                    print("⚠️  Server startup failed, falling back to direct execution...")
                }
            }
        }

        // Fallback: Direct execution
        try await runDirectly(modelName: resolvedModelName)
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

        // Prepare request
        let message = Message(role: "user", content: messageContent)
        let request = CompletionRequest(
            model: modelName,
            messages: [message],
            temperature: temperature,
            top_p: topP,
            max_tokens: maxTokens,
            stream: stream
        )

        showResponseHeader()

        // Send HTTP request with fallback
        do {
            if stream {
                try await sendStreamingRequest(request)
            }
            else {
                try await sendNonStreamingRequest(request)
            }

            showCompletionIndicator()
        }
        catch {
            // If server request fails, fall back to direct execution
            fputs("\n⚠️  Server request failed, falling back to direct execution...\n", stdout)
            fflush(stdout)
            try await runDirectly(modelName: modelName)
        }
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

    // MARK: - Direct Execution (Fallback)

    private func runDirectly(modelName: String) async throws {
        // Animation for model loading and response generation
        let animatedMessagePrefix = "Generating response"
        var stopAnimationSignal = false
        let spinnerFrames = ["/", "-", "\\", "|"]
        var animationDisplayTask: Task<Void, Never>?

        // Defer block to ensure animation line is cleared if an error occurs or scope is exited prematurely
        defer {
            if animationDisplayTask != nil, !stopAnimationSignal {
                stopAnimationSignal = true
                animationDisplayTask?.cancel() // Request cancellation

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

        animationDisplayTask = Task.detached {
            var frameIndex = 0
            let messagePart = "\(animatedMessagePrefix)... " // e.g., "Generating response... "

            while !stopAnimationSignal, !Task.isCancelled {
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

        // Use ModelPool to properly handle both LLM and VLM models
        let modelPool = ModelPool.shared

        // Validate and process image files
        try validateImageFiles()
        var processedImages: [MLXLMCommon.UserInput.Image] = []
        if !imagePaths.isEmpty {
            for imagePath in imagePaths {
                let imageURL = URL(fileURLWithPath: imagePath)
                processedImages.append(.url(imageURL))
            }
        }

        // Copy images to avoid capture issues in async closure
        let imagesToUse = processedImages

        // Stop animation before starting output
        if animationDisplayTask != nil {
            stopAnimationSignal = true
            animationDisplayTask?.cancel()
            try? await Task.sleep(nanoseconds: 50_000_000) // 50ms

            // Clear animation line
            let cleanupMessageSample = "\(animatedMessagePrefix)... \(spinnerFrames[0])  "
            let lineToClear = String(repeating: " ", count: cleanupMessageSample.utf8.count + 5)
            fputs("\r\(lineToClear)\r", stdout)
            fflush(stdout)
        }

        showResponseHeader()

        let output = try await modelPool.run(modelName: modelName) { runner in
            // Create chat messages with images if provided
            let chatMessages: [MLXLMCommon.Chat.Message] = [
                MLXLMCommon.Chat.Message(role: .user, content: prompt, images: imagesToUse)
            ]

            // Use UserInput with both chat and images
            let userInput = MLXLMCommon.UserInput(chat: chatMessages)

            if stream {
                // Use streaming output for real-time response
                let result = try await runner.runChat(
                    userInput: userInput,
                    parameters: .init(
                        maxTokens: maxTokens,
                        temperature: temperature,
                        topP: topP,
                        repetitionPenalty: repetitionPenalty
                    ),
                    onToken: { chunk in
                        // Print each token as it's generated
                        fputs(chunk, stdout)
                        fflush(stdout)
                    }
                )

                // Print newline after completion
                fputs("\n", stdout)
                fflush(stdout)

                return result.output
            }
            else {
                // Use non-streaming for complete response at once
                let result = try await runner.runChatNonStream(
                    userInput: userInput,
                    parameters: .init(
                        maxTokens: maxTokens,
                        temperature: temperature,
                        topP: topP,
                        repetitionPenalty: repetitionPenalty
                    )
                )

                return result.output
            }
        }

        // For non-streaming mode, print the complete output
        if !stream {
            print(output)
        }

        showCompletionIndicator()
    }
}

import CoreImage
import Foundation
import MLX
import MLXLLM
@preconcurrency import MLXLMCommon
import MLXVLM
import OSLog
import Tokenizers
import struct Tokenizers.ToolSpec

// MARK: - ModelRunner

/// An actor responsible for running model inference.
private let modelRunnerLogger: Logger = .init(subsystem: "SwamaKit", category: "ModelRunner")

// MARK: - InferenceSafetyLimits

private enum InferenceSafetyLimits {
    static let multimodalContextLimit = 4096
}

// MARK: - ModelRunner

public actor ModelRunner {
    public struct ChatRunResult: Sendable {
        public let output: String
        public let analysis: String?
        public let promptTokens: Int
        public let completionInfo: GenerateCompletionInfo?
        public let toolCalls: [MLXLMCommon.ToolCall]
        public let rawText: String
    }

    // MARK: Lifecycle

    /// - Parameter promptCacheStore: defaults to the process-wide singleton; overridable so
    ///   tests can inject an isolated store instead of sharing global state.
    public init(container: ModelContainer, promptCacheStore: PromptCacheStore = .shared) {
        self.container = container
        self.promptCacheStore = promptCacheStore
    }

    // MARK: Public

    /// Runs the model with the given prompt and parameters, returning only the generated output string.
    public func run(prompt: String, images _: [Data]? = nil, parameters: GenerateParameters) async throws -> String {
        // Use new chat-based method for consistency
        let chatMessages: [MLXLMCommon.Chat.Message] = [.user(prompt)]
        let result = try await runWithChatUsage(chatMessages: chatMessages, parameters: parameters)
        return result.output
    }

    /// Runs the model with chat messages, returning the generated output and token usage.
    public nonisolated func runWithChatUsage(
        chatMessages: [MLXLMCommon.Chat.Message],
        parameters: GenerateParameters
    ) async throws -> ChatRunResult {
        let userInput = MLXLMCommon.UserInput(chat: chatMessages)
        return try await runChat(
            userInput: userInput,
            parameters: parameters
        )
    }

    /// Non-streaming chat execution - collects all output and returns at the end
    public nonisolated func runChatNonStream(
        userInput: MLXLMCommon.UserInput,
        parameters: GenerateParameters
    ) async throws -> ChatRunResult {
        // For non-streaming, we don't provide callbacks, so runChat will accumulate internally
        try await runChat(
            userInput: userInput,
            parameters: parameters
        )
    }

    /// Unified method for running chat with optional streaming and tool calls support
    public nonisolated func runChat(
        userInput: MLXLMCommon.UserInput,
        parameters: GenerateParameters,
        onToken: (@Sendable (String) async throws -> Void)? = nil,
        onToolCall: (@Sendable (MLXLMCommon.ToolCall) async throws -> Void)? = nil
    ) async throws -> ChatRunResult {
        try await withError {
            var output = ""
            var promptTokens = 0
            var capturedCompletionInfo: GenerateCompletionInfo?
            var toolCalls: [MLXLMCommon.ToolCall] = []

            let rawOutputStorage = RawOutputBuffer()
            let hasMediaInput = userInput.hasMediaContent
            let configuredContextLimit = await ContextLimitConfig.shared.currentLimit()
            let effectiveContextLimit = hasMediaInput
                ? min(configuredContextLimit, InferenceSafetyLimits.multimodalContextLimit)
                : configuredContextLimit

            if hasMediaInput, effectiveContextLimit < configuredContextLimit {
                modelRunnerLogger.info(
                    "Multimodal request context limit clamped from \(configuredContextLimit) to \(effectiveContextLimit)"
                )
            }

            var effectiveParameters = parameters
            if effectiveParameters.maxKVSize == nil {
                effectiveParameters.maxKVSize = effectiveContextLimit
            }
            let generationParameters = effectiveParameters

            var effectiveInput = userInput
            if case let .chat(messages) = userInput.prompt {
                let trimmedMessages = try await trimChatMessagesInternal(
                    chatMessages: messages,
                    tools: userInput.tools,
                    limit: effectiveContextLimit,
                    container: container,
                    processing: userInput.processing,
                    additionalContext: userInput.additionalContext
                )
                effectiveInput = MLXLMCommon.UserInput(
                    chat: trimmedMessages,
                    processing: userInput.processing,
                    tools: userInput.tools,
                    additionalContext: userInput.additionalContext
                )
            }

            let lmInput = try await container.prepare(input: effectiveInput)

            promptTokens = tokenLength(lmInput.text.tokens)
            guard promptTokens <= effectiveContextLimit else {
                throw ContextLimitError.exceededAfterTrimming(
                    limit: effectiveContextLimit,
                    promptTokens: promptTokens
                )
            }

            let maxKVSize = generationParameters.maxKVSize ?? effectiveContextLimit
            let modelName = await container.configuration.name
            let fullTokens = promptCacheTokenArray(lmInput.text.tokens)

            let cacheDecision = resolvePromptCacheReuse(
                enabled: PromptCacheConfig.isEnabled,
                hasMediaInput: hasMediaInput,
                modelName: modelName,
                newTokens: fullTokens,
                maxKVSize: maxKVSize,
                store: promptCacheStore
            )

            let run: PromptCacheGenerationRun = try await container.perform { context in
                switch cacheDecision {
                case .miss(.disabled), .miss(.multimodal):
                    // Cache untouched -- feature off, or bypassed for multimodal content (image
                    // embeddings aren't in the token sequence a KV cache indexes by). This is
                    // today's path exactly, unchanged from before this feature existed.
                    let stream = try generate(
                        input: lmInput,
                        parameters: generationParameters,
                        context: context
                    )
                    return PromptCacheGenerationRun(stream: stream, task: nil, cache: nil, prefillMs: nil)

                case let .reuse(matchedLength, reusedCache, _):
                    // Cache hit: feed only the unmatched suffix, but prime the penalty
                    // processors with the FULL history (not just the suffix), or their windowed
                    // state would silently diverge from the uncached baseline.
                    let suffixInput = LMInput(tokens: MLXArray(Array(fullTokens[matchedLength...])))
                    return try buildPromptCacheGenerationRun(
                        iterInput: suffixInput,
                        cache: reusedCache,
                        fullHistory: lmInput.text.tokens,
                        context: context,
                        generationParameters: generationParameters
                    )

                case .miss:
                    // A cacheable miss (no prior slot, mismatch, rotated, or a maxKVSize change):
                    // full prefill on a fresh cache, captured so it can be stored for next turn.
                    let freshCache = context.model.newCache(parameters: generationParameters)
                    return try buildPromptCacheGenerationRun(
                        iterInput: lmInput,
                        cache: freshCache,
                        fullHistory: nil,
                        context: context,
                        generationParameters: generationParameters
                    )
                }
            }

            logPromptCacheDecision(
                decision: cacheDecision,
                promptTokens: promptTokens,
                prefillMs: run.prefillMs
            )

            var thrownError: Error?
            for await generationEvent in run.stream {
                if Task.isCancelled {
                    break
                }

                do {
                    switch generationEvent {
                    case let .chunk(chunkString):
                        rawOutputStorage.append(chunkString)

                        if let onToken {
                            // Awaited (and fully written) before the next generation event is
                            // processed, so writes land on the channel in generation order and are
                            // guaranteed drained by the time this call returns; a failed write
                            // throws here and stops generation.
                            try await onToken(chunkString)
                        }
                        else {
                            // Only accumulate if no onToken callback (for non-streaming)
                            output += chunkString
                        }

                    case let .info(info):
                        capturedCompletionInfo = info

                    case let .toolCall(toolCall):
                        // Always accumulate tool calls for the return value
                        toolCalls.append(toolCall)
                        // Also send to callback if provided (for streaming)
                        if let onToolCall {
                            try await onToolCall(toolCall)
                        }
                    }
                }
                catch {
                    thrownError = error
                    break
                }
            }

            // Whether the loop above finished normally, broke on cancellation, or broke on a
            // thrown error, the background generation task may still be touching the cache for a
            // few more ms (see `generateTask`'s doc comment). Await it before this function either
            // stores the cache or returns/rethrows, exactly as ChatSession does -- this is also
            // what makes it safe for the very next request on this model (ModelPool.run serializes
            // by model name) to start touching the same cache/weights right after this returns.
            if let task = run.task {
                await task.value
            }

            // Only ever store on a clean, uncancelled, unthrown completion: on any abort the slot
            // stays absent (it was already removed from the store by `checkout` inside
            // `resolvePromptCacheReuse`, or was never created for a fresh miss), never
            // half-written.
            if let cache = run.cache, thrownError == nil, !Task.isCancelled {
                storePromptCacheIfReusable(
                    cache: cache,
                    fullTokens: fullTokens,
                    promptTokenCount: promptTokens,
                    maxKVSize: maxKVSize,
                    modelName: modelName,
                    store: promptCacheStore
                )
            }

            if let thrownError {
                throw thrownError
            }

            let rawOutput = rawOutputStorage.consume()
            let resolvedOutput = output.isEmpty ? rawOutput : output

            return ChatRunResult(
                output: resolvedOutput,
                analysis: nil,
                promptTokens: promptTokens,
                completionInfo: capturedCompletionInfo,
                toolCalls: toolCalls,
                rawText: rawOutput
            )
        }
    }

    // MARK: - Existing methods

    // MARK: Private

    private let container: ModelContainer
    private let promptCacheStore: PromptCacheStore
}

// MARK: - RawOutputBuffer

private final class RawOutputBuffer: @unchecked Sendable {
    private var storage: String = ""

    func append(_ chunk: String) {
        storage.append(chunk)
    }

    func consume() -> String {
        defer { storage.removeAll(keepingCapacity: false) }
        return storage
    }
}

private func trimChatMessagesInternal(
    chatMessages: [MLXLMCommon.Chat.Message],
    tools: [ToolSpec]?,
    limit: Int,
    container: ModelContainer,
    processing: MLXLMCommon.UserInput.Processing,
    additionalContext: [String: any Sendable]?
) async throws -> [MLXLMCommon.Chat.Message] {
    guard limit > 0 else {
        return chatMessages
    }
    guard !chatMessages.isEmpty else {
        return chatMessages
    }

    func isProtected(_ message: MLXLMCommon.Chat.Message) -> Bool {
        if message.role == .system || message.role == .tool {
            return true
        }
        return !message.images.isEmpty || !message.videos.isEmpty
    }

    func buildInput(with messages: [MLXLMCommon.Chat.Message]) -> MLXLMCommon.UserInput {
        MLXLMCommon.UserInput(
            chat: messages,
            processing: processing,
            tools: tools,
            additionalContext: additionalContext
        )
    }

    func hasMedia(_ messages: [MLXLMCommon.Chat.Message]) -> Bool {
        messages.contains { !$0.images.isEmpty || !$0.videos.isEmpty }
    }

    func hasNonEmptyUserMessage(_ messages: [MLXLMCommon.Chat.Message]) -> Bool {
        messages.contains {
            $0.role == .user && !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }

    func countTokensForTrim(_ messages: [MLXLMCommon.Chat.Message]) async throws -> Int {
        // For text-only chat, use model-accurate token counting via prepare(input:)
        // to avoid template-estimation mismatch for multimodal-capable models.
        if !hasMedia(messages) {
            return try await tokenCount(for: buildInput(with: messages), container: container)
        }

        return try await estimateTokenCount(
            messages: messages,
            tools: tools,
            additionalContext: additionalContext,
            container: container
        )
    }

    var workingMessages = chatMessages
    var didTrimContent = false
    var trimmableIndices = workingMessages.enumerated()
        .filter { !isProtected($0.element) }
        .map(\.offset)

    var currentTokenCount = try await countTokensForTrim(workingMessages)
    let initialTokenCount = currentTokenCount

    var trimPointer = 0

    while currentTokenCount > limit, trimPointer < trimmableIndices.count {
        let index = trimmableIndices[trimPointer]
        let originalContent = workingMessages[index].content

        if originalContent.isEmpty {
            trimPointer += 1
            continue
        }

        let tokens = await container.encode(originalContent)
        if tokens.isEmpty {
            workingMessages[index].content = ""
        }
        else {
            var bestContent: String?
            var low = 0
            var high = tokens.count

            while low <= high {
                let mid = (low + high) / 2
                let prefix = Array(tokens.prefix(mid))
                let decoded = await container.decode(tokens: prefix)
                workingMessages[index].content = decoded

                let count = try await countTokensForTrim(workingMessages)
                if count <= limit {
                    bestContent = decoded
                    low = mid + 1
                }
                else {
                    high = mid - 1
                }
            }

            workingMessages[index].content = bestContent ?? ""
        }

        currentTokenCount = try await countTokensForTrim(workingMessages)
        didTrimContent = true

        if workingMessages[index].content.isEmpty {
            workingMessages.remove(at: index)
            trimmableIndices.remove(at: trimPointer)
            trimmableIndices = trimmableIndices.map { $0 > index ? $0 - 1 : $0 }
            continue
        }

        if currentTokenCount > limit {
            trimPointer += 1
        }
    }

    // Never collapse to an empty-user prompt. If trimming removed all user text,
    // restore the latest non-empty user message from the original request.
    if !hasNonEmptyUserMessage(workingMessages),
       let fallbackUser = chatMessages.reversed().first(where: {
           $0.role == .user && !$0.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
       })
    {
        workingMessages = [fallbackUser]
        didTrimContent = true
    }

    let finalInput = buildInput(with: workingMessages)
    let finalTokenCount = try await tokenCount(for: finalInput, container: container)

    guard finalTokenCount <= limit else {
        modelRunnerLogger.error(
            "Context limit hit; prompt still \(finalTokenCount) tokens with limit \(limit)"
        )
        throw ContextLimitError.exceededAfterTrimming(limit: limit, promptTokens: finalTokenCount)
    }

    if didTrimContent, finalTokenCount < initialTokenCount {
        modelRunnerLogger.info(
            "Context trimmed to \(finalTokenCount) tokens (limit \(limit))"
        )
    }

    return workingMessages
}

private func tokenCount(for input: MLXLMCommon.UserInput, container: ModelContainer) async throws -> Int {
    let prepared = try await container.prepare(input: input)
    return tokenLength(prepared.text.tokens)
}

private func estimateTokenCount(
    messages: [MLXLMCommon.Chat.Message],
    tools: [ToolSpec]?,
    additionalContext: [String: any Sendable]?,
    container: ModelContainer
) async throws -> Int {
    let rawMessages: [MLXLMCommon.Message] = messages.map { message in
        [
            "role": message.role.rawValue,
            "content": message.content
        ]
    }

    let templateTokens: [Int]
    do {
        templateTokens = try await container.perform { context in
            try context.tokenizer.applyChatTemplate(
                messages: rawMessages,
                tools: tools,
                additionalContext: additionalContext
            )
        }
    }
    catch {
        let prompt = messages.map(\.content).joined(separator: "\n\n")
        templateTokens = await container.encode(prompt)
    }

    let mediaItems = messages.reduce(into: 0) { count, message in
        count += message.images.count
        count += message.videos.count
    }
    let estimatedMediaTokens = mediaItems * 400

    return templateTokens.count + estimatedMediaTokens
}

private extension MLXLMCommon.UserInput {
    var hasMediaContent: Bool {
        switch prompt {
        case .text:
            false
        case let .chat(messages):
            messages.contains { !$0.images.isEmpty || !$0.videos.isEmpty }
        case let .messages(messages):
            messages.contains { message in
                message.keys.contains { key in
                    let normalized = key.lowercased()
                    return normalized.contains("image") || normalized.contains("video")
                }
            }
        }
    }
}

private func tokenLength(_ tokens: MLXArray) -> Int {
    switch tokens.ndim {
    case 0:
        1
    case 1:
        tokens.count
    default:
        tokens.dim(-1)
    }
}

// MARK: - Prompt-cache integration

/// One generation attempt's stream/task/cache, as built inside `container.perform` and consumed
/// back in `runChat` after crossing the actor boundary.
///
/// `task` and `cache` are `nil` together on the two paths that never touch `PromptCacheStore` at
/// all (`SWAMA_PROMPT_CACHE=0`, and multimodal requests, which bypass the cache entirely): those
/// keep calling `generate(input:parameters:context:)` exactly as before this feature existed, so
/// there is no task to await or cache to store. Whenever `cache` is non-nil, `task` is too --
/// `runChat` awaits it before ever touching `cache`, exactly as `ChatSession` does.
///
/// `@unchecked Sendable`: `cache` (when present) is `[KVCache]`, a mutable, non-Sendable MLX
/// type; safety comes from this value being built once inside `container.perform` and consumed
/// exactly once back in `runChat`, never shared concurrently.
private struct PromptCacheGenerationRun: @unchecked Sendable {
    let stream: AsyncStream<Generation>
    let task: Task<Void, Never>?
    let cache: [KVCache]?
    let prefillMs: Double?
}

/// Wraps a `LogitProcessor` so `TokenIterator.prepare`'s call to `processor?.prompt(...)` --
/// which only sees the tokens actually fed to this iterator, i.e. the cache-hit suffix -- is
/// redirected to the full post-template token history instead. Repetition/presence/frequency
/// penalties must see the same history the uncached path would give them (in particular
/// `RepetitionContext`'s ring buffer keeps only the last `repetitionContextSize` tokens of
/// whatever it's primed with), or their windowed state silently diverges from the cache-off
/// baseline the moment a request crosses the suffix boundary.
private struct FullHistoryLogitProcessor: LogitProcessor {
    var inner: LogitProcessor
    let fullHistory: MLXArray

    mutating func prompt(_: MLXArray) {
        inner.prompt(fullHistory)
    }

    func process(logits: MLXArray) -> MLXArray {
        inner.process(logits: logits)
    }

    mutating func didSample(token: MLXArray) {
        inner.didSample(token: token)
    }
}

/// Builds a `TokenIterator` (applying, on a cache hit, the full-history processor
/// wrapper) and starts it via `generateTask`, capturing the `Task` so the caller can await it
/// before ever touching `cache` again -- unlike `generate(input:cache:parameters:context:)`,
/// which starts the same task but discards the handle.
///
/// - Parameter fullHistory: the full post-template token sequence to prime penalty
///   processors with, when non-nil (a cache hit, where `iterInput` is only the unmatched
///   suffix). `nil` on a miss, where `iterInput` already *is* the full prompt and needs no
///   wrapper.
private func buildPromptCacheGenerationRun(
    iterInput: LMInput,
    cache: [KVCache],
    fullHistory: MLXArray?,
    context: ModelContext,
    generationParameters: GenerateParameters
) throws -> PromptCacheGenerationRun {
    let baseProcessor: LogitProcessor? = generationParameters.processor()

    let processor: LogitProcessor?
    if let fullHistory, let baseProcessor {
        processor = FullHistoryLogitProcessor(inner: baseProcessor, fullHistory: fullHistory)
    }
    else {
        processor = baseProcessor
    }

    let prefillStart = Date()
    let iterator = try TokenIterator(
        input: iterInput,
        model: context.model,
        cache: cache,
        processor: processor,
        sampler: generationParameters.sampler(),
        prefillStepSize: generationParameters.prefillStepSize,
        maxTokens: generationParameters.maxTokens
    )
    let prefillMs = Date().timeIntervalSince(prefillStart) * 1000

    let (stream, task) = generateTask(
        promptTokenCount: iterInput.text.tokens.size,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator
    )

    return PromptCacheGenerationRun(stream: stream, task: task, cache: cache, prefillMs: prefillMs)
}

/// Converts a token `MLXArray` (as produced by `container.prepare(input:)`) to a plain `[Int]`
/// for prefix matching and slot storage.
private func promptCacheTokenArray(_ tokens: MLXArray) -> [Int] {
    tokens.asType(.int32).asArray(Int32.self).map(Int.init)
}

/// Logs the per-request prompt-cache metrics required for diagnosability: a silently-zero-hit
/// cache should never look identical to a working one in the logs.
private func logPromptCacheDecision(
    decision: PromptCacheResolution,
    promptTokens: Int,
    prefillMs: Double?
) {
    let cachedPrefix: Int
    let reason: PromptCacheMissReason?
    switch decision {
    case let .reuse(matchedLength, _, reuseReason):
        cachedPrefix = matchedLength
        reason = reuseReason
    case let .miss(missReason):
        cachedPrefix = 0
        reason = missReason
    }
    let prefilledSuffix = promptTokens - cachedPrefix
    let prefillDescription = prefillMs.map { String(format: "%.1f", $0) } ?? "n/a"
    let reasonDescription = reason?.rawValue ?? "none"

    modelRunnerLogger.info(
        "prompt_cache prompt_tokens=\(promptTokens) cached_prefix=\(cachedPrefix) prefilled_suffix=\(prefilledSuffix) prefill_ms=\(prefillDescription, privacy: .public) reason=\(reasonDescription, privacy: .public)"
    )
}

/// After a generation completes cleanly, trims `cache` back down to exactly the prompt tokens
/// (discarding the tokens generated during this turn -- see the plan's "what's in the cache
/// after generation" note) and stores it for the next turn, unless rotation makes it unsafe to
/// reuse, in which case the slot is simply not written and the miss is logged for diagnosability.
private func storePromptCacheIfReusable(
    cache: [KVCache],
    fullTokens: [Int],
    promptTokenCount: Int,
    maxKVSize: Int,
    modelName: String,
    store: PromptCacheStore
) {
    guard promptCacheIsReusable(cache) else {
        modelRunnerLogger.info(
            "prompt_cache store skipped model=\(modelName, privacy: .public) reason=\(PromptCacheMissReason.rotated.rawValue, privacy: .public)"
        )
        return
    }

    for layer in cache {
        layer.trim(layer.offset - promptTokenCount)
    }

    store.checkin(
        modelName: modelName,
        slot: PromptCacheSlot(tokens: fullTokens, cache: cache, maxKVSize: maxKVSize)
    )
}

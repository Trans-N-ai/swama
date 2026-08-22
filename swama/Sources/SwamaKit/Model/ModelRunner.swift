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
            let kvConfig = PromptCacheKVConfig(parameters: generationParameters, maxKVSize: maxKVSize)
            let modelName = await container.configuration.name
            let fullTokens = promptCacheTokenArray(lmInput.text.tokens)

            let cacheDecision = resolvePromptCacheReuse(
                enabled: PromptCacheConfig.isEnabled,
                hasMediaInput: hasMediaInput,
                modelName: modelName,
                newTokens: fullTokens,
                kvConfig: kvConfig,
                store: promptCacheStore
            )

            // `resolvePromptCacheReuse` cannot see `generationParameters` at all, so it has no way
            // to know whether a `.reuse` it just returned would actually need
            // `FullHistoryLogitProcessor` wrapped around a processor that also wants a quantized
            // KV cache -- the one combination no `TokenIterator` initializer can honour correctly
            // (see `PromptCacheIteratorStrategy.bypass`). Override that specific case to an
            // equivalent `.miss` here, once, so both the switch below and the log line after it
            // agree on the same effective decision without duplicating this check.
            let effectiveCacheDecision: PromptCacheResolution =
                if case let .reuse(_, _, _, epoch) = cacheDecision,
                promptCacheIteratorStrategy(
                    needsFullHistoryWrapper: true,
                    hasPenaltyProcessor: generationParameters.processor() != nil,
                    kvBits: generationParameters.kvBits
                ) == .bypass {
                    .miss(reason: .kvQuantizationUnsupported, epoch: epoch)
                }
                else {
                    cacheDecision
                }

            // Captured up front (before generation starts) on every path that will ever write a
            // slot back -- `nil` only for `.disabled`/`.multimodal`, which never touch the store
            // at all. `resolvePromptCacheReuse` already captured this from `checkout` (reuse,
            // and the checked-out-but-rejected miss reasons) or `currentEpoch` (`.noSlot`); see
            // `PromptCacheEpoch`'s doc comment for why it has to be this early.
            let cacheEpoch: PromptCacheEpoch? =
                switch effectiveCacheDecision {
                case let .reuse(_, _, _, epoch):
                    epoch
                case let .miss(_, epoch):
                    epoch
                }

            let run: PromptCacheGenerationRun = try await container.perform { context in
                switch effectiveCacheDecision {
                case .miss(.disabled, _),
                     .miss(.kvQuantizationUnsupported, _),
                     .miss(.multimodal, _):
                    // Cache untouched (feature off, or bypassed for multimodal content -- image
                    // embeddings aren't in the token sequence a KV cache indexes by), or a cache
                    // hit we've deliberately declined to reuse because its full-history wrapper
                    // would silently drop the caller's KV-quantization settings
                    // (.kvQuantizationUnsupported). In every one of these cases, any slot
                    // `resolvePromptCacheReuse` already checked out is simply never checked back
                    // in below (`run.cache` is `nil` here), exactly like every other
                    // rejected-but-checked-out miss reason.
                    let stream = try generate(
                        input: lmInput,
                        parameters: generationParameters,
                        context: context
                    )
                    return PromptCacheGenerationRun(stream: stream, task: nil, cache: nil, prefillMs: nil)

                case let .reuse(matchedLength, reusedCache, _, _):
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
                    // A cacheable miss (no prior slot, mismatch, rotated, or a KV config change):
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
                decision: effectiveCacheDecision,
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
            // half-written. `cacheEpoch` is non-nil whenever `run.cache` is (see the comment
            // where it's captured above); binding it alongside the `run.cache` check rather than
            // asserting that pairing means a future divergence skips the store instead of
            // trapping.
            if let cache = run.cache, let cacheEpoch, thrownError == nil, Task.isCancelled == false {
                storePromptCacheIfReusable(
                    cache: cache,
                    fullTokens: fullTokens,
                    promptTokenCount: promptTokens,
                    kvConfig: kvConfig,
                    modelName: modelName,
                    epoch: cacheEpoch,
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

// MARK: - PromptCacheGenerationRun

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

// MARK: - FullHistoryLogitProcessor

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

// MARK: - PromptCacheIteratorError

/// Thrown when `buildPromptCacheGenerationRun` is reached in a state its
/// `PromptCacheIteratorStrategy` says cannot happen -- `runChat` routes `.bypass` to the ordinary
/// uncached path before ever calling it, and `.processorWrapper` is only ever returned when both
/// a processor and a full history exist to wrap.
///
/// This is deliberately a thrown error rather than a `preconditionFailure`: swama is a long-lived
/// server, and a future refactor that broke the invariant should fail this one request (the
/// caller already surfaces a thrown error as a failed completion) rather than take the whole
/// daemon down with every other in-flight request. Silently falling back to the parameters
/// initializer is not an option -- that would drop the caller's KV-quantization settings, exactly
/// what this fix exists to prevent.
enum PromptCacheIteratorError: Error {
    case unreachableStrategy(PromptCacheIteratorStrategy)
}

// MARK: - PromptCacheIteratorStrategy

/// Which `TokenIterator` initializer `buildPromptCacheGenerationRun` should use for one
/// generation attempt. Factored out as a pure function of three plain values (rather than
/// asserted on a constructed `TokenIterator`, whose `kvBits`/`kvGroupSize`/`quantizedKVStart` are
/// internal `let`s not visible from SwamaKit at all) so the initializer-selection decision itself
/// is directly unit-testable, independent of MLX/model machinery.
enum PromptCacheIteratorStrategy: Equatable {
    /// `TokenIterator(input:model:cache:parameters:)` -- kvBits/kvGroupSize/quantizedKVStart are
    /// honoured because this initializer reads them directly off `GenerateParameters`. Used
    /// whenever no full-history wrapper is needed at all: every cacheable-miss path (`iterInput`
    /// already *is* the full prompt) and any cache hit whose `GenerateParameters` yield no
    /// penalty processor (there is nothing for `FullHistoryLogitProcessor` to wrap).
    case parameters

    /// `TokenIterator(input:model:cache:processor:sampler:prefillStepSize:maxTokens:)` with a
    /// `FullHistoryLogitProcessor`-wrapped processor -- a cache hit with a penalty processor,
    /// where the wrapper is required to re-prime penalty state with the full conversation
    /// history rather than just the unmatched suffix. Safe only because this overload's
    /// hardcoded `kvBits: nil` is indistinguishable from what an unset `kvBits` would have
    /// produced via the other initializer: `maybeQuantizeKVCache` is already a no-op whenever
    /// `kvBits == nil`, regardless of `kvGroupSize`/`quantizedKVStart` -- see `.bypass` for the
    /// case where that is *not* true.
    case processorWrapper

    /// Neither initializer can correctly serve this request: a cache hit needs the full-history
    /// wrapper (a penalty processor is configured) *and* the caller wants a quantized KV cache
    /// (`kvBits != nil`), but the only initializer that accepts a custom processor hardcodes
    /// `kvBits: nil` and cannot be told otherwise -- and the initializer that does honour
    /// `kvBits` does not accept a custom processor. There is no combination of the two upstream
    /// initializers that satisfies both requirements at once, so the caller must not build a
    /// `TokenIterator` here at all: decline this cache hit and fall back to an ordinary,
    /// uncached prefill for this one request instead (see
    /// `PromptCacheMissReason.kvQuantizationUnsupported`).
    case bypass
}

/// Pure decision function backing `PromptCacheIteratorStrategy`.
///
/// - Parameters:
///   - needsFullHistoryWrapper: `true` on a cache hit (`iterInput` is only the unmatched
///     suffix, so a penalty processor needs `FullHistoryLogitProcessor` to see the full
///     history); `false` on a miss, where `iterInput` already is the full prompt.
///   - hasPenaltyProcessor: whether `generationParameters.processor()` returned non-nil --
///     i.e. whether there is anything to wrap in the first place.
///   - kvBits: `generationParameters.kvBits`, forwarded as-is.
func promptCacheIteratorStrategy(
    needsFullHistoryWrapper: Bool,
    hasPenaltyProcessor: Bool,
    kvBits: Int?
) -> PromptCacheIteratorStrategy {
    guard needsFullHistoryWrapper, hasPenaltyProcessor else {
        return .parameters
    }

    return kvBits == nil ? .processorWrapper : .bypass
}

/// Builds a `TokenIterator` (applying, on a cache hit with a penalty processor, the full-history
/// wrapper -- see `PromptCacheIteratorStrategy`) and starts it via `generateTask`, capturing the
/// `Task` so the caller can await it before ever touching `cache` again -- unlike
/// `generate(input:cache:parameters:context:)`, which starts the same task but discards the
/// handle.
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
    let strategy = promptCacheIteratorStrategy(
        needsFullHistoryWrapper: fullHistory != nil,
        hasPenaltyProcessor: baseProcessor != nil,
        kvBits: generationParameters.kvBits
    )

    let prefillStart = Date()
    let iterator: TokenIterator
    switch strategy {
    case .parameters:
        iterator = try TokenIterator(
            input: iterInput,
            model: context.model,
            cache: cache,
            parameters: generationParameters
        )

    case .processorWrapper:
        guard let baseProcessor, let fullHistory else {
            throw PromptCacheIteratorError.unreachableStrategy(.processorWrapper)
        }

        let wrappedProcessor = FullHistoryLogitProcessor(inner: baseProcessor, fullHistory: fullHistory)
        iterator = try TokenIterator(
            input: iterInput,
            model: context.model,
            cache: cache,
            processor: wrappedProcessor,
            sampler: generationParameters.sampler(),
            prefillStepSize: generationParameters.prefillStepSize,
            maxTokens: generationParameters.maxTokens
        )

    case .bypass:
        // Unreachable: `runChat` already rewrites a `.bypass` decision to
        // `.miss(.kvQuantizationUnsupported, _)` before ever entering the branch that calls this
        // function (see the `effectiveCacheDecision` override in `runChat`), so this function is
        // only ever invoked for `.parameters`/`.processorWrapper`. Failing loudly rather than
        // silently falling back to `.parameters` keeps this file's "assert, don't assume"
        // register -- see `promptCacheLayerIsReusable`'s rotation guard -- without crashing a
        // long-lived server (see `PromptCacheIteratorError`).
        throw PromptCacheIteratorError.unreachableStrategy(.bypass)
    }
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
    case let .reuse(matchedLength, _, reuseReason, _):
        cachedPrefix = matchedLength
        reason = reuseReason

    case let .miss(missReason, _):
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
/// reuse, or `epoch` has been invalidated by a `drop(modelName:)`/`dropAll()` that ran while this
/// generation was in flight (see `PromptCacheEpoch`), in which case the slot is simply not
/// written and the miss is logged for diagnosability.
private func storePromptCacheIfReusable(
    cache: [KVCache],
    fullTokens: [Int],
    promptTokenCount: Int,
    kvConfig: PromptCacheKVConfig,
    modelName: String,
    epoch: PromptCacheEpoch,
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

    let stored = store.checkin(
        modelName: modelName,
        slot: PromptCacheSlot(tokens: fullTokens, cache: cache, kvConfig: kvConfig),
        epoch: epoch
    )
    if stored == false {
        modelRunnerLogger.info(
            "prompt_cache store skipped model=\(modelName, privacy: .public) reason=\(PromptCacheMissReason.invalidated.rawValue, privacy: .public)"
        )
    }
}

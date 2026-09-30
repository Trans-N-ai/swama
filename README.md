# Swama

[![Swift](https://img.shields.io/badge/Swift-6.2-orange.svg)](https://swift.org)
[![macOS](https://img.shields.io/badge/macOS-15.4+-blue.svg)](https://www.apple.com/macos/)
[![MLX](https://img.shields.io/badge/MLX-Swift-green.svg)](https://github.com/ml-explore/mlx-swift)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> English | [中文](README_CN.md) | [日本語](README_JA.md)

**Swama** is a local AI runtime for Apple Silicon Macs, written in Swift on Apple's [MLX](https://github.com/ml-explore/mlx-swift).
It runs language, vision, embedding, speech-recognition and text-to-speech models on your Mac and serves them through an
OpenAI-compatible API, a command-line tool and a menu bar app.

- **OpenAI-compatible API**: chat completions (streaming, tool calling, image input), a stateless subset of Responses,
  embeddings, audio transcription and text-to-speech (experimental).
- **Model aliases**: `swama run qwen3.5 "…"` downloads the model from Hugging Face on first use.
- **Menu bar app**: runs the server in the background, installs the `swama` command and sets the context limit.

## Requirements

- Apple Silicon Mac, macOS 15.4 or later
- Building from source only: Xcode with the Swift 6.2 toolchain

## Install

**Homebrew**

```bash
brew install swama
```

**App download**: get `Swama.dmg` from [Releases](https://github.com/Trans-N-ai/swama/releases), drag `Swama.app` into
Applications and open it. If macOS blocks the first launch, allow it under **System Settings › Privacy & Security**.
Then choose **Install Command Line Tool…** in the menu bar to add `swama` to your PATH.

**From source**

```bash
git clone https://github.com/Trans-N-ai/swama.git
cd swama/swama
swift build -c release
mv .build/release/swama .build/release/swama-bin   # the app bundles the CLI under this name

cd ../swama-macos/Swama
xcodebuild -project Swama.xcodeproj -scheme Swama -configuration Release
```

## Quick start

```bash
swama run qwen3.5 "Hello!"                          # downloads on first use
swama run qwen3.5 "What's in this image?" -i photo.jpg
swama serve --host 127.0.0.1 --port 28100          # API server
```

```bash
curl http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5", "messages": [{"role": "user", "content": "Hello!"}]}'
```

`swama serve` binds to `0.0.0.0` by default, which makes the API reachable from other machines on your network.
Pass `--host 127.0.0.1` to keep it local.

## Models

Any MLX model on Hugging Face can be used by its full id (for example `mlx-community/Qwen3.5-9B-4bit`). Common models
also have short aliases:

| Kind | Aliases (default first) |
| --- | --- |
| Language | `qwen3.5` (35B-A3B), `qwen3.5-0.8b` / `-2b` / `-4b` / `-9b` / `-27b` / `-122b-a10b` / `-397b-a17b`, `qwen3`, `qwen3-1.7b` / `-30b` / `-32b` / `-235b`, `qwen2.5`, `llama3.2`, `llama3.2-1b`, `llama3.3`, `gpt-oss`, `gpt-oss-120b`, `deepseek-r1`, `deepseek-r1-8b`, `deepseek-coder`, `smollm` |
| Vision | `qwen3.5` (all Qwen3.5 sizes accept images), `gemma3`, `gemma3-1b` / `-12b` / `-27b`, `qwen3-vl`, `qwen3-vl-2b` / `-8b` / `-32b` / `-30b` / `-235b`, `-thinking` variants |
| Speech recognition | `qwen3-asr`, `qwen3-asr-1.7b`, `whisper` (large-v3-turbo) and `whisper-tiny` / `-base` / `-small` / `-medium` / `-large`, `parakeet`, `sensevoice`, `glm-asr`, `voxtral`, `canary`, `moonshine`, `nemotron-asr`, `cohere-transcribe`, `moss-transcribe-diarize`, `wav2vec2`, `mms-asr` |
| Text-to-speech (experimental) | `kokoro`, `orpheus`, `qwen3-tts`, `marvis`, `chatterbox`, `vyvo`, `fish-speech`, `soprano`, `pocket-tts`, `echo-tts`, `kitten-tts`, `irodori-tts`, `omnivoice`, `moss-tts`, `moss-ttsd`, `moss-tts-local` |

The complete alias → model mapping is in
[`ModelAliases.swift`](swama/Sources/SwamaKit/Model/ModelAliases.swift). FireRedASR2 is also supported by its full
repository id.
Known issue: in our testing on 2.4.0, `gemma3` (4B) and `qwen3-vl` (4B) fail to load
([#23](https://github.com/Trans-N-ai/swama/issues/23)); use `qwen3.5` for image input.

## API

The server listens on port 28100 by default (`--port` or `SWAMA_PORT` to change it).

| Endpoint | Notes |
| --- | --- |
| `GET /v1/models` | Downloaded models |
| `POST /v1/chat/completions` | Streaming (`"stream": true`), tool calling, `image_url` input for vision models |
| `POST /v1/responses` | Stateless subset — see below |
| `POST /v1/embeddings` | Embedding models such as `mlx-community/embeddinggemma-300m-4bit` |
| `POST /v1/audio/transcriptions` | Multipart upload, local speech recognition |
| `POST /v1/audio/speech` | Text-to-speech (experimental) |

```bash
# Image input
curl http://localhost:28100/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.5",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What do you see?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}]}]}'

# Tool calling
curl http://localhost:28100/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.5",
  "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]}'

# Embeddings
curl http://localhost:28100/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/embeddinggemma-300m-4bit", "input": ["Hello world"]}'

# Transcription
curl http://localhost:28100/v1/audio/transcriptions -F "file=@audio.wav" -F "model=qwen3-asr"

# Text-to-speech
curl http://localhost:28100/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Hello from Swama", "response_format": "wav"}' --output speech.wav
```

Voices: Orpheus `dan` `jess` `leo` `mia` `tara` `zac` `zoe`; Marvis `conversational_a` `conversational_b`;
Qwen3-TTS and VyvoTTS `en-us-1`; Kokoro defaults to `af_heart`, KittenTTS to `Bella`.

<details>
<summary><b><code>/v1/responses</code> support matrix</b></summary>

`POST /v1/responses` implements an honest, stateless subset of the OpenAI Responses API.

- **Supported**: string or message-item `input`, `instructions`, `input_text` and `input_image` parts, custom `function`
  tools including multi-turn `function_call` / `function_call_output` items, `tool_choice` `"auto"`/`"none"`, basic
  sampling (`temperature`, `top_p`, `max_output_tokens`), non-streaming `Response` objects, and typed SSE streaming
  with monotonic `sequence_number`.
- **Accepted with an explicit local meaning** (the fixed envelope Codex CLI sends — validated, documented, never
  silently honoured):
  - `client_metadata` — client-side tracing only; validated and ignored.
  - `prompt_cache_key` — accepted as a hint and currently not used. Swama's local cache reuses a KV prefix by comparing
    actual prompt tokens, and holds one entry per model, so sessions interleaved against the same model evict each
    other. This is **not** an equivalent of the hosted prompt cache.
  - `reasoning` — the empty object, or `summary: "auto"` (what Codex sends). `auto` leaves the choice to the server and
    a local model emits no reasoning items, so producing none satisfies it.
  - `include: ["reasoning.encrypted_content"]` — accepted; there are no reasoning items to return.
  - `parallel_tool_calls` — this server never executes tools and always emits function-call items one after another,
    satisfying `false`.
  - `tools[{"type":"web_search","external_web_access":false}]` — **accepted and then dropped before the model sees
    it.** Codex CLI 0.147.0 advertises this tool unconditionally and no client setting removes it
    (`tools.web_search = false` only flips `external_web_access`), so refusing it would make the CLI unusable against
    Swama. This is a deliberate, narrow **compatibility degradation, not support**: per OpenAI's reference,
    `external_web_access: false` does not disable search — it runs web search in an offline, cache-only mode over
    OpenAI's own index. Swama has no such index, performs no search whatsoever, and never emits a `web_search_call`.
    Any request that genuinely needs search capability is not supported here. `true`, a missing or non-boolean flag,
    any additional key (`filters`, `search_context_size`, ...), and a `tool_choice` naming `web_search` all remain
    hard 400s.
- **Rejected with an explicit 400 (never silently ignored)**: server-side state (`store: true`,
  `previous_response_id`, `conversation`, `prompt`), `background: true`, hosted/built-in and MCP tools, Structured
  Outputs (`text.format` other than plain text, `response_format`), `truncation: "auto"`, forced `tool_choice`,
  `reasoning.effort` and any `reasoning.summary` other than `"auto"`, any other `include` entry, `max_tool_calls`,
  `service_tier`, `text.verbosity`, `tools[].strict: true`, hosted tool kinds other than the exact offline
  `web_search` shape above (including `namespace` — run Codex with `--disable multi_agent`), image URLs outside
  `http`/`https`/`data:image/...`, every unlisted top-level field, and wrong-typed known fields.
- Responses are not stored: there is no retrieve/cancel/delete by response id.

</details>

## Command line

| Command | Purpose |
| --- | --- |
| `swama run <model> <prompt>` | One-off generation. `-i` image (repeatable), `-t` temperature, `--top-p`, `-n` max tokens, `--repetition-penalty`, `--no-stream` |
| `swama serve` | Start the API server (`--host`, `--port`) — also the default when no command is given |
| `swama pull <model>` | Download a model or alias |
| `swama list [--format json]` | Downloaded models |
| `swama rm <model>` | Delete a downloaded model |
| `swama transcribe <audio>` | Speech to text (`-m` model, `-l` language, `-f simple\|verbose`) |
| `swama create <path> -n <name>` | Register a model directory you already have under a name |
| `swama logs [--follow]` | Read the JSONL diagnostics log |
| `swama menubar` | Run as a menu bar app |

`run` and `serve` accept `--context-limit` (default 16384 tokens).

Environment variables: `SWAMA_PORT` (server port), `SWAMA_MODELS` (model directory, default `~/.swama/models`),
`SWAMA_CONTEXT_LIMIT`, `SWAMA_REGISTRY` (`HUGGING_FACE` or `MODEL_SCOPE`), `SWAMA_PROMPT_CACHE=0` (disable prompt
caching).

## Development

The Swift package is in `swama/` (`swift build`, `swift test`) and the macOS app in `swama-macos/`. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

Built on [mlx-swift](https://github.com/ml-explore/mlx-swift), [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm),
[mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift), [swift-transformers](https://github.com/huggingface/swift-transformers),
[swift-nio](https://github.com/apple/swift-nio) and [swift-argument-parser](https://github.com/apple/swift-argument-parser).

## License

MIT — see [LICENSE](LICENSE). Questions and bug reports: [Issues](https://github.com/Trans-N-ai/swama/issues) ·
[Discussions](https://github.com/Trans-N-ai/swama/discussions).

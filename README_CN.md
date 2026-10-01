# Swama

[![Swift](https://img.shields.io/badge/Swift-6.2-orange.svg)](https://swift.org)
[![macOS](https://img.shields.io/badge/macOS-15.4+-blue.svg)](https://www.apple.com/macos/)
[![MLX](https://img.shields.io/badge/MLX-Swift-green.svg)](https://github.com/ml-explore/mlx-swift)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [English](README.md) | 中文 | [日本語](README_JA.md)

**Swama** 是面向 Apple Silicon Mac 的本地 AI 运行时，用 Swift 编写，基于 Apple 的 [MLX](https://github.com/ml-explore/mlx-swift)。
它在你的 Mac 上运行语言、视觉、嵌入、语音识别和语音合成模型，并通过 OpenAI 兼容 API、命令行工具和菜单栏应用提供服务。

- **OpenAI 兼容 API**：聊天补全（流式、工具调用、图片输入）、Responses 的无状态子集、嵌入、音频转录，以及语音合成（实验性）。
- **决策打分**：SGLang 风格的 `/v1/decisions`，不生成文字，直接给选择题、打分题和是非题的各选项打分。
- **模型别名**：`swama run qwen3.5 "…"` 首次使用时自动从 Hugging Face 下载。
- **菜单栏应用**：在后台运行服务、安装 `swama` 命令、设置上下文长度上限。

## 系统要求

- Apple Silicon Mac，macOS 15.4 或更高
- 仅从源码构建时需要：带 Swift 6.2 工具链的 Xcode

## 安装

**Homebrew**

```bash
brew install swama
```

**下载应用**：从 [Releases](https://github.com/Trans-N-ai/swama/releases) 下载 `Swama.dmg`，把 `Swama.app` 拖进“应用程序”并打开。
如果 macOS 阻止首次启动，请在**系统设置 › 隐私与安全性**中允许。之后在菜单栏选择 **Install Command Line Tool…**，把 `swama` 加入 PATH。

**从源码构建**

```bash
git clone https://github.com/Trans-N-ai/swama.git
cd swama/swama
swift build -c release
mv .build/release/swama .build/release/swama-bin   # 应用以这个文件名打包 CLI

cd ../swama-macos/Swama
xcodebuild -project Swama.xcodeproj -scheme Swama -configuration Release
```

## 快速开始

```bash
swama run qwen3.5 "你好！"                          # 首次使用时自动下载
swama run qwen3.5 "图片里有什么？" -i photo.jpg
swama serve --host 127.0.0.1 --port 28100          # 启动 API 服务
```

```bash
curl http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5", "messages": [{"role": "user", "content": "你好！"}]}'
```

`swama serve` 默认绑定 `0.0.0.0`，同一网络里的其他机器也能访问 API。只想本机使用时请加 `--host 127.0.0.1`。

## 模型

Hugging Face 上的任何 MLX 模型都可以用完整 ID 使用（例如 `mlx-community/Qwen3.5-9B-4bit`）。常用模型有简短别名：

| 类型 | 别名（第一个为默认） |
| --- | --- |
| 语言 | `qwen3.5`（35B-A3B）、`qwen3.5-0.8b` / `-2b` / `-4b` / `-9b` / `-27b` / `-122b-a10b` / `-397b-a17b`、`qwen3`、`qwen3-1.7b` / `-30b` / `-32b` / `-235b`、`qwen2.5`、`llama3.2`、`llama3.2-1b`、`llama3.3`、`gpt-oss`、`gpt-oss-120b`、`deepseek-r1`、`deepseek-r1-8b`、`deepseek-coder`、`smollm` |
| 视觉 | `qwen3.5`（所有 Qwen3.5 尺寸都能看图）、`gemma3`、`gemma3-1b` / `-12b` / `-27b`、`qwen3-vl`、`qwen3-vl-2b` / `-8b` / `-32b` / `-30b` / `-235b`、`-thinking` 变体 |
| 语音识别 | `qwen3-asr`、`qwen3-asr-1.7b`、`whisper`（large-v3-turbo）及 `whisper-tiny` / `-base` / `-small` / `-medium` / `-large`、`parakeet`、`sensevoice`、`glm-asr`、`voxtral`、`canary`、`moonshine`、`nemotron-asr`、`cohere-transcribe`、`moss-transcribe-diarize`、`wav2vec2`、`mms-asr` |
| 语音合成（实验性） | `kokoro`、`orpheus`、`qwen3-tts`、`marvis`、`chatterbox`、`vyvo`、`fish-speech`、`soprano`、`pocket-tts`、`echo-tts`、`kitten-tts`、`irodori-tts`、`omnivoice`、`moss-tts`、`moss-ttsd`、`moss-tts-local` |

完整的别名 → 模型对照见 [`ModelAliases.swift`](swama/Sources/SwamaKit/Model/ModelAliases.swift)。FireRedASR2 也受支持，请使用完整仓库 ID。
已知问题：我们在 2.4.0 上测试时，`gemma3`（4B）和 `qwen3-vl`（4B）无法加载（Gemma 的问题可能与 [#23](https://github.com/Trans-N-ai/swama/issues/23) 有关），图片输入请用 `qwen3.5`。

## API

服务默认监听 28100 端口（用 `--port` 或 `SWAMA_PORT` 修改）。

| 端点 | 说明 |
| --- | --- |
| `GET /v1/models` | 已下载的模型 |
| `POST /v1/chat/completions` | 流式（`"stream": true`）、工具调用、视觉模型的 `image_url` 输入 |
| `POST /v1/responses` | 无状态子集，支持范围见[英文 README 的支持矩阵](README.md#api) |
| `POST /v1/decisions` | 不生成文字，直接给选择题、打分题和是非题打分，说明见[英文 README](README.md#api)（含已知限制：是非题只读小写 `yes`/`no`，MoE 模型概率漂移更大） |
| `POST /v1/embeddings` | 嵌入模型，例如 `mlx-community/embeddinggemma-300m-4bit` |
| `POST /v1/audio/transcriptions` | multipart 上传，本地语音识别 |
| `POST /v1/audio/speech` | 语音合成（实验性） |

```bash
# 图片输入
curl http://localhost:28100/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.5",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "你看到了什么？"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}]}]}'

# 工具调用
curl http://localhost:28100/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.5",
  "messages": [{"role": "user", "content": "东京天气怎么样？"}],
  "tools": [{"type": "function", "function": {"name": "get_weather",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]}'

# 嵌入
curl http://localhost:28100/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/embeddinggemma-300m-4bit", "input": ["Hello world"]}'

# 转录
curl http://localhost:28100/v1/audio/transcriptions -F "file=@audio.wav" -F "model=qwen3-asr"

# 语音合成
curl http://localhost:28100/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Hello from Swama", "response_format": "wav"}' --output speech.wav
```

音色：Orpheus `dan` `jess` `leo` `mia` `tara` `zac` `zoe`；Marvis `conversational_a` `conversational_b`；
Qwen3-TTS 和 VyvoTTS `en-us-1`；Kokoro 默认 `af_heart`，KittenTTS 默认 `Bella`。

## 命令行

| 命令 | 用途 |
| --- | --- |
| `swama run <model> <prompt>` | 单次生成。`-i` 图片（可多次）、`-t` 温度、`--top-p`、`-n` 最大 token 数、`--repetition-penalty`、`--no-stream` |
| `swama serve` | 启动 API 服务（`--host`、`--port`），不带命令时默认执行它 |
| `swama pull <model>` | 下载模型或别名 |
| `swama list [--format json]` | 已下载的模型 |
| `swama rm <model>` | 删除已下载的模型 |
| `swama transcribe <audio>` | 语音转文字（`-m` 模型、`-l` 语言、`-f simple\|json\|verbose`） |
| `swama create <path> -n <name>` | 把已有的模型目录注册成一个名字 |
| `swama logs [--follow]` | 读取 JSONL 诊断日志 |
| `swama menubar` | 以菜单栏应用运行 |

`run` 和 `serve` 支持 `--context-limit`（默认 16384 token）。

环境变量：`SWAMA_PORT`（服务端口）、`SWAMA_MODELS`（模型目录，默认 `~/.swama/models`）、`SWAMA_CONTEXT_LIMIT`、
`SWAMA_REGISTRY`（`HUGGING_FACE` 或 `MODEL_SCOPE`）、`SWAMA_PROMPT_CACHE=0`（关闭提示缓存）、
`SWAMA_DIAGNOSTICS_PATH`（诊断日志位置）、`SWAMA_DIAGNOSTICS_DISABLED=1`（关闭诊断日志）。

## 开发

Swift 包在 `swama/`（`swift build`、`swift test`），macOS 应用在 `swama-macos/`。流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，
漏洞报告见 [SECURITY.md](SECURITY.md)。

基于 [mlx-swift](https://github.com/ml-explore/mlx-swift)、[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)、
[mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift)、[swift-transformers](https://github.com/huggingface/swift-transformers)、
[swift-nio](https://github.com/apple/swift-nio) 和 [swift-argument-parser](https://github.com/apple/swift-argument-parser)。

## 许可证

MIT，详见 [LICENSE](LICENSE)。问题和 bug 报告：[Issues](https://github.com/Trans-N-ai/swama/issues) ·
[Discussions](https://github.com/Trans-N-ai/swama/discussions)。

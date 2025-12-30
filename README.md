# Swama

[![Swift](https://img.shields.io/badge/Swift-6.2-orange.svg)](https://swift.org)
[![macOS](https://img.shields.io/badge/macOS-15.0+-blue.svg)](https://www.apple.com/macos/)
[![MLX](https://img.shields.io/badge/MLX-Swift-green.svg)](https://github.com/ml-explore/mlx-swift)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> English | [中文](README_CN.md) | [日本語](README_JA.md)

**Swama** is a high-performance machine learning runtime written in pure Swift, designed specifically for macOS and built on Apple's MLX framework. It provides a powerful and easy-to-use solution for local LLM (Large Language Model) and VLM (Vision Language Model) inference.

## ✨ Features

- 🚀 **High Performance**: Built on Apple MLX framework, optimized for Apple Silicon
- 🔌 **OpenAI Compatible API**: Standard `/v1/chat/completions`, `/v1/embeddings`, and `/v1/audio/transcriptions` endpoint support with tool calling
- 📱 **Menu Bar App**: Elegant macOS native menu bar integration
- 💻 **Command Line Tools**: Complete CLI support for model management and inference
- 🖼️ **Multimodal Support**: Support for both text and image inputs
- 🎤 **Local Audio Transcription**: Built-in speech recognition with Whisper (no cloud required)
- 🔍 **Text Embeddings**: Built-in embedding generation for semantic search and RAG applications
- 📦 **Smart Model Management**: Automatic downloading, caching, and version management
- 🔄 **Streaming Responses**: Real-time streaming text generation support
- 🌍 **HuggingFace Integration**: Direct model downloads from HuggingFace Hub

## 🏗️ Architecture

Swama features a modular architecture design:

- **SwamaKit**: Core framework library containing all business logic
- **Swama CLI**: Command-line tool providing complete model management and inference functionality
- **Swama.app**: macOS menu bar application with graphical interface and background services

## 📋 System Requirements

- macOS 15.0 or later (Sequoia)
- Apple Silicon (M1/M2/M3/M4)
- Xcode 16.0+ (for compilation)
- Swift 6.2+

## 🛠️ Installation

### 📱 Download Pre-built App (Recommended)

1. **Download the latest release**
   - Go to [Releases](https://github.com/Trans-N-ai/swama/releases)
   - Download `Swama.dmg` from the latest release

2. **Install the app**
   - Double-click `Swama.dmg` to mount the disk image
   - Drag `Swama.app` to the `Applications` folder
   - Launch Swama from Applications or Spotlight
   
   **Note**: On first launch, macOS may show a security warning. If this happens:
   - Go to **System Preferences > Security & Privacy > General**
   - Click **"Open Anyway"** next to the Swama app message
   - Or right-click the app and select **"Open"** from the context menu

3. **Install CLI tools**
   - Open Swama from the menu bar
   - Click "Install Command Line Tool…" to add `swama` command to your PATH

### 🔧 Build from Source (Advanced)

For developers who want to build from source:

```bash
# Clone the repository
git clone https://github.com/Trans-N-ai/swama.git
cd swama

# Build CLI tool
cd swama
swift build -c release
mv .build/release/swama .build/release/swama-bin

# Build macOS app (requires Xcode)
cd ../swama-macos/Swama
xcodebuild -project Swama.xcodeproj -scheme Swama -configuration Release
```

## 🚀 Quick Start

After installing Swama.app, you can use either the menu bar app or command line:

### 1. Instant Inference with Model Aliases

```bash
# Use short aliases instead of full model names - auto-downloads if needed!
swama run qwen3 "Hello, AI"
swama run llama3.2 "Tell me a joke"
swama run gemma3 "What's in this image?" -i /path/to/image.jpg

# Traditional way (also works)
swama run mlx-community/Llama-3.2-1B-Instruct-4bit "Hello, how are you?"

# List downloaded models
swama list
```

**✨ Smart Features:**
- **Model Aliases**: Use friendly names like `qwen3`, `llama3.2`, `deepseek-r1`, `gpt-oss` instead of long URLs
- **Auto-Download**: Models are automatically downloaded on first use - no need to `pull` first!
- **Cache Management**: Downloaded models are cached for future use

### 2. Available Model Aliases

#### Language Models (LLM)

| Alias | Full Model Name | Size | Description |
|-------|-----------------|------|-------------|
| `qwen3` | `mlx-community/Qwen3-8B-4bit` | 4.3 GB | Qwen3 8B (default) |
| `qwen3-1.7b` | `mlx-community/Qwen3-1.7B-4bit` | 938.4 MB | Qwen3 1.7B (lightweight) |
| `qwen3-30b` | `mlx-community/Qwen3-30B-A3B-4bit` | 16.0 GB | Qwen3 30B (high-capacity) |
| `qwen3-32b` | `mlx-community/Qwen3-32B-4bit` | 17.2 GB | Qwen3 32B (ultra-scale) |
| `qwen3-235b` | `mlx-community/Qwen3-235B-A22B-4bit` | 123.2 GB | Qwen3 235B (trillion-scale) |
| `llama3.2` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | 1.7 GB | Llama 3.2 3B (default) |
| `llama3.2-1b` | `mlx-community/Llama-3.2-1B-Instruct-4bit` | 876.3 MB | Llama 3.2 1B (fastest) |
| `deepseek-r1` | `mlx-community/DeepSeek-R1-0528-4bit` | ~32 GB | DeepSeek R1 (reasoning model) |
| `deepseek-r1-8b` | `mlx-community/DeepSeek-R1-0528-Qwen3-8B-8bit` | 8.6 GB | DeepSeek R1 based on Qwen3-8B |
| `qwen2.5` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | 4.0 GB | Qwen 2.5 7B |
| `gpt-oss` | `lmstudio-community/gpt-oss-20b-MLX-8bit` | ~20 GB | GPT-OSS 20B (21B params, 3.6B active) |
| `gpt-oss-120b` | `lmstudio-community/gpt-oss-120b-MLX-8bit` | ~120 GB | GPT-OSS 120B (117B params, 5.1B active) |

#### Vision Language Models (VLM)

| Alias | Full Model Name | Size | Description |
|-------|-----------------|------|-------------|
| `gemma3` | `mlx-community/gemma-3-4b-it-4bit` | 3.2 GB | Gemma 3 4B (default VLM) |
| `gemma3-27b` | `mlx-community/gemma-3-27b-it-4bit` | 15.7 GB | Gemma 3 27B (large-scale VLM) |
| `qwen3-vl` | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | ~4 GB | Qwen3-VL 4B (default VLM) |
| `qwen3-vl-2b` | `mlx-community/Qwen3-VL-2B-Instruct-4bit` | ~2 GB | Qwen3-VL 2B (lightweight) |
| `qwen3-vl-8b` | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | ~8 GB | Qwen3-VL 8B (balanced) |

#### Audio Models (Speech Recognition)

| Alias | Full Model Name | Size | Description |
|-------|-----------------|------|-------------|
| `whisper-large` | `mlx-community/whisper-large-v3-4bit` | 1.6 GB | Whisper Large v3 (highest accuracy) |
| `whisper-medium` | `mlx-community/whisper-medium-4bit` | 791.1 MB | Whisper Medium (balanced) |
| `whisper-small` | `mlx-community/whisper-small-4bit` | 251.7 MB | Whisper Small (fast) |
| `whisper-base` | `mlx-community/whisper-base-4bit` | 77.2 MB | Whisper Base (faster) |
| `whisper-tiny` | `mlx-community/whisper-tiny-4bit` | 40.1 MB | Whisper Tiny (fastest) |
| `funasr` | `mlx-community/Fun-ASR-Nano-2512-4bit` | ~200 MB | FunASR Nano (multilingual) |
| `funasr-mlt` | `mlx-community/Fun-ASR-MLT-Nano-2512-4bit` | ~200 MB | FunASR MLT (multilingual transcription) |

### 3. Start API Service

```bash
# Or start without specifying model (can switch via API)
swama serve --host 0.0.0.0 --port 28100
```

### 4. API Usage

#### 🔌 OpenAI Compatible API

Swama provides a fully OpenAI-compatible API endpoint, allowing you to use it with existing tools and integrations:

```bash
# Get available models
curl http://localhost:28100/v1/models

# Chat completion using aliases (auto-downloads if needed)
curl -X POST http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# Streaming response with DeepSeek R1
curl -X POST http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [
      {"role": "user", "content": "Solve this step by step: What is 15% of 240?"}
    ],
    "stream": true
  }'

# Generate text embeddings
curl -X POST http://localhost:28100/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "Text embeddings"],
    "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
  }'

# Transcribe audio files (local processing)
curl -X POST http://localhost:28100/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large" \
  -F "response_format=json"

# Text-to-speech (TTS)
curl -X POST http://localhost:28100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus",
    "input": "Hello from Swama TTS",
    "voice": "tara",
    "response_format": "wav"
  }' --output speech.wav

# TTS models: orpheus, marvis, chatterbox, chatterbox-turbo, outetts, cosyvoice2, cosyvoice3
# Voice-supported models: orpheus, marvis
# Orpheus voices: tara, leah, jess, leo, dan, mia, zac, zoe
# Marvis voices: conversational_a, conversational_b
# CosyVoice requires explicit reference audio and is not supported by the OpenAI-compatible endpoint

# Tool calling (function calling)
curl -X POST http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'

# Multimodal support (vision language models)
curl -X POST http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What do you see in this image?"},
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
      }
    ]
  }'
```

## 📚 Command Reference

### Model Management

```bash
# Download model (supports both aliases and full names)
swama pull qwen3                    # Using alias
swama pull whisper-large            # Download speech recognition model
swama pull mlx-community/Qwen3-8B-4bit  # Using full name

# List local models and available aliases
swama list [--format json]

# Run inference (auto-downloads if model not found locally)
swama run qwen3 "Your prompt here"              # Using alias - downloads automatically!
swama run deepseek-coder "Write a Python function"  # Another alias
swama run <full-model-name> <prompt> [options]      # Using full name

# Transcribe audio files
swama transcribe audio.wav --model whisper-large --language en
```

### Server

```bash
# Start API server
swama serve [--host HOST] [--port PORT]
```

### Model Aliases

Swama supports convenient aliases for popular models. Use these short names instead of full model URLs:

```bash
# Examples with different model families
swama run qwen3 "Explain machine learning"           # Qwen3 8B
swama run llama3.2-1b "Quick question: what is AI?"  # Llama 3.2 1B (fastest)
swama run deepseek-r1 "Think step by step: 2+2*3"    # DeepSeek R1 (reasoning)
```

### Options

- `--temperature <value>`: Sampling temperature (0.0-2.0)
- `--top-p <value>`: Nucleus sampling parameter (0.0-1.0)
- `--max-tokens <number>`: Maximum number of tokens to generate
- `--repetition-penalty <value>`: Repetition penalty factor

## 🔧 Development

### Dependencies

- [swift-nio](https://github.com/apple/swift-nio) - High-performance networking framework
- [swift-argument-parser](https://github.com/apple/swift-argument-parser) - Command-line argument parsing
- [mlx-swift](https://github.com/ml-explore/mlx-swift) - Apple MLX Swift bindings
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) - MLX Swift language models
- [mlx-swift-audio](https://github.com/DePasqualeOrg/mlx-swift-audio) - MLX Swift audio processing (Whisper, FunASR)

### Building

```bash
# Development build
swift build

# Release build
swift build -c release

# Run tests
swift test

# Generate Xcode project
swift package generate-xcodeproj
```

## 🤝 Contributing

We welcome community contributions! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow Swift coding style guidelines
- Add tests for new features
- Update relevant documentation
- Ensure all tests pass

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Apple MLX](https://github.com/ml-explore/mlx) team for the excellent machine learning framework
- [Swift NIO](https://github.com/apple/swift-nio) for high-performance networking support
- All contributors and community members

## 📞 Support

- 📝 [Issue Tracker](https://github.com/Trans-N-ai/swama/issues)
- 💬 [Discussions](https://github.com/Trans-N-ai/swama/discussions)

## 🗺️ Roadmap

- TODO

---

**Swama** - Bringing the best local AI experience to macOS users 🚀

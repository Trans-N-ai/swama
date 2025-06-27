# Swama

[![Swift](https://img.shields.io/badge/Swift-6.1-orange.svg)](https://swift.org)
[![macOS](https://img.shields.io/badge/macOS-14.0+-blue.svg)](https://www.apple.com/macos/)
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

- macOS 14.0 or later
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15.0+ (for compilation)
- Swift 6.1+

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
- **Model Aliases**: Use friendly names like `qwen3`, `llama3.2`, `deepseek-r1` instead of long URLs
- **Auto-Download**: Models are automatically downloaded on first use - no need to `pull` first!
- **Cache Management**: Downloaded models are cached for future use

### 2. Available Model Aliases

| Alias | Full Model Name | Description |
|-------|----------------|-------------|
| `qwen3` | `mlx-community/Qwen3-8B-4bit` | Qwen3 8B (default) |
| `qwen3-1.7b` | `mlx-community/Qwen3-1.7B-4bit` | Qwen3 1.7B (lightweight) |
| `llama3.2` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | Llama 3.2 3B (default) |
| `gemma3` | `mlx-community/gemma-3-27b-it-4bit` | Gemma 3 (VLM - vision language model) |
| `deepseek-r1` | `mlx-community/DeepSeek-R1-0528-4bit` | DeepSeek R1 (reasoning) |
| `qwen2.5` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Qwen 2.5 7B |
| `whisper-large` | `openai_whisper-large-v3` | Whisper Large (speech recognition) |
| `whisper-base` | `openai_whisper-base` | Whisper Base (faster, lower accuracy) |

### 3. Start API Service

```bash
# Or start without specifying model (can switch via API)
swama serve --host 0.0.0.0 --port 28100
```

### 5. API Usage

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
swama serve [--host HOST] [--port PORT] [--model MODEL_ALIAS]
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
- [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) - MLX Swift examples and models

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

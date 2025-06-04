# Swama

[![Swift](https://img.shields.io/badge/Swift-6.1-orange.svg)](https://swift.org)
[![macOS](https://img.shields.io/badge/macOS-14.0+-blue.svg)](https://www.apple.com/macos/)
[![MLX](https://img.shields.io/badge/MLX-Swift-green.svg)](https://github.com/ml-explore/mlx-swift)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [中文](README_CN.md) | [日本語](README_JA.md) | English

**Swama** is a high-performance machine learning runtime written in pure Swift, designed specifically for macOS and built on Apple's MLX framework. It provides a powerful and easy-to-use solution for local LLM (Large Language Model) and VLM (Vision Language Model) inference.

## ✨ Features

- 🚀 **High Performance**: Built on Apple MLX framework, optimized for Apple Silicon
- 🔌 **OpenAI Compatible API**: Standard `/v1/chat/completions` endpoint support
- 📱 **Menu Bar App**: Elegant macOS native menu bar integration
- 💻 **Command Line Tools**: Complete CLI support for model management and inference
- 🖼️ **Multimodal Support**: Support for both text and image inputs
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
- Apple Silicon (M1/M2/M3)
- Xcode 15.0+ (for compilation)
- Swift 6.1+

## 🛠️ Installation

### 📱 Download Pre-built App (Recommended)

1. **Download the latest release**
   - Go to [Releases](https://github.com/Trans-N-ai/swama/releases)
   - Download `Swama.zip` from the latest release
   - Extract the zip file

2. **Install the app**
   ```bash
   # Move to Applications folder
   mv Swama.app /Applications/
   
   # Launch the app
   open /Applications/Swama.app
   ```
   
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
swift build -c release
sudo cp .build/release/swama /usr/local/bin/

# Build macOS app (requires Xcode)
cd swama-macos/Swama
xcodebuild -project Swama.xcodeproj -scheme Swama -configuration Release
```

## 🚀 Quick Start

After installing Swama.app, you can use either the menu bar app or command line:

### 1. Instant Inference with Model Aliases

```bash
# Use short aliases instead of full model names - auto-downloads if needed!
swama run qwen3 "Hello, AI!"
swama run llama3.2 "Tell me a joke"
swama run deepseek-r1 "Explain quantum computing"

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
| `llama3.2-1b` | `mlx-community/Llama-3.2-1B-Instruct-4bit` | Llama 3.2 1B (fastest) |
| `deepseek-r1` | `mlx-community/DeepSeek-R1-0528-4bit` | DeepSeek R1 (reasoning) |
| `deepseek-coder` | `mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` | DeepSeek Coder |
| `qwen2.5` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Qwen 2.5 7B |

### 3. Start API Service

```bash
# Or start without specifying model (can switch via API)
swama serve --host 0.0.0.0 --port 28100
```

### 4. Menu Bar App

```bash
# Launch menu bar application
swama menubar
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
```

#### 🛠️ Community Tool Integration

Since Swama provides OpenAI-compatible endpoints, you can easily integrate it with popular community tools:

**🤖 AI Coding Assistants:**
```bash
# Continue.dev - Add to config.json
{
  "models": [{
    "title": "Swama Local",
    "provider": "openai",
    "model": "qwen3",
    "apiBase": "http://localhost:28100/v1"
  }]
}

# Cursor - Set custom API endpoint
# API Base URL: http://localhost:28100/v1
# Model: qwen3 or deepseek-coder
```

**💬 Chat Interfaces:**
```bash
# Open WebUI (formerly Ollama WebUI)
# Add OpenAI API connection:
# Base URL: http://localhost:28100/v1
# API Key: not-required

# LibreChat
# Add to .env file:
OPENAI_API_KEY=not-required
OPENAI_REVERSE_PROXY=http://localhost:28100/v1

# ChatBox
# Add OpenAI API provider with base URL: http://localhost:28100/v1
```

**🔧 Development Tools:**
```python
# Python with OpenAI library
import openai

client = openai.OpenAI(
    base_url="http://localhost:28100/v1",
    api_key="not-required"  # Swama doesn't require API keys
)

response = client.chat.completions.create(
    model="qwen3",
    messages=[{"role": "user", "content": "Hello from Python!"}]
)
```

```javascript
// Node.js with OpenAI library
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://localhost:28100/v1',
  apiKey: 'not-required'
});

const completion = await openai.chat.completions.create({
  model: 'deepseek-coder',
  messages: [{ role: 'user', content: 'Write a hello world function' }]
});
```

**📊 Popular Integrations:**
- **Langchain/LlamaIndex**: Use OpenAI provider with custom base URL
- **AutoGen**: Configure as OpenAI endpoint for multi-agent conversations  
- **Semantic Kernel**: Add as OpenAI chat completion service
- **Flowise/Langflow**: Connect via OpenAI node with custom endpoint
- **Anything**: Any tool supporting OpenAI API can connect to Swama!

## 📚 Command Reference

### Model Management

```bash
# Download model (supports both aliases and full names)
swama pull qwen3                    # Using alias
swama pull mlx-community/Qwen3-8B-4bit  # Using full name

# List local models and available aliases
swama list [--format json]

# Run inference (auto-downloads if model not found locally)
swama run qwen3 "Your prompt here"              # Using alias - downloads automatically!
swama run deepseek-coder "Write a Python function"  # Another alias
swama run <full-model-name> <prompt> [options]      # Using full name
```

### Server

```bash
# Start API server
swama serve [--host HOST] [--port PORT] [--model MODEL_ALIAS]

# Start menu bar app
swama menubar
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

## 🖼️ Multimodal Support

Swama supports vision language models and can process image inputs:

```bash
curl -X POST http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/llava-v1.6-mistral-7b-hf-4bit",
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

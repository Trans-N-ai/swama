# Swama

[![Swift](https://img.shields.io/badge/Swift-6.1-orange.svg)](https://swift.org)
[![macOS](https://img.shields.io/badge/macOS-14.0+-blue.svg)](https://www.apple.com/macos/)
[![MLX](https://img.shields.io/badge/MLX-Swift-green.svg)](https://github.com/ml-explore/mlx-swift)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [English](README.md) |  中文版本 | [日本語](README_JA.md) 

**Swama** 是一个用纯 Swift 编写的高性能机器学习运行时，专为 macOS 设计，基于 Apple 的 MLX 框架。它为本地 LLM（大语言模型）和 VLM（视觉语言模型）推理提供了强大且易用的解决方案。

## ✨ 特性

- 🚀 **高性能**: 基于 Apple MLX 框架，针对 Apple Silicon 优化
- 🔌 **OpenAI 兼容 API**: 提供标准的 `/v1/chat/completions`、`/v1/embeddings` 和 `/v1/audio/transcriptions` 端点，支持工具调用
- 📱 **菜单栏应用**: 优雅的 macOS 原生菜单栏集成
- 💻 **命令行工具**: 完整的 CLI 支持用于模型管理和推理
- 🖼️ **多模态支持**: 同时支持文本和图像输入
- 🎤 **本地音频转录**: 内置 Whisper 语音识别（无需云服务）
- 🔍 **文本嵌入**: 内置嵌入生成功能，支持语义搜索和 RAG 应用
- 📦 **智能模型管理**: 自动下载、缓存和版本管理
- 🔄 **流式响应**: 支持实时流式文本生成
- 🌍 **HuggingFace 集成**: 直接从 HuggingFace Hub 下载模型

## 🏗️ 架构

Swama 采用模块化架构设计：

- **SwamaKit**: 核心框架库，包含所有业务逻辑
- **Swama CLI**: 命令行工具，提供完整的模型管理和推理功能
- **Swama.app**: macOS 菜单栏应用，提供图形界面和后台服务

## 📋 系统要求

- macOS 14.0 或更高版本
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15.0+ (用于编译)
- Swift 6.1+

## 🛠️ 安装

### 📱 下载预构建应用（推荐）

1. **下载最新版本**
   - 访问 [Releases](https://github.com/Trans-N-ai/swama/releases) 页面
   - 从最新版本中下载 `Swama.dmg`

2. **安装应用**
   - 双击 `Swama.dmg` 挂载磁盘镜像
   - 将 `Swama.app` 拖拽到 `Applications` 文件夹
   - 从应用程序或聚焦搜索启动 Swama
   
   **注意**: 首次启动时，macOS 可能会显示安全警告。如果出现此情况：
   - 前往 **系统偏好设置 > 安全性与隐私 > 通用**
   - 点击 Swama 应用信息旁边的 **"仍要打开"**
   - 或右键点击应用并从菜单中选择 **"打开"**

3. **安装命令行工具**
   - 从菜单栏打开 Swama
   - 点击"Install Command Line Tool…"将 `swama` 命令添加到 PATH

### 🔧 从源码构建（高级用户）

适合想要从源码构建的开发者：

```bash
# 克隆仓库
git clone https://github.com/Trans-N-ai/swama.git
cd swama

# 构建 CLI 工具
cd swama
swift build -c release
mv .build/release/swama .build/release/swama-bin

# 构建 macOS 应用（需要 Xcode）
cd ../swama-macos/Swama
xcodebuild -project Swama.xcodeproj -scheme Swama -configuration Release
```

## 🚀 快速开始

安装 Swama.app 后，您可以使用菜单栏应用或命令行：

### 1. 使用模型别名即时推理

```bash
# 使用简短的别名而不是完整模型名 - 需要时自动下载！
swama run qwen3 "你好，AI"
swama run llama3.2 "给我讲个笑话"
swama run gemma3 "这张图片里有什么？" -i /path/to/image.jpg

# 传统方式（同样有效）
swama run mlx-community/Llama-3.2-1B-Instruct-4bit "Hello, how are you?"

# 查看已下载的模型
swama list
```

**✨ 智能特性:**
- **模型别名**: 使用友好的名称如 `qwen3`、`llama3.2`、`deepseek-r1` 而不是长链接
- **自动下载**: 首次使用时自动下载模型 - 无需先执行 `pull`！
- **缓存管理**: 下载的模型会被缓存以供后续使用

### 2. 可用的模型别名

| 别名 | 完整模型名 | 描述 |
|-------|----------------|-------------|
| `qwen3` | `mlx-community/Qwen3-8B-4bit` | Qwen3 8B (默认) |
| `qwen3-1.7b` | `mlx-community/Qwen3-1.7B-4bit` | Qwen3 1.7B (轻量级) |
| `llama3.2` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | Llama 3.2 3B (默认) |
| `gemma3` | `mlx-community/gemma-3-27b-it-4bit` | Gemma 3 (VLM - 视觉语言模型) |
| `deepseek-r1` | `mlx-community/DeepSeek-R1-0528-4bit` | DeepSeek R1 (推理型) |
| `qwen2.5` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Qwen 2.5 7B |
| `whisper-large` | `openai_whisper-large-v3` | Whisper Large (语音识别) |
| `whisper-base` | `openai_whisper-base` | Whisper Base (更快，精度较低) |

### 3. 启动 API 服务

```bash
# 或不指定模型启动（可通过 API 切换）
swama serve --host 0.0.0.0 --port 28100
```

### 5. API 使用

#### 🔌 OpenAI 兼容 API

Swama 提供完全兼容 OpenAI 的 API 端点，允许您将其与现有工具和集成一起使用：

```bash
# 获取可用模型
curl http://localhost:28100/v1/models

# 使用别名的聊天补全（需要时自动下载）
curl -X POST http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {"role": "user", "content": "你好！"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# 使用 DeepSeek R1 的流式响应
curl -X POST http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [
      {"role": "user", "content": "逐步解决这个问题：240 的 15% 是多少？"}
    ],
    "stream": true
  }'

# 生成文本嵌入
curl -X POST http://localhost:28100/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "Text embeddings"],
    "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
  }'

# 音频文件转录（本地处理）
curl -X POST http://localhost:28100/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large" \
  -F "response_format=json"

# 工具调用（函数调用）
curl -X POST http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "东京的天气如何？"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "获取当前天气",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "城市名称"}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'

# 多模态支持（视觉语言模型）
curl -X POST http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "你在这张图片中看到了什么？"},
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
      }
    ]
  }'
```

## 📚 命令参考

### 模型管理

```bash
# 下载模型（支持别名和完整名称）
swama pull qwen3                    # 使用别名
swama pull whisper-large            # 下载语音识别模型
swama pull mlx-community/Qwen3-8B-4bit  # 使用完整名称

# 列出本地模型和可用别名
swama list [--format json]

# 运行推理（如果本地未找到模型会自动下载）
swama run qwen3 "你的提示词"              # 使用别名 - 自动下载！
swama run deepseek-coder "写一个Python函数"  # 另一个别名
swama run <完整模型名> <提示词> [选项]      # 使用完整名称

# 转录音频文件
swama transcribe audio.wav --model whisper-large --language zh
```

### 服务器

```bash
# 启动 API 服务器
swama serve [--host HOST] [--port PORT] [--model MODEL_ALIAS]
```

### 模型别名

Swama 支持流行模型的便捷别名。使用这些简短名称而不是完整的模型 URL：

```bash
# 不同模型系列的示例
swama run qwen3 "解释机器学习"           # Qwen3 8B
swama run llama3.2-1b "快速问题：什么是AI？"  # Llama 3.2 1B (最快)
swama run deepseek-r1 "逐步思考：2+2*3"    # DeepSeek R1 (推理型)
```

### 选项

- `--temperature <value>`: 采样温度 (0.0-2.0)
- `--top-p <value>`: 核采样参数 (0.0-1.0)
- `--max-tokens <number>`: 最大生成令牌数
- `--repetition-penalty <value>`: 重复惩罚因子

## 🔧 开发

### 依赖项

- [swift-nio](https://github.com/apple/swift-nio) - 高性能网络框架
- [swift-argument-parser](https://github.com/apple/swift-argument-parser) - 命令行参数解析
- [mlx-swift](https://github.com/ml-explore/mlx-swift) - Apple MLX Swift 绑定
- [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) - MLX Swift 示例和模型

### 构建

```bash
# 开发构建
swift build

# 发布构建
swift build -c release

# 运行测试
swift test

# 生成 Xcode 项目
swift package generate-xcodeproj
```

## 🤝 贡献

我们欢迎社区贡献！请参考以下步骤：

1. Fork 此仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

### 开发指南

- 遵循 Swift 代码风格指南
- 为新功能添加测试
- 更新相关文档
- 确保所有测试通过

## 📝 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Apple MLX](https://github.com/ml-explore/mlx) 团队提供的优秀机器学习框架
- [Swift NIO](https://github.com/apple/swift-nio) 提供的高性能网络支持
- 所有贡献者和社区成员

## 📞 支持

- 📝 [问题反馈](https://github.com/Trans-N-ai/swama/issues)
- 💬 [讨论区](https://github.com/Trans-N-ai/swama/discussions)
- 📧 邮件: info@trans-n.ai

## 🗺️ 路线图

- TODO

---

**Swama** - 为 macOS 用户带来最佳的本地 AI 体验 🚀

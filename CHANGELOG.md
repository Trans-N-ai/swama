# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Swama
- Swift-based machine learning runtime for macOS
- OpenAI-compatible API server
- Command-line interface for model management
- macOS menu bar application
- Support for LLM and VLM inference
- Model aliasing system for easy model access
- Automatic model downloading from HuggingFace
- Streaming response support

### Features
- **High Performance**: Built on Apple MLX framework, optimized for Apple Silicon
- **OpenAI Compatible API**: Standard `/v1/chat/completions` endpoint support
- **Menu Bar App**: Elegant macOS native menu bar integration
- **Command Line Tools**: Complete CLI support for model management and inference
- **Multimodal Support**: Support for both text and image inputs
- **Smart Model Management**: Automatic downloading, caching, and version management
- **Streaming Responses**: Real-time streaming text generation support
- **HuggingFace Integration**: Direct model downloads from HuggingFace Hub

### System Requirements
- macOS 14.0 or later
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15.0+ (for compilation)
- Swift 6.1+

## [v1.0.0] - 2025-06-04

### Added
- Initial public release

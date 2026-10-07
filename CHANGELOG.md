# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.5.1] - 2026-10-07

### Fixed
- The macOS app now ships `swama_SwamaCore.bundle` next to the embedded CLI. Since v2.4.0, the app crashed on its
  first inference on any Mac other than the build machine because the CLI could not load that bundle (introduced in
  #148). The v2.5.0 assets were withdrawn within minutes of publication; v2.4.0 users should upgrade.

## [2.5.0] - 2026-10-07

### Added
- Local Decisions API on `POST /v1/decisions`, following the OpenAI Decisions API (openai-openapi `4a4020d8`):
  `predicate`, `choice` and `score` questions with optional names, string or user-message input with inline
  data-URL images (up to 4), answers in question order with typed choice values and candidate confidence, and
  OpenAI-shaped `usage` (#155, #172).
- SystemOne API adapter on `POST /v1/systemone`, sharing the Decisions engine (#167), with Clef-style images (#171),
  an empty state accepted like SGLang (#168), and two-letter labels for more than 26 choices (#169).
- Image inputs for decisions at the Core and scorer layers (#170).
- `detail: "low"` on every image in a decision resizes images to 512 px for faster, lower-detail answers (#174).

### Changed
- Decision images are no longer enlarged beyond what the model needs; small images are read at their own size (#175).
- Old SGLang-format requests to `/v1/decisions` get an explicit message pointing to the OpenAI format (#174).
- Decision post-processing runs on the device, and reasoning openers are cached (#166).
- Decision prompts are tokenized before taking the exclusive model slot (#165), and repeated tokenization in
  scoring is reduced (#157).
- Waiting model operations are admitted in arrival order instead of polling (#164).
- mlx-swift-lm updated, with the tokenizer bridge isolated (#159).
- READMEs trimmed and corrected against the current code (#156).

### Fixed
- Abandoned generations are cancelled when the client disconnects (#161).
- Model ID validation no longer rejects every ID in Swift 6.4 Release builds (#160).

## [v1.0.0] - 2025-06-04

### Added
- Initial public release
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


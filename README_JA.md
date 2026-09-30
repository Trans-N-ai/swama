# Swama

[![Swift](https://img.shields.io/badge/Swift-6.2-orange.svg)](https://swift.org)
[![macOS](https://img.shields.io/badge/macOS-15.4+-blue.svg)](https://www.apple.com/macos/)
[![MLX](https://img.shields.io/badge/MLX-Swift-green.svg)](https://github.com/ml-explore/mlx-swift)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [English](README.md) | [中文](README_CN.md) | 日本語

**Swama** は Apple Silicon Mac 向けのローカル AI ランタイムです。Swift で書かれ、Apple の [MLX](https://github.com/ml-explore/mlx-swift) の上に構築されています。
言語・画像・埋め込み・音声認識・音声合成モデルを Mac 上で動かし、OpenAI 互換 API、コマンドラインツール、メニューバーアプリとして提供します。

- **OpenAI 互換 API**：チャット補完（ストリーミング、ツール呼び出し、画像入力）、Responses のステートレスなサブセット、埋め込み、音声文字起こし、音声合成（実験的）。
- **モデルエイリアス**：`swama run qwen3.5 "…"` で、初回利用時に Hugging Face から自動ダウンロードします。
- **メニューバーアプリ**：サーバーをバックグラウンドで実行し、`swama` コマンドのインストールとコンテキスト上限の設定ができます。

## 動作環境

- Apple Silicon Mac、macOS 15.4 以降
- ソースからビルドする場合のみ：Swift 6.2 ツールチェーンを含む Xcode

## インストール

**Homebrew**

```bash
brew install swama
```

**アプリのダウンロード**：[Releases](https://github.com/Trans-N-ai/swama/releases) から `Swama.dmg` を取得し、`Swama.app` をアプリケーションフォルダにドラッグして開きます。
初回起動が macOS にブロックされた場合は、**システム設定 › プライバシーとセキュリティ** で許可してください。その後メニューバーの **Install Command Line Tool…** で `swama` を PATH に追加します。

**ソースからビルド**

```bash
git clone https://github.com/Trans-N-ai/swama.git
cd swama/swama
swift build -c release
mv .build/release/swama .build/release/swama-bin   # アプリはこの名前で CLI を同梱します

cd ../swama-macos/Swama
xcodebuild -project Swama.xcodeproj -scheme Swama -configuration Release
```

## クイックスタート

```bash
swama run qwen3.5 "こんにちは！"                    # 初回利用時に自動ダウンロード
swama run qwen3.5 "この画像には何が写っていますか？" -i photo.jpg
swama serve --host 127.0.0.1 --port 28100          # API サーバー
```

```bash
curl http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5", "messages": [{"role": "user", "content": "こんにちは！"}]}'
```

`swama serve` はデフォルトで `0.0.0.0` にバインドするため、同じネットワーク上の他のマシンからも API にアクセスできます。ローカルのみで使う場合は `--host 127.0.0.1` を指定してください。

## モデル

Hugging Face 上の MLX モデルはフル ID でそのまま使えます（例：`mlx-community/Qwen3.5-9B-4bit`）。よく使うモデルには短いエイリアスがあります：

| 種類 | エイリアス（先頭がデフォルト） |
| --- | --- |
| 言語 | `qwen3.5`（35B-A3B）、`qwen3.5-0.8b` / `-2b` / `-4b` / `-9b` / `-27b` / `-122b-a10b` / `-397b-a17b`、`qwen3`、`qwen3-1.7b` / `-30b` / `-32b` / `-235b`、`qwen2.5`、`llama3.2`、`llama3.2-1b`、`llama3.3`、`gpt-oss`、`gpt-oss-120b`、`deepseek-r1`、`deepseek-r1-8b`、`deepseek-coder`、`smollm` |
| 画像 | `qwen3.5`（Qwen3.5 は全サイズで画像入力に対応）、`gemma3`、`gemma3-1b` / `-12b` / `-27b`、`qwen3-vl`、`qwen3-vl-2b` / `-8b` / `-32b` / `-30b` / `-235b`、`-thinking` 版 |
| 音声認識 | `qwen3-asr`、`qwen3-asr-1.7b`、`whisper`（large-v3-turbo）と `whisper-tiny` / `-base` / `-small` / `-medium` / `-large`、`parakeet`、`sensevoice`、`glm-asr`、`voxtral`、`canary`、`moonshine`、`nemotron-asr`、`cohere-transcribe`、`moss-transcribe-diarize`、`wav2vec2`、`mms-asr` |
| 音声合成（実験的） | `kokoro`、`orpheus`、`qwen3-tts`、`marvis`、`chatterbox`、`vyvo`、`fish-speech`、`soprano`、`pocket-tts`、`echo-tts`、`kitten-tts`、`irodori-tts`、`omnivoice`、`moss-tts`、`moss-ttsd`、`moss-tts-local` |

エイリアスとモデルの完全な対応表は [`ModelAliases.swift`](swama/Sources/SwamaKit/Model/ModelAliases.swift) にあります。FireRedASR2 もフルリポジトリ ID で利用できます。
既知の問題：2.4.0 での検証では `gemma3`（4B）と `qwen3-vl`（4B）が読み込めません（[#23](https://github.com/Trans-N-ai/swama/issues/23)）。画像入力には `qwen3.5` を使ってください。

## API

サーバーはデフォルトでポート 28100 で待ち受けます（`--port` または `SWAMA_PORT` で変更）。

| エンドポイント | 説明 |
| --- | --- |
| `GET /v1/models` | ダウンロード済みのモデル |
| `POST /v1/chat/completions` | ストリーミング（`"stream": true`）、ツール呼び出し、画像モデルへの `image_url` 入力 |
| `POST /v1/responses` | ステートレスなサブセット。対応範囲は[英語 README のサポートマトリクス](README.md#api)を参照 |
| `POST /v1/embeddings` | 埋め込みモデル（例：`mlx-community/embeddinggemma-300m-4bit`） |
| `POST /v1/audio/transcriptions` | multipart アップロード、ローカル音声認識 |
| `POST /v1/audio/speech` | 音声合成（実験的） |

```bash
# 画像入力
curl http://localhost:28100/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.5",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "何が見えますか？"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}]}]}'

# ツール呼び出し
curl http://localhost:28100/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen3.5",
  "messages": [{"role": "user", "content": "東京の天気は？"}],
  "tools": [{"type": "function", "function": {"name": "get_weather",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]}'

# 埋め込み
curl http://localhost:28100/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/embeddinggemma-300m-4bit", "input": ["Hello world"]}'

# 文字起こし
curl http://localhost:28100/v1/audio/transcriptions -F "file=@audio.wav" -F "model=qwen3-asr"

# 音声合成
curl http://localhost:28100/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"model": "kokoro", "input": "Hello from Swama", "response_format": "wav"}' --output speech.wav
```

ボイス：Orpheus `dan` `jess` `leo` `mia` `tara` `zac` `zoe`、Marvis `conversational_a` `conversational_b`、
Qwen3-TTS と VyvoTTS `en-us-1`、Kokoro のデフォルトは `af_heart`、KittenTTS は `Bella`。

## コマンドライン

| コマンド | 用途 |
| --- | --- |
| `swama run <model> <prompt>` | 単発の生成。`-i` 画像（複数可）、`-t` 温度、`--top-p`、`-n` 最大トークン数、`--repetition-penalty`、`--no-stream` |
| `swama serve` | API サーバーを起動（`--host`、`--port`）。コマンドを省略した場合もこれが実行されます |
| `swama pull <model>` | モデルまたはエイリアスをダウンロード |
| `swama list [--format json]` | ダウンロード済みのモデル |
| `swama rm <model>` | ダウンロード済みのモデルを削除 |
| `swama transcribe <audio>` | 音声をテキストに（`-m` モデル、`-l` 言語、`-f simple\|verbose`） |
| `swama create <path> -n <name>` | 手元のモデルディレクトリを名前で登録 |
| `swama logs [--follow]` | JSONL 診断ログを読む |
| `swama menubar` | メニューバーアプリとして実行 |

`run` と `serve` は `--context-limit`（デフォルト 16384 トークン）に対応しています。

環境変数：`SWAMA_PORT`（サーバーポート）、`SWAMA_MODELS`（モデルディレクトリ、デフォルト `~/.swama/models`）、`SWAMA_CONTEXT_LIMIT`、
`SWAMA_REGISTRY`（`HUGGING_FACE` または `MODEL_SCOPE`）、`SWAMA_PROMPT_CACHE=0`（プロンプトキャッシュを無効化）。

## 開発

Swift パッケージは `swama/`（`swift build`、`swift test`）、macOS アプリは `swama-macos/` にあります。手順は [CONTRIBUTING.md](CONTRIBUTING.md)、
脆弱性の報告は [SECURITY.md](SECURITY.md) を参照してください。

[mlx-swift](https://github.com/ml-explore/mlx-swift)、[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)、
[mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift)、[swift-transformers](https://github.com/huggingface/swift-transformers)、
[swift-nio](https://github.com/apple/swift-nio)、[swift-argument-parser](https://github.com/apple/swift-argument-parser) を利用しています。

## ライセンス

MIT。詳細は [LICENSE](LICENSE) を参照してください。質問やバグ報告：[Issues](https://github.com/Trans-N-ai/swama/issues) ·
[Discussions](https://github.com/Trans-N-ai/swama/discussions)。

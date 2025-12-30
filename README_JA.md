# Swama

[![Swift](https://img.shields.io/badge/Swift-6.2-orange.svg)](https://swift.org)
[![macOS](https://img.shields.io/badge/macOS-15.0+-blue.svg)](https://www.apple.com/macos/)
[![MLX](https://img.shields.io/badge/MLX-Swift-green.svg)](https://github.com/ml-explore/mlx-swift)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [English](README.md) | [中文](README_CN.md) | 日本語

**Swama** は、macOS専用に設計され、AppleのMLXフレームワーク上に構築されたピュアSwiftで書かれた高性能機械学習ランタイムです。ローカルLLM（大規模言語モデル）およびVLM（視覚言語モデル）推論のための強力で使いやすいソリューションを提供します。

## ✨ 特徴

- 🚀 **高性能**: Apple MLXフレームワーク上に構築、Apple Silicon向けに最適化
- 🔌 **OpenAI互換API**: 標準の `/v1/chat/completions`、`/v1/embeddings`、`/v1/audio/transcriptions`、および `/v1/audio/speech`（experimental）エンドポイントをサポート、Tool Calling対応
- 📱 **メニューバーアプリ**: エレガントなmacOSネイティブメニューバー統合
- 💻 **コマンドラインツール**: モデル管理と推論のための完全なCLIサポート
- 🖼️ **マルチモーダルサポート**: テキストと画像の両方の入力をサポート
- 🎤 **ローカル音声文字起こし**: Whisper内蔵音声認識（クラウド不要）
- 🔍 **テキスト埋め込み**: セマンティック検索とRAGアプリケーション用の組み込み埋め込み生成
- 📦 **スマートモデル管理**: 自動ダウンロード、キャッシュ、バージョン管理
- 🔄 **ストリーミングレスポンス**: リアルタイムストリーミングテキスト生成をサポート
- 🌍 **HuggingFace統合**: HuggingFace Hubからの直接モデルダウンロード

## 🏗️ アーキテクチャ

Swamaはモジュラーアーキテクチャ設計を採用しています：

- **SwamaKit**: すべてのビジネスロジックを含むコアフレームワークライブラリ
- **Swama CLI**: 完全なモデル管理と推論機能を提供するコマンドラインツール
- **Swama.app**: グラフィカルインターフェースとバックグラウンドサービスを備えたmacOSメニューバーアプリケーション

## 📋 システム要件

- macOS 15.0以降 (Sequoia)
- Apple Silicon (M1/M2/M3/M4)
- Xcode 16.0+ (コンパイル用)
- Swift 6.2+

## 🛠️ インストール

### 📱 ビルド済みアプリのダウンロード（推奨）

1. **最新リリースをダウンロード**
   - [Releases](https://github.com/Trans-N-ai/swama/releases) ページにアクセス
   - 最新リリースから `Swama.dmg` をダウンロード

2. **アプリのインストール**
   - `Swama.dmg` をダブルクリックしてディスクイメージをマウント
   - `Swama.app` を `Applications` フォルダにドラッグ
   - アプリケーションまたはSpotlightからSwamaを起動
   
   **注意**: 初回起動時、macOS がセキュリティ警告を表示する場合があります。この場合：
   - **システム環境設定 > セキュリティとプライバシー > 一般** に移動
   - Swama アプリメッセージの横にある **「このまま開く」** をクリック
   - またはアプリを右クリックしてコンテキストメニューから **「開く」** を選択

3. **コマンドラインツールのインストール**
   - メニューバーから Swama を開く
   - 「Install Command Line Tool…」をクリックして `swama` コマンドを PATH に追加

### 🔧 ソースからビルド（上級者向け）

ソースからビルドしたい開発者向け：

```bash
# リポジトリをクローン
git clone https://github.com/Trans-N-ai/swama.git
cd swama

# CLI ツールをビルド
cd swama
swift build -c release
mv .build/release/swama .build/release/swama-bin

# macOS アプリをビルド（Xcode が必要）
cd ../swama-macos/Swama
xcodebuild -project Swama.xcodeproj -scheme Swama -configuration Release
```

## 🚀 クイックスタート

Swama.app をインストール後、メニューバーアプリまたはコマンドラインを使用できます：

### 1. モデルエイリアスを使った即座の推論

```bash
# 長いモデル名の代わりに短いエイリアスを使用 - 必要に応じて自動ダウンロード！
swama run qwen3 "こんにちは、AI"
swama run llama3.2 "ジョークを教えて"
swama run gemma3 "この画像には何が写っていますか？" -i /path/to/image.jpg

# 従来の方法（同様に動作）
swama run mlx-community/Llama-3.2-1B-Instruct-4bit "こんにちは、元気ですか？"

# ダウンロード済みモデルの一覧表示
swama list
```

**✨ スマート機能:**
- **モデルエイリアス**: 長いURLの代わりに `qwen3`、`llama3.2`、`deepseek-r1`、`gpt-oss` などの使いやすい名前を使用
- **自動ダウンロード**: 初回使用時に自動でモデルをダウンロード - 事前に `pull` する必要なし！
- **キャッシュ管理**: ダウンロードしたモデルは将来の使用のためにキャッシュされます

### 2. 利用可能なモデルエイリアス

#### 言語モデル (LLM)

| エイリアス | 完全なモデル名 | サイズ | 説明 |
|-------|-----------------|------|-------------|
| `qwen3` | `mlx-community/Qwen3-8B-4bit` | 4.3 GB | Qwen3 8B (デフォルト) |
| `qwen3-1.7b` | `mlx-community/Qwen3-1.7B-4bit` | 938.4 MB | Qwen3 1.7B (軽量) |
| `qwen3-30b` | `mlx-community/Qwen3-30B-A3B-4bit` | 16.0 GB | Qwen3 30B (大規模) |
| `qwen3-32b` | `mlx-community/Qwen3-32B-4bit` | 17.2 GB | Qwen3 32B (超大規模) |
| `qwen3-235b` | `mlx-community/Qwen3-235B-A22B-4bit` | 123.2 GB | Qwen3 235B (超大規模) |
| `llama3.2` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | 1.7 GB | Llama 3.2 3B (デフォルト) |
| `llama3.2-1b` | `mlx-community/Llama-3.2-1B-Instruct-4bit` | 876.3 MB | Llama 3.2 1B (最速) |
| `deepseek-r1` | `mlx-community/DeepSeek-R1-0528-4bit` | 約 32 GB | DeepSeek R1 (推論モデル) |
| `deepseek-r1-8b` | `mlx-community/DeepSeek-R1-0528-Qwen3-8B-8bit` | 8.6 GB | DeepSeek R1 (Qwen3-8Bベース) |
| `qwen2.5` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | 4.0 GB | Qwen 2.5 7B |
| `gpt-oss` | `lmstudio-community/gpt-oss-20b-MLX-8bit` | 約 20 GB | GPT-OSS 20B (21B パラメータ、3.6B アクティブ) |
| `gpt-oss-120b` | `lmstudio-community/gpt-oss-120b-MLX-8bit` | 約 120 GB | GPT-OSS 120B (117B パラメータ、5.1B アクティブ) |

#### 視覚言語モデル (VLM)

| エイリアス | 完全なモデル名 | サイズ | 説明 |
|-------|-----------------|------|-------------|
| `gemma3` | `mlx-community/gemma-3-4b-it-4bit` | 3.2 GB | Gemma 3 4B (デフォルト VLM) |
| `gemma3-27b` | `mlx-community/gemma-3-27b-it-4bit` | 15.7 GB | Gemma 3 27B (大規模 VLM) |
| `qwen3-vl` | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | 約 4 GB | Qwen3-VL 4B (デフォルト VLM) |
| `qwen3-vl-2b` | `mlx-community/Qwen3-VL-2B-Instruct-4bit` | 約 2 GB | Qwen3-VL 2B (軽量) |
| `qwen3-vl-8b` | `mlx-community/Qwen3-VL-8B-Instruct-4bit` | 約 8 GB | Qwen3-VL 8B (バランス型) |

#### 音声モデル (音声認識)

| エイリアス | 完全なモデル名 | サイズ | 説明 |
|-------|-----------------|------|-------------|
| `whisper-large` | `mlx-community/whisper-large-v3-4bit` | 1.6 GB | Whisper Large v3 (最高精度) |
| `whisper-medium` | `mlx-community/whisper-medium-4bit` | 791.1 MB | Whisper Medium (バランス型) |
| `whisper-small` | `mlx-community/whisper-small-4bit` | 251.7 MB | Whisper Small (高速) |
| `whisper-base` | `mlx-community/whisper-base-4bit` | 77.2 MB | Whisper Base (より高速) |
| `whisper-tiny` | `mlx-community/whisper-tiny-4bit` | 40.1 MB | Whisper Tiny (最速) |
| `funasr` | `mlx-community/Fun-ASR-Nano-2512-4bit` | 約 200 MB | FunASR Nano (多言語) |
| `funasr-mlt` | `mlx-community/Fun-ASR-MLT-Nano-2512-4bit` | 約 200 MB | FunASR MLT (多言語転写) |

#### テキスト読み上げモデル (TTS)

| エイリアス | 完全なモデル名 | サイズ | 説明 |
|-------|-----------------|------|-------------|
| `orpheus` | `mlx-community/orpheus-3b-0.1-ft-4bit` | - | - |
| `marvis` | `Marvis-AI/marvis-tts-100m-v0.2-MLX-6bit` | - | - |
| `chatterbox` | `mlx-community/Chatterbox-TTS-q4` | - | - |
| `chatterbox-turbo` | `mlx-community/Chatterbox-Turbo-TTS-q4` | - | - |
| `outetts` | `mlx-community/Llama-OuteTTS-1.0-1B-4bit` | - | - |
| `cosyvoice2` | `mlx-community/CosyVoice2-0.5B-4bit` | - | - |
| `cosyvoice3` | `mlx-community/Fun-CosyVoice3-0.5B-2512-4bit` | - | - |

### 3. APIサービスの開始

```bash
# またはモデルを指定せずに開始（API経由で切り替え可能）
swama serve --host 0.0.0.0 --port 28100
```

### 4. API使用

#### 🔌 OpenAI互換API

SwamaはOpenAI完全互換のAPIエンドポイントを提供し、既存のツールや統合と一緒に使用できます：

注：`/v1/audio/speech` は experimental です。

```bash
# 利用可能なモデルの取得
curl http://localhost:28100/v1/models

# エイリアスを使ったチャット補完（必要に応じて自動ダウンロード）
curl -X POST http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [
      {"role": "user", "content": "こんにちは！"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'

# DeepSeek R1を使ったストリーミングレスポンス
curl -X POST http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [
      {"role": "user", "content": "段階的に解決してください：240の15%はいくつですか？"}
    ],
    "stream": true
  }'

# テキスト埋め込みの生成
curl -X POST http://localhost:28100/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "Text embeddings"],
    "model": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
  }'

# 音声ファイルの文字起こし（ローカル処理）
curl -X POST http://localhost:28100/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large" \
  -F "response_format=json"

# テキスト読み上げ（TTS、experimental）
curl -X POST http://localhost:28100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "orpheus",
    "input": "Hello from Swama TTS",
    "voice": "tara",
    "response_format": "wav"
  }' --output speech.wav

# TTSモデル: orpheus, marvis, chatterbox, chatterbox-turbo, outetts, cosyvoice2, cosyvoice3
# 音色対応モデル: orpheus, marvis
# Orpheus音色: tara, leah, jess, leo, dan, mia, zac, zoe
# Marvis音色: conversational_a, conversational_b
# CosyVoice は明示的な参照音声が必要なため、OpenAI互換エンドポイントでは未対応

# ツール呼び出し（関数呼び出し）
curl -X POST http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "東京の天気はどうですか？"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "現在の天気を取得",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "都市名"}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'

# マルチモーダルサポート（視覚言語モデル）
curl -X POST http://localhost:28100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "この画像に何が写っていますか？"},
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
      }
    ]
  }'
```

## 📚 コマンドリファレンス

### モデル管理

```bash
# モデルのダウンロード（エイリアスと完全な名前の両方をサポート）
swama pull qwen3                    # エイリアスを使用
swama pull whisper-large            # 音声認識モデルをダウンロード
swama pull mlx-community/Qwen3-8B-4bit  # 完全な名前を使用

# ローカルモデルと利用可能なエイリアスの一覧表示
swama list [--format json]

# 推論の実行（ローカルでモデルが見つからない場合は自動ダウンロード）
swama run qwen3 "あなたのプロンプト"              # エイリアスを使用 - 自動ダウンロード！
swama run deepseek-coder "Python関数を書いて"  # 別のエイリアス
swama run <完全なモデル名> <プロンプト> [オプション]      # 完全な名前を使用

# 音声ファイルの文字起こし
swama transcribe audio.wav --model whisper-large --language ja
```

### サーバー

```bash
# APIサーバーの開始
swama serve [--host HOST] [--port PORT]
```

### オプション

- `--temperature <value>`: サンプリング温度（0.0-2.0）
- `--top-p <value>`: Nucleus samplingパラメータ（0.0-1.0）
- `--max-tokens <number>`: 生成する最大トークン数
- `--repetition-penalty <value>`: 繰り返しペナルティ係数

## 🔧 開発

### 依存関係

- [swift-nio](https://github.com/apple/swift-nio) - 高性能ネットワーキングフレームワーク
- [swift-argument-parser](https://github.com/apple/swift-argument-parser) - コマンドライン引数解析
- [mlx-swift](https://github.com/ml-explore/mlx-swift) - Apple MLX Swiftバインディング
- [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) - MLX Swift言語モデル
- [mlx-swift-audio](https://github.com/DePasqualeOrg/mlx-swift-audio) - MLX Swift音声処理（Whisper、FunASR）

### ビルド

```bash
# 開発ビルド
swift build

# リリースビルド
swift build -c release

# テストの実行
swift test

# Xcodeプロジェクトの生成
swift package generate-xcodeproj
```

## 🤝 貢献

コミュニティからの貢献を歓迎しています！以下の手順に従ってください：

1. このリポジトリをフォーク
2. 機能ブランチを作成（`git checkout -b feature/amazing-feature`）
3. 変更をコミット（`git commit -m 'Add some amazing feature'`）
4. ブランチにプッシュ（`git push origin feature/amazing-feature`）
5. プルリクエストを開く

### 開発ガイドライン

- Swiftコーディングスタイルガイドラインに従う
- 新機能にはテストを追加
- 関連ドキュメントを更新
- すべてのテストが通ることを確認

## 📝 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

- 優れた機械学習フレームワークを提供してくれた[Apple MLX](https://github.com/ml-explore/mlx)チーム
- 高性能ネットワーキングサポートを提供する[Swift NIO](https://github.com/apple/swift-nio)
- すべての貢献者とコミュニティメンバー

## 📞 サポート

- 📝 [Issue Tracker](https://github.com/Trans-N-ai/swama/issues)
- 💬 [ディスカッション](https://github.com/Trans-N-ai/swama/discussions)
- 📧 Email: info@trans-n.ai

## 🗺️ ロードマップ

- TODO

---

**Swama** - macOSユーザーに最高のローカルAI体験を提供 🚀

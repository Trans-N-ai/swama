# Swama

[![Swift](https://img.shields.io/badge/Swift-6.1-orange.svg)](https://swift.org)
[![macOS](https://img.shields.io/badge/macOS-14.0+-blue.svg)](https://www.apple.com/macos/)
[![MLX](https://img.shields.io/badge/MLX-Swift-green.svg)](https://github.com/ml-explore/mlx-swift)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [English](README.md) | [中文](README_CN.md) | 日本語

**Swama** は、macOS専用に設計され、AppleのMLXフレームワーク上に構築されたピュアSwiftで書かれた高性能機械学習ランタイムです。ローカルLLM（大規模言語モデル）およびVLM（視覚言語モデル）推論のための強力で使いやすいソリューションを提供します。

## ✨ 特徴

- 🚀 **高性能**: Apple MLXフレームワーク上に構築、Apple Silicon向けに最適化
- 🔌 **OpenAI互換API**: 標準の `/v1/chat/completions`、`/v1/embeddings`、および `/v1/audio/transcriptions` エンドポイントをサポート、Tool Calling対応
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

- macOS 14.0以降
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15.0+（コンパイル用）
- Swift 6.1+

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
- **モデルエイリアス**: 長いURLの代わりに `qwen3`、`llama3.2`、`deepseek-r1` などの使いやすい名前を使用
- **自動ダウンロード**: 初回使用時に自動でモデルをダウンロード - 事前に `pull` する必要なし！
- **キャッシュ管理**: ダウンロードしたモデルは将来の使用のためにキャッシュされます

### 2. 利用可能なモデルエイリアス

| エイリアス | 完全なモデル名 | 説明 |
|-------|----------------|-------------|
| `qwen3` | `mlx-community/Qwen3-8B-4bit` | Qwen3 8B (デフォルト) |
| `qwen3-1.7b` | `mlx-community/Qwen3-1.7B-4bit` | Qwen3 1.7B (軽量) |
| `llama3.2` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | Llama 3.2 3B (デフォルト) |
| `gemma3` | `mlx-community/gemma-3-27b-it-4bit` | Gemma 3 (VLM - 視覚言語モデル) |
| `deepseek-r1` | `mlx-community/DeepSeek-R1-0528-4bit` | DeepSeek R1 (推論型) |
| `qwen2.5` | `mlx-community/Qwen2.5-7B-Instruct-4bit` | Qwen 2.5 7B |
| `whisper-large` | `openai_whisper-large-v3` | Whisper Large (音声認識) |
| `whisper-base` | `openai_whisper-base` | Whisper Base (高速、低精度) |

### 3. APIサービスの開始

```bash
# またはモデルを指定せずに開始（API経由で切り替え可能）
swama serve --host 0.0.0.0 --port 28100
```

### 5. API使用

#### 🔌 OpenAI互換API

SwamaはOpenAI完全互換のAPIエンドポイントを提供し、既存のツールや統合と一緒に使用できます：

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
- [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) - MLX Swiftサンプルとモデル

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

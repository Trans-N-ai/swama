// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "swama",
    platforms: [
        .macOS("15.4")
    ],
    products: [
        .library(
            name: "SwamaCore",
            targets: ["SwamaCore"]
        ),
        .library(
            name: "SwamaKit",
            targets: ["SwamaKit"]
        ),
        .library(
            name: "SwamaServer",
            targets: ["SwamaServer"]
        ),
        .library(
            name: "SwamaAppSupport",
            targets: ["SwamaAppSupport"]
        ),
        .executable(
            name: "swama",
            targets: ["Swama"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.3.3"),
        // Bind the MLX 0.33-compatible formal model-library release.
        // Keep FoundationModels integration disabled for the existing tokenizer bridge.
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            exact: "3.33.0",
            traits: []
        ),
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.33.0"),
        .package(
            url: "https://github.com/Blaizzy/mlx-audio-swift.git",
            exact: "0.2.0"
        ),
    ],
    targets: [
        .target(
            name: "SwamaCore",
            dependencies: [
                .target(name: "SwamaRuntime"),
            ],
            path: "Sources/SwamaCore",
            resources: [
                .copy("Resources/mlx-swift_Cmlx.bundle")
            ]
        ),
        // Transitional package-only no-Audio runtime. SwamaKit remains byte-stable through v2;
        // this derived implementation is lineage-gated and is removed with the v3 migration.
        .target(
            name: "SwamaRuntime",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/SwamaRuntime"
        ),
        .target(
            name: "SwamaKit",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
                .product(name: "MLXAudioSTT", package: "mlx-audio-swift"),
                .product(name: "MLXAudioTTS", package: "mlx-audio-swift"),
            ],
            path: "Sources/SwamaKit",
            resources: []
        ),
        .target(
            name: "SwamaServer",
            dependencies: [
                .target(name: "SwamaCore"),
                .target(name: "SwamaKit"),
                .product(name: "NIO", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXAudioCore", package: "mlx-audio-swift"),
            ],
            path: "Sources/SwamaServer"
        ),
        .target(
            name: "SwamaAppSupport",
            dependencies: [
                .target(name: "SwamaKit"),
                .target(name: "SwamaServer"),
            ],
            path: "Sources/SwamaAppSupport"
        ),
        .executableTarget(
            name: "Swama",
            dependencies: [
                .target(name: "SwamaCore"),
                .target(name: "SwamaKit"),
                .target(name: "SwamaServer"),
                .target(name: "SwamaAppSupport"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/Swama",
            sources: ["CLI"]
        ),
        // Tests for SwamaKit
        .testTarget(
            name: "SwamaKitTests",
            dependencies: [
                "SwamaRuntime",
                "SwamaCore",
                "SwamaKit",
                "SwamaServer",
                .product(name: "NIOEmbedded", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
            ]
        ),
        .testTarget(
            name: "SwamaRuntimeTests",
            dependencies: [
                "SwamaRuntime",
                "SwamaKit",
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "SwamaCoreTests",
            dependencies: ["SwamaCore"]
        ),
        .testTarget(
            name: "SwamaCLITests",
            dependencies: ["Swama", "SwamaCore", "SwamaRuntime"]
        ),
    ]
)

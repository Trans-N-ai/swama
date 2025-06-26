// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "swama",
    platforms: [
        .macOS("14.0") // Ensure macOS version is appropriate for dependencies like MLX
    ],
    products: [
        .library(
            name: "SwamaKit",
            targets: ["SwamaKit"]
        ),
        .executable(
            name: "swama",
            targets: ["Swama"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.4"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples", branch: "main"),
        .package(url: "https://github.com/mzbac/mlx.embeddings.git", branch: "main"),
        .package(url: "https://github.com/argmaxinc/WhisperKit.git", from: "0.13.0"),
    ],
    targets: [
        .target(
            name: "SwamaKit",
            dependencies: [
                .product(name: "NIO", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXVLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "mlx_embeddings", package: "mlx.embeddings"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "WhisperKit", package: "WhisperKit"),
            ],
            path: "Sources/SwamaKit",
            resources: []
        ),
        .executableTarget(
            name: "Swama",
            dependencies: [
                .target(name: "SwamaKit"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/Swama",
            sources: ["CLI"]
        ),
        // Tests for SwamaKit
        .testTarget(
            name: "SwamaKitTests",
            dependencies: ["SwamaKit"]
        ),
    ]
)

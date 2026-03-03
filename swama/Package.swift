// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "swama",
    platforms: [
        .macOS("15.4")
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
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        .package(url: "https://github.com/ml-explore/mlx-swift", branch: "main"),
        .package(url: "https://github.com/DePasqualeOrg/mlx-swift-audio.git", revision: "2a0e99ecc66f5c04b20d6d8751a4c8817cc4f621"),
    ],
    targets: [
        .target(
            name: "SwamaKit",
            dependencies: [
                .product(name: "NIO", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXAudio", package: "mlx-swift-audio"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
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

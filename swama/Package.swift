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
        // Last revision before FoundationModelsIntegration became a default trait;
        // newer commits require a newer Xcode 27 FoundationModels SDK.
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            revision: "10e0cb7442920d3f67a08e067d6670334e9dadef"
        ),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMajor(from: "0.31.4")),
        .package(
            url: "https://github.com/Blaizzy/mlx-audio-swift.git",
            revision: "cae704f53bc32a3d0b606823828fbc5bedaaf388"
        ),
    ],
    targets: [
        .target(
            name: "SwamaKit",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
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
                "SwamaKit",
                "SwamaServer",
                .product(name: "NIOEmbedded", package: "swift-nio"),
            ]
        ),
    ]
)

// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SwamaAcceptanceFixture",
    platforms: [.macOS("15.4")],
    dependencies: [
        .package(path: "../../swama"),
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm",
            revision: "10e0cb7442920d3f67a08e067d6670334e9dadef"
        )
    ],
    targets: [
        .executableTarget(
            name: "SwamaAcceptanceProbe",
            dependencies: [
                .product(name: "SwamaKit", package: "swama"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm")
            ],
            path: "Sources/SwamaAcceptanceProbe"
        )
    ]
)

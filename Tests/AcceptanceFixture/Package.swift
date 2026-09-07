// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SwamaAcceptanceFixture",
    platforms: [.macOS("15.4")],
    dependencies: [
        .package(path: "../../swama")
    ],
    targets: [
        .executableTarget(
            name: "SwamaAcceptanceProbe",
            dependencies: [
                .product(name: "SwamaCore", package: "swama")
            ],
            path: "Sources/SwamaAcceptanceProbe"
        )
    ]
)

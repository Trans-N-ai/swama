// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SwamaAcceptance",
    platforms: [.macOS("15.4")],
    products: [
        .executable(name: "swama-acceptance", targets: ["SwamaAcceptance"])
    ],
    targets: [
        .target(
            name: "SwamaAcceptanceKit",
            path: "Sources/SwamaAcceptanceKit"
        ),
        .executableTarget(
            name: "SwamaAcceptance",
            dependencies: ["SwamaAcceptanceKit"],
            path: "Sources/SwamaAcceptance"
        ),
        .testTarget(
            name: "SwamaAcceptanceTests",
            dependencies: ["SwamaAcceptanceKit"],
            path: "Tests/SwamaAcceptanceTests"
        )
    ]
)

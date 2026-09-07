// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SwamaAcceptance",
    platforms: [.macOS("15.4")],
    products: [
        .executable(name: "swama-acceptance", targets: ["SwamaAcceptance"])
    ],
    dependencies: [
        .package(
            url: "https://github.com/swiftlang/swift-syntax.git",
            revision: "79e4b74a295b6eb74a8b585e3a39d29e70c1dbd1"
        )
    ],
    targets: [
        .target(
            name: "SwamaAcceptanceKit",
            dependencies: [
                .product(name: "SwiftParser", package: "swift-syntax"),
                .product(name: "SwiftSyntax", package: "swift-syntax")
            ],
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

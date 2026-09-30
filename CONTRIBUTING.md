# Contributing to Swama

Pull requests are currently limited to repository collaborators. Bug reports and feature requests from everyone are
welcome in [Issues](https://github.com/Trans-N-ai/swama/issues); security problems go through [SECURITY.md](SECURITY.md)
instead.

## Reporting an issue

Please include the Swama version (`swama --version`), macOS version, Mac model, the command or request you ran, what you
expected, what happened, and any error output. `swama logs` prints the diagnostics log, which usually helps.

## Development setup

- Apple Silicon Mac, macOS 15.4 or later
- Xcode with the Swift 6.2 toolchain
- [SwiftFormat](https://github.com/nicklockwood/SwiftFormat) 0.56.2 (the version CI pins)

```bash
# CLI and libraries (Swift package in swama/)
cd swama
swift build -c release
swift test

# macOS app — it bundles the CLI from swama/.build/arm64-apple-macosx/release/swama-bin
mv .build/release/swama .build/release/swama-bin
cd ../swama-macos/Swama
xcodebuild -project Swama.xcodeproj -scheme Swama -configuration Release
```

## Before opening a pull request

- `swiftformat . --lint` passes from the repository root (configuration in `.swiftformat`).
- `Tools/Versioning/version.sh check` passes. Version bumps go in their own pull request; see
  [Tools/Versioning/README.md](Tools/Versioning/README.md).
- `swift test` passes locally. CI currently runs only the format and version checks — the test job is disabled — so
  running the tests is on you.
- New behaviour has tests, and user-facing changes update all three READMEs (`README.md`, `README_CN.md`,
  `README_JA.md`).
- Keep each pull request to one change, describe what changed and how you verified it, and reference related issues.
- A pull request needs an approving review from someone other than its author before it is merged.

## License

By contributing, you agree that your contributions are licensed under the MIT License.

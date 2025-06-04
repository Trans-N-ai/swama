# Contributing to Swama

Thank you for your interest in contributing to Swama! We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/swama.git`
3. Create a feature branch: `git checkout -b feature/amazing-feature`

## Development Setup

### Prerequisites

- macOS 14.0 or later
- Apple Silicon (M1/M2/M3/M4)
- Xcode 15.0+ 
- Swift 6.1+

### Building

```bash
# Build CLI tool
cd swama
swift build -c release

# Build macOS app
cd swama-macos/Swama
xcodebuild -project Swama.xcodeproj -scheme Swama -configuration Release
```

### Testing

```bash
# Run tests
swift test
```

## Code Style

- Follow Swift API Design Guidelines
- Use meaningful variable and function names
- Add comments for complex logic
- Ensure code is properly formatted

## Submitting Changes

1. Ensure your code builds without warnings
2. Add tests for new features
3. Update documentation if needed
4. Commit your changes with clear commit messages
5. Push to your fork and submit a pull request

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Keep pull requests focused on a single feature or fix

## Reporting Issues

When reporting issues, please include:

- macOS version
- Hardware (Apple Silicon model)
- Steps to reproduce
- Expected vs actual behavior
- Error messages or logs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

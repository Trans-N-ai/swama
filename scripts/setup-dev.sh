#!/bin/bash

# Swama Development Setup Script
# This script helps set up the development environment for Swama

set -e

echo "🚀 Setting up Swama development environment..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This project requires macOS. Detected: $OSTYPE"
    exit 1
fi

# Check macOS version
macos_version=$(sw_vers -productVersion)
echo "📱 macOS version: $macos_version"

# Check if Xcode is installed
if ! command -v xcodebuild &> /dev/null; then
    echo "❌ Xcode is required but not installed. Please install Xcode from the App Store."
    exit 1
fi

xcode_version=$(xcodebuild -version | head -n 1)
echo "🔨 $xcode_version"

# Check if Swift is available
if ! command -v swift &> /dev/null; then
    echo "❌ Swift is required but not found. Please ensure Xcode Command Line Tools are installed."
    exit 1
fi

swift_version=$(swift --version | head -n 1)
echo "🔶 $swift_version"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "🍺 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "🍺 Homebrew found: $(brew --version | head -n 1)"
fi

# Install SwiftFormat if not present
if ! command -v swiftformat &> /dev/null; then
    echo "📝 Installing SwiftFormat..."
    brew install swiftformat
else
    echo "📝 SwiftFormat found: $(swiftformat --version)"
fi

# Install SwiftLint if not present (optional but recommended)
if ! command -v swiftlint &> /dev/null; then
    echo "🔍 Installing SwiftLint..."
    brew install swiftlint
else
    echo "🔍 SwiftLint found: $(swiftlint version)"
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📚 Next steps:"
echo "  1. Build CLI: cd swama && swift build"
echo "  2. Run tests: cd swama && swift test"
echo "  3. Build macOS app: cd swama-macos/Swama && xcodebuild -project Swama.xcodeproj -scheme Swama build"
echo "  4. Format code: swiftformat ."
echo "  5. Lint code: swiftlint"
echo ""
echo "📖 Read CONTRIBUTING.md for more information about contributing to the project."

#!/bin/bash

# Exit immediately if a command exits with a non-zero status,
# treat unset variables as an error, and ensure pipeline errors propagate
set -euo pipefail

# ---------------------------------------------
# 🔧 CONFIGURATION
# ---------------------------------------------

APP_NAME="Swama"
APP_BUNDLE="${APP_NAME}.app"
APP_DIR="/Users/xingyue/Desktop/Swama/${APP_BUNDLE}"
OUTPUT_DIR="./release"
ZIP_PATH="${OUTPUT_DIR}/${APP_NAME}.zip"

# These should be provided as environment variables or set beforehand
DEV_ID="${CODESIGN_IDENTITY:-}"             # e.g. "Developer ID Application: John Doe (TEAMID12345)"
TEAM_ID="${TEAM_ID:-}"                      # Apple Developer Team ID
KEYCHAIN_PROFILE="${KEYCHAIN_PROFILE:-}"    # Notarization keychain profile name (configured via `xcrun notarytool store-credentials`)

# ---------------------------------------------
# 📦 PREPARE OUTPUT FOLDER
# ---------------------------------------------

echo "🚀 Packaging $APP_NAME..."

if [[ -z "$DEV_ID" || -z "$TEAM_ID" || -z "$KEYCHAIN_PROFILE" ]]; then
  echo "❌ Missing required environment variables."
  echo "   Please export CODESIGN_IDENTITY, TEAM_ID, and KEYCHAIN_PROFILE."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------
# 🔐 CODESIGN
# ---------------------------------------------

echo "🔐 Signing swama-bin..."
codesign --force --timestamp --options runtime \
  --sign "$DEV_ID" "$APP_DIR/Contents/Helpers/swama-bin"

echo "🔐 Signing full .app bundle..."
codesign --force --timestamp --options runtime --deep \
  --sign "$DEV_ID" "$APP_DIR"

# ---------------------------------------------
# 📁 ZIP FOR NOTARIZATION
# ---------------------------------------------

echo "📦 Creating ZIP archive for notarization..."
ditto -c -k --keepParent "$APP_DIR" "$ZIP_PATH"

# ---------------------------------------------
# ☁️ NOTARIZATION
# ---------------------------------------------

echo "☁️ Submitting to Apple notarization service..."
xcrun notarytool submit "$ZIP_PATH" \
  --keychain-profile "$KEYCHAIN_PROFILE" \
  --team-id "$TEAM_ID" \
  --wait

# ---------------------------------------------
# 📌 STAPLE TICKET
# ---------------------------------------------

echo "📌 Stapling notarization ticket..."
xcrun stapler staple "$APP_DIR"

# ---------------------------------------------
# 🔍 VERIFICATION (OPTIONAL)
# ---------------------------------------------

echo "🔍 Verifying notarization..."
if xcrun stapler validate "$APP_DIR" &>/dev/null; then
  echo "✅ Stapling verified successfully"
else
  echo "⚠️  Stapling verification failed (but app should still work)"
fi

if spctl --assess --type execute --verbose=4 "$APP_DIR" 2>&1 | grep -q "accepted"; then
  echo "✅ Gatekeeper verification passed"
else
  echo "⚠️  Gatekeeper verification failed"
fi

# ---------------------------------------------
# ✅ COMPLETE
# ---------------------------------------------

echo "✅ Notarization and packaging complete."
echo "App path: $APP_DIR" 
echo "ZIP archive: $ZIP_PATH"

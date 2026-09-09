#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPOSITORY_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
VERSION_FILE="$REPOSITORY_ROOT/VERSION"

CLI_FILE="$REPOSITORY_ROOT/swama/Sources/Swama/CLI/Command.swift"
KIT_DIAGNOSTICS_FILE="$REPOSITORY_ROOT/swama/Sources/SwamaKit/Diagnostics/SwamaDiagnostics.swift"
RUNTIME_DIAGNOSTICS_FILE="$REPOSITORY_ROOT/swama/Sources/SwamaRuntime/Diagnostics/SwamaDiagnostics.swift"
RUNTIME_LINEAGE_TEST_FILE="$REPOSITORY_ROOT/swama/Tests/SwamaRuntimeTests/RuntimeLineageTests.swift"
XCODE_PROJECT_FILE="$REPOSITORY_ROOT/swama-macos/Swama/Swama.xcodeproj/project.pbxproj"

usage() {
    cat <<'USAGE'
Usage:
  Tools/Versioning/version.sh current
  Tools/Versioning/version.sh check [vMAJOR.MINOR.PATCH]
  Tools/Versioning/version.sh set MAJOR.MINOR.PATCH

The VERSION file is authoritative. `set` synchronizes the five generated
product-version mirrors. `check` fails when a mirror or an optional tag differs.
USAGE
}

fail() {
    printf 'version: %s\n' "$*" >&2
    exit 1
}

validate_version() {
    printf '%s\n' "$1" | LC_ALL=C grep -Eq '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$' ||
        fail "expected MAJOR.MINOR.PATCH, got '$1'"
}

read_version() {
    [ -f "$VERSION_FILE" ] || fail "missing VERSION"
    [ "$(wc -l < "$VERSION_FILE" | tr -d ' ')" = "1" ] || fail "VERSION must contain exactly one line"

    version=$(sed -n '1p' "$VERSION_FILE")
    validate_version "$version"
    printf '%s\n' "$version"
}

assert_count() {
    file=$1
    needle=$2
    expected=$3
    actual=$(grep -F -c -- "$needle" "$file" || true)
    [ "$actual" = "$expected" ] ||
        fail "$file: expected $expected occurrence(s) of '$needle', found $actual"
}

check_mirrors() {
    version=$1
    assert_count "$CLI_FILE" "version: \"$version\"," 1
    assert_count "$KIT_DIAGNOSTICS_FILE" "?? \"$version\"" 1
    assert_count "$RUNTIME_DIAGNOSTICS_FILE" "?? \"$version\"" 1
    assert_count "$XCODE_PROJECT_FILE" "MARKETING_VERSION = $version;" 2

    kit_diagnostics_sha=$(shasum -a 256 "$KIT_DIAGNOSTICS_FILE" | awk '{ print $1 }')
    runtime_diagnostics_sha=$(shasum -a 256 "$RUNTIME_DIAGNOSTICS_FILE" | awk '{ print $1 }')
    assert_count "$RUNTIME_LINEAGE_TEST_FILE" "legacySHA256: \"$kit_diagnostics_sha\"" 1
    assert_count "$RUNTIME_LINEAGE_TEST_FILE" "runtimeSHA256: \"$runtime_diagnostics_sha\"" 1
}

replace_fixed() {
    file=$1
    old_text=$2
    new_text=$3
    OLD_TEXT=$old_text NEW_TEXT=$new_text perl -0pi -e 's/\Q$ENV{OLD_TEXT}\E/$ENV{NEW_TEXT}/g' "$file"
}

check_tag() {
    version=$1
    requested_tag=${2:-}

    if [ -z "$requested_tag" ] && [ "${GITHUB_REF_TYPE:-}" = "tag" ]; then
        requested_tag=${GITHUB_REF_NAME:-}
    fi

    if [ -n "$requested_tag" ]; then
        [ "$requested_tag" = "v$version" ] ||
            fail "tag '$requested_tag' does not match VERSION '$version' (expected 'v$version')"
    fi
}

command=${1:-}
case "$command" in
current)
    [ "$#" = "1" ] || {
        usage >&2
        exit 2
    }
    read_version
    ;;
check)
    [ "$#" -le "2" ] || {
        usage >&2
        exit 2
    }
    current_version=$(read_version)
    check_mirrors "$current_version"
    check_tag "$current_version" "${2:-}"
    printf 'Version %s is consistent across all five product mirrors.\n' "$current_version"
    ;;
set)
    [ "$#" = "2" ] || {
        usage >&2
        exit 2
    }
    next_version=$2
    validate_version "$next_version"
    current_version=$(read_version)
    check_mirrors "$current_version"

    if [ "$next_version" != "$current_version" ]; then
        current_kit_diagnostics_sha=$(shasum -a 256 "$KIT_DIAGNOSTICS_FILE" | awk '{ print $1 }')
        current_runtime_diagnostics_sha=$(shasum -a 256 "$RUNTIME_DIAGNOSTICS_FILE" | awk '{ print $1 }')

        replace_fixed "$CLI_FILE" "version: \"$current_version\"," "version: \"$next_version\","
        replace_fixed "$KIT_DIAGNOSTICS_FILE" "?? \"$current_version\"" "?? \"$next_version\""
        replace_fixed "$RUNTIME_DIAGNOSTICS_FILE" "?? \"$current_version\"" "?? \"$next_version\""
        replace_fixed "$XCODE_PROJECT_FILE" "MARKETING_VERSION = $current_version;" "MARKETING_VERSION = $next_version;"

        next_kit_diagnostics_sha=$(shasum -a 256 "$KIT_DIAGNOSTICS_FILE" | awk '{ print $1 }')
        next_runtime_diagnostics_sha=$(shasum -a 256 "$RUNTIME_DIAGNOSTICS_FILE" | awk '{ print $1 }')
        replace_fixed "$RUNTIME_LINEAGE_TEST_FILE" \
            "legacySHA256: \"$current_kit_diagnostics_sha\"" \
            "legacySHA256: \"$next_kit_diagnostics_sha\""
        replace_fixed "$RUNTIME_LINEAGE_TEST_FILE" \
            "runtimeSHA256: \"$current_runtime_diagnostics_sha\"" \
            "runtimeSHA256: \"$next_runtime_diagnostics_sha\""
        printf '%s\n' "$next_version" > "$VERSION_FILE"
    fi

    check_mirrors "$next_version"
    printf 'Updated product version from %s to %s.\n' "$current_version" "$next_version"
    ;;
*)
    usage >&2
    exit 2
    ;;
esac

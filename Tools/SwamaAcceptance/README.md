# Swama acceptance harness

This directory is a standalone Swift package that measures Swama from outside
the product package. It does not import or depend on `SwamaKit`.

The companion `Tests/AcceptanceFixture` package is the library consumer. It
depends on `../../swama` exactly as another Swift package would and exposes the
small `SwamaAcceptanceProbe` executable used by the harness.

This split is deliberate:

- candidate Swift compilation or runtime failure cannot take down an already
  built harness;
- the probe cannot use `internal` or `package` Swama APIs;
- the production `swama/Package.swift` contains no acceptance-only target;
- all process control, HTTP streaming, RSS sampling, hashing, report sealing,
  architecture checks and comparisons remain Swift-native.

## Build the referee first

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift build \
  --package-path Tools/SwamaAcceptance \
  --configuration release

HARNESS="$(DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcrun swift build \
  --package-path Tools/SwamaAcceptance \
  --configuration release \
  --show-bin-path)/swama-acceptance"
```

Run the resulting binary directly. Do not use `swift run` for a candidate: the
binary identity is part of every report and must already be frozen before the
candidate build starts.

## Commands

```bash
"$HARNESS" architecture --repo-root "$PWD"

"$HARNESS" baseline \
  --repo-root "$PWD" \
  --output /tmp/swama-baseline.json

"$HARNESS" compare \
  --repo-root "$PWD" \
  --baseline /path/held-by-reviewer.json \
  --candidate /tmp/swama-candidate.json
```

The default Swift/Xcode build path is `/Applications/Xcode.app`. The default
Metal compiler path is `/Applications/Xcode-beta.app`; they can be overridden
independently with `--developer-dir` and `--metal-developer-dir`.

The harness builds the current resolved `mlx-swift` Metal sources into a new
temporary directory and copies only that just-produced library beside each
executable. It never searches old build directories for a metallib.

## Report trust boundary

Reports contain a canonical JSON self-hash. This detects accidental edits and
truncation; it is not authentication because anyone can edit and reseal a
report. The baseline of record must therefore be generated and retained by an
independent reviewer. An author-generated baseline cannot replace it.

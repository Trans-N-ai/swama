# Product versioning

`VERSION` is the single authored Swama product version. The version tool keeps
the CLI, diagnostics fallbacks, both macOS app build configurations, and the
diagnostics source-lineage seals in sync.

Prepare a version bump from the repository root:

```sh
Tools/Versioning/version.sh set 2.4.0
Tools/Versioning/version.sh check
```

Review and merge the resulting version-only change before creating a tag. On
the merged commit, verify the intended tag and create it through the normal
signed release process:

```sh
Tools/Versioning/version.sh check v2.4.0
```

CI runs the same check for every pull request, branch push, and `v*` tag. A tag
whose name differs from `v$(cat VERSION)` fails closed. `CURRENT_PROJECT_VERSION`
is a separate build number and is intentionally not changed by this tool.

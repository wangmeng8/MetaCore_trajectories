#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${METACOREBENCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HLE_REPO_DIR="${HLE_WITH_TOOLS_REPO_DIR:-$ROOT_DIR/.external/hle_with_tools}"
PATCH_FILE="$ROOT_DIR/patches/hle_with_tools/outcome_caa_agent_extra_body.patch"

[[ -f "$HLE_REPO_DIR/hle_eval/run_agent_predictions.py" ]] || {
  printf 'ERROR: expected the restored patched HLE source at %s\n' "$HLE_REPO_DIR" >&2
  exit 2
}
[[ -f "$PATCH_FILE" ]] || {
  printf 'ERROR: HLE patch not found: %s\n' "$PATCH_FILE" >&2
  exit 2
}

# Migration archives omit .git. Do not accidentally discover a parent checkout
# and silently skip these paths when git apply runs from restored sources.
export GIT_CEILING_DIRECTORIES="$(cd "$HLE_REPO_DIR/.." && pwd)"

if git -C "$HLE_REPO_DIR" apply --reverse --check "$PATCH_FILE" >/dev/null 2>&1; then
  printf 'HLE Outcome-CAA agent extra-body patch already applied.\n'
elif git -C "$HLE_REPO_DIR" apply --check "$PATCH_FILE" >/dev/null 2>&1; then
  git -C "$HLE_REPO_DIR" apply "$PATCH_FILE"
  printf 'Applied HLE Outcome-CAA agent extra-body patch.\n'
else
  printf 'ERROR: HLE patch does not apply cleanly; inspect local HLE changes first.\n' >&2
  exit 2
fi

printf 'Main agent steering is now isolated by HLE_WITH_TOOLS_AGENT_EXTRA_BODY_JSON.\n'
printf 'Auxiliary requests preserve Vanilla fields and reject steering.\n'

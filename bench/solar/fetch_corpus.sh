#!/usr/bin/env bash
# Fetch the pinned Solidity corpora (shallow clones). Does not vendor them into git.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/pins.env"

DEST="${CORPUS_DIR:-"$ROOT/.corpus"}"
mkdir -p "$DEST"

clone_pin() {
  local url="$1" dest="$2" rev="$3" expected="${4:-}"
  if [[ -d "$dest/.git" ]]; then
    if [[ "$(git -C "$dest" rev-parse HEAD)" == "$expected" && -n "$expected" ]]; then
      echo "already at $expected  $dest"
      return 0
    fi
  fi
  rm -rf "$dest"
  # Tags (v1.0.0, v5.7.0, ...) can be cloned with --branch. Bare commits need a fetch.
  if git clone --depth 1 --branch "$rev" "$url" "$dest" 2>/dev/null; then
    :
  else
    git clone --depth 1 "$url" "$dest"
    git -C "$dest" fetch --depth 1 origin "$rev"
    git -C "$dest" checkout --detach "$rev"
  fi
  local got
  got="$(git -C "$dest" rev-parse HEAD)"
  echo "checked out $got  $dest  (requested $rev)"
  if [[ -n "$expected" && "$got" != "$expected" ]]; then
    echo "error: expected commit $expected but got $got" >&2
    exit 1
  fi
}

clone_pin "$OZ_REPO" "$DEST/openzeppelin-contracts" "$OZ_REV" "$OZ_COMMIT"
clone_pin "$UNISWAP_V3_REPO" "$DEST/v3-core" "$UNISWAP_V3_REV" "$UNISWAP_V3_COMMIT"
clone_pin "$UNISWAP_V4_REPO" "$DEST/v4-core" "$UNISWAP_V4_REV" "$UNISWAP_V4_COMMIT"
clone_pin "$SOLMATE_REPO" "$DEST/solmate" "$SOLMATE_COMMIT" "$SOLMATE_COMMIT"

echo "corpus ready under $DEST"

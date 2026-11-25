#!/usr/bin/env bash
# Build pinned Solar from source and download the pinned solc static binary.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/pins.env"

TOOLS="${TOOLS_DIR:-"$ROOT/.tools"}"
mkdir -p "$TOOLS"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

if ! need_cmd hyperfine; then
  if need_cmd apt-get && need_cmd sudo; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq hyperfine time
  elif need_cmd cargo; then
    cargo install hyperfine --locked
  else
    echo "error: install hyperfine (https://github.com/sharkdp/hyperfine) and GNU time" >&2
    exit 1
  fi
fi
if ! need_cmd cargo || ! need_cmd rustc; then
  echo "error: Rust (cargo, rustc) is required to build Solar" >&2
  exit 1
fi

SOLC_BIN="$TOOLS/solc"
if [[ ! -x "$SOLC_BIN" ]] || ! "$SOLC_BIN" --version 2>/dev/null | grep -q "$SOLC_VERSION"; then
  echo "downloading solc $SOLC_VERSION"
  curl -fsSL "$SOLC_URL" -o "$SOLC_BIN"
  chmod +x "$SOLC_BIN"
fi
"$SOLC_BIN" --version

SOLAR_SRC="$TOOLS/solar-src"
SOLAR_BIN="$TOOLS/solar"
need_build=1
if [[ -x "$SOLAR_BIN" ]]; then
  if "$SOLAR_BIN" --version 2>/dev/null | grep -q "$SOLAR_REV"; then
    need_build=0
  fi
fi
# Reuse a prebuilt binary from SOLAR_PREBUILT if it matches the pin.
if [[ "$need_build" -eq 1 && -n "${SOLAR_PREBUILT:-}" && -x "${SOLAR_PREBUILT}" ]]; then
  if "$SOLAR_PREBUILT" --version 2>/dev/null | grep -q "$SOLAR_REV"; then
    ln -sfn "$SOLAR_PREBUILT" "$SOLAR_BIN"
    need_build=0
  fi
fi
if [[ "$need_build" -eq 1 ]]; then
  if [[ ! -d "$SOLAR_SRC/.git" ]]; then
    rm -rf "$SOLAR_SRC"
    git clone --depth 1 "$SOLAR_REPO" "$SOLAR_SRC"
  fi
  git -C "$SOLAR_SRC" fetch --depth 1 origin "$SOLAR_REV"
  git -C "$SOLAR_SRC" checkout --detach "$SOLAR_REV"
  echo "building solar @ $SOLAR_REV (release)"
  cargo build --release --locked --bin solar --manifest-path "$SOLAR_SRC/Cargo.toml"
  ln -sfn "$SOLAR_SRC/target/release/solar" "$SOLAR_BIN"
fi
"$SOLAR_BIN" --version

echo "tools ready: $SOLAR_BIN  $SOLC_BIN"

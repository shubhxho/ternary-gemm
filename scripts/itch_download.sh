#!/usr/bin/env bash
# Download one public Nasdaq TotalView-ITCH 5.0 sample day and cut the replay slice from it.
#
#   scripts/itch_download.sh                      # 2019-01-30, AAPL,MSFT,INTC,CSCO,AMD -> data/itch/
#   ITCH_FILE=08302019.NASDAQ_ITCH50.gz SYMBOLS=AAPL,SPY scripts/itch_download.sh
#
# The full-day file is ~4.8 GB gzip (~12 GB raw, ~368M messages) and is NOT committed. The slice
# is a valid framed ITCH file with system events, the stock-directory entries and every message
# for the chosen symbols (~220 MB / 7.2M messages for the default five).
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${ITCH_DATA:-data/itch}
FILE=${ITCH_FILE:-01302019.NASDAQ_ITCH50.gz}
SYMBOLS=${SYMBOLS:-AAPL,MSFT,INTC,CSCO,AMD}
URL="https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/$FILE"
mkdir -p "$DATA"
if [ ! -s "$DATA/$FILE" ]; then
  echo "downloading $URL"
  curl -fL --retry 3 -C - -o "$DATA/$FILE.part" "$URL"
  mv "$DATA/$FILE.part" "$DATA/$FILE"
fi
cargo build --release --bin itch --no-default-features --features avx2
SLICE="$DATA/slice-${FILE%%.*}-${SYMBOLS//,/_}.itch"
gzip -dc "$DATA/$FILE" | ./target/release/itch filter --symbols "$SYMBOLS" --out "$SLICE"
ln -sf "$(basename "$SLICE")" "$DATA/slice.itch"
sha256sum "$SLICE"
echo "slice: $SLICE (linked as $DATA/slice.itch)"

#!/usr/bin/env bash
# Benchmark script: runs both Python and Rust microgpt implementations and compares timings.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
RUNS=${1:-3}

echo "=== microGPT Benchmark ==="
echo "Runs per implementation: $RUNS"
echo ""

# Ensure input.txt exists
if [ ! -f "$ROOT/rust/input.txt" ]; then
    echo "Downloading input.txt..."
    curl -sL https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt \
        -o "$ROOT/rust/input.txt"
fi
cp "$ROOT/rust/input.txt" "$ROOT/python/input.txt" 2>/dev/null || true

# Build Rust version
echo "Building Rust (--release)..."
cd "$ROOT/rust"
cargo build --release 2>&1 | tail -1
RUST_BIN="$ROOT/rust/target/release/microgpt"
echo ""

# --- Python benchmarks ---
echo "=== Python (microgpt.py) ==="
PY_TIMES=()
for i in $(seq 1 "$RUNS"); do
    cd "$ROOT/python"
    START=$(date +%s%N)
    python3 microgpt.py 2>&1 | tail -3
    END=$(date +%s%N)
    ELAPSED=$(( (END - START) / 1000000 ))
    PY_TIMES+=("$ELAPSED")
    echo "  Run $i: ${ELAPSED}ms"
done
echo ""

# --- Rust benchmarks ---
echo "=== Rust (microgpt) ==="
RUST_TIMES=()
for i in $(seq 1 "$RUNS"); do
    cd "$ROOT/rust"
    START=$(date +%s%N)
    "$RUST_BIN" 2>&1 | tail -3
    END=$(date +%s%N)
    ELAPSED=$(( (END - START) / 1000000 ))
    RUST_TIMES+=("$ELAPSED")
    echo "  Run $i: ${ELAPSED}ms"
done
echo ""

# --- Summary ---
echo "=== Results ==="
# Sort and pick median
IFS=$'\n' PY_SORTED=($(sort -n <<<"${PY_TIMES[*]}")); unset IFS
IFS=$'\n' RUST_SORTED=($(sort -n <<<"${RUST_TIMES[*]}")); unset IFS
MID=$(( RUNS / 2 ))
PY_MEDIAN=${PY_SORTED[$MID]}
RUST_MEDIAN=${RUST_SORTED[$MID]}
SPEEDUP=$(echo "scale=1; $PY_MEDIAN / $RUST_MEDIAN" | bc)

echo "Python median: ${PY_MEDIAN}ms"
echo "Rust median:   ${RUST_MEDIAN}ms"
echo "Speedup:       ${SPEEDUP}x"

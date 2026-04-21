#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

shopt -s nullglob
CONFIGS=(configs/*.yaml)
shopt -u nullglob

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "No .yaml files found in configs/"
    exit 1
fi

TOTAL=${#CONFIGS[@]}
OK=0
FAILED=0
FAILED_LIST=()

echo "Found $TOTAL config(s):"
for c in "${CONFIGS[@]}"; do
    echo "  - $c"
done
echo

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    idx=$((i + 1))

    echo "=================================================================="
    echo "[$idx/$TOTAL] Running $config"
    echo "=================================================================="

    SECONDS=0
    python run_experiments.py "$config"
    status=$?
    elapsed=$SECONDS

    if [ $status -eq 0 ]; then
        OK=$((OK + 1))
        echo ">>> OK: $config (${elapsed}s)"
    else
        FAILED=$((FAILED + 1))
        FAILED_LIST+=("$config")
        echo ">>> FAILED (exit=$status): $config (${elapsed}s)"
    fi
    echo
done

echo "=================================================================="
echo "Summary: $OK ok, $FAILED failed (out of $TOTAL)"
if [ $FAILED -gt 0 ]; then
    echo "Failed configs:"
    for f in "${FAILED_LIST[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
exit 0

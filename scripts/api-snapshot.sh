#!/usr/bin/env bash
# Public API snapshot of fusion-symbols.
#
#   scripts/api-snapshot.sh --write   regenerate api/*.txt
#   scripts/api-snapshot.sh --check   regenerate into a temporary directory and
#                                     diff against api/; exit 1 on any difference
#
# Needs `cargo-public-api` 0.52.0 and the pinned nightly below (rustdoc JSON);
# the rendering of rustdoc JSON changes between nightlies, so CI installs the
# same dated toolchain (`.github/workflows/ci.yml`, `api`). Override with
# API_SNAPSHOT_TOOLCHAIN only to try a newer nightly, then re-pin both. A
# snapshot is review evidence, not a proof of behaviour.
set -euo pipefail
cd "$(dirname "$0")/.."

toolchain="${API_SNAPSHOT_TOOLCHAIN:-nightly-2026-09-19}"

mode="${1:-}"
case "$mode" in
--write | --check) ;;
*)
    echo "usage: $0 --write | --check" >&2
    exit 2
    ;;
esac

# Format: <file-suffix> <features ("-" for the default set)>. `lie-generated`
# and `anyonwiki-unitary` add data and capability, not items, so the default
# set and the set with every feature cover the surface.
COMBOS=(
    "default -"
    "all disk-cache,anyonwiki-unitary"
)

if [ "$mode" = --write ]; then
    out_dir=api
else
    out_dir=$(mktemp -d)
    trap 'rm -rf "$out_dir"' EXIT
fi
mkdir -p "$out_dir"

for combo in "${COMBOS[@]}"; do
    read -r suffix features <<<"$combo"
    args=(-ss)
    if [ "$features" != - ]; then
        args+=(--features "$features")
    fi
    if [ "$suffix" = default ]; then
        file="$out_dir/fusion-symbols.txt"
    else
        file="$out_dir/fusion-symbols+$suffix.txt"
    fi
    echo "api-snapshot: fusion-symbols ($suffix)" >&2
    # `#[doc(hidden)]` items are reachable public API (the token-guarded
    # category hooks included), so they are snapshotted too; cargo-public-api
    # omits them unless rustdoc documents them.
    RUSTUP_TOOLCHAIN="$toolchain" RUSTDOCFLAGS="-Zunstable-options --document-hidden-items" \
        cargo public-api "${args[@]}" >"$file"
done

if [ "$mode" = --check ]; then
    status=0
    for file in "$out_dir"/*.txt; do
        name=$(basename "$file")
        if [ ! -f "api/$name" ]; then
            echo "api-snapshot: api/$name is missing (run $0 --write)" >&2
            status=1
            continue
        fi
        if ! diff -u "api/$name" "$file"; then
            status=1
        fi
    done
    for file in api/*.txt; do
        name=$(basename "$file")
        if [ ! -f "$out_dir/$name" ]; then
            echo "api-snapshot: api/$name is stale (no such feature combination)" >&2
            status=1
        fi
    done
    if [ "$status" -ne 0 ]; then
        echo "api-snapshot: the public API differs from api/; review the diff and run $0 --write" >&2
        exit 1
    fi
    echo "api-snapshot: ok" >&2
fi

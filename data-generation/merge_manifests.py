"""
merge_manifests.py
──────────────────
Run this after all Condor jobs complete to merge partial manifests
into the final causal_manifest.json.

    python merge_manifests.py
    python merge_manifests.py --check   # just report counts, don't write
"""

import argparse
import glob
import json
import os
import re

MANIFEST_DIR  = "manifests"
MANIFEST_PATH = "causal_manifest.json"


def _chunk_id(path):
    m = re.search(r"manifest_(\d+)\.json$", path)
    return int(m.group(1)) if m else -1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true",
                   help="Report counts without writing the merged manifest.")
    args = p.parse_args()

    partial_files = sorted(
        glob.glob(os.path.join(MANIFEST_DIR, "manifest_*.json")),
        key=_chunk_id,
    )

    if not partial_files:
        print(f"No partial manifests found in {MANIFEST_DIR}/")
        return

    all_entries = []
    for path in partial_files:
        with open(path) as fh:
            entries = json.load(fh)
        all_entries.extend(entries)
        print(f"  {path}: {len(entries)} entries")

    print(f"\nTotal entries: {len(all_entries)} from {len(partial_files)} partial manifests")

    # Check for missing chunks by comparing against expected job count
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from generate_data_causal import total_jobs
        expected = total_jobs()
        present = len(partial_files)
        if present < expected:
            print(f"WARNING: only {present}/{expected} chunk manifests present — "
                  f"{expected - present} jobs may not have completed")
        else:
            print(f"All {expected} chunk manifests present")
    except Exception:
        pass

    # Deduplicate by abaf path — last occurrence wins (handles resubmitted jobs)
    seen = {}
    duplicates = 0
    for entry in all_entries:
        key = entry["abaf"]
        if key in seen:
            duplicates += 1
        seen[key] = entry

    if duplicates:
        print(f"WARNING: {duplicates} duplicate entries found (same abaf path) — deduplicating")
        all_entries = list(seen.values())

    if args.check:
        print("(--check mode: not writing merged manifest)")
        return

    with open(MANIFEST_PATH, "w") as fh:
        json.dump(all_entries, fh, indent=2)
    print(f"Merged manifest written: {MANIFEST_PATH}  ({len(all_entries)} entries)")


if __name__ == "__main__":
    main()

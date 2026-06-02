"""
merge_tier_labels.py — Consolidate per-ABAF tier label JSONs into one file.

Reads every {abaf_stem}.json written by tier_labels.py (chunk build) and
merges them into a single dataset/tier_labels.json dict:

    {abaf_stem: {assumption_name: tier_str, ...}, ...}

The key is the ABAF file stem (e.g. "causal_er_n3_d0.3_mna_a0.01_i45_full"),
NOT instance_id, because the same base instance can appear with multiple probe
roles (initial_full, easy_sat, boundary_sat, boundary_unsat) each mapping to a
different ABAF file.  run_config.py looks up tiers via Path(entry["abaf"]).stem.

Also verifies against the manifest (expected count = 17,032) and prints a
per-tier summary with counts of missing/errored instances.

Usage
-----
    python scripts-sweep/merge_tier_labels.py
    python scripts-sweep/merge_tier_labels.py \\
        --tier-dir dataset/tier_labels/ \\
        --manifest causal_manifest.json \\
        --out      dataset/tier_labels.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier-dir", default=str(REPO_ROOT / "dataset" / "tier_labels"))
    p.add_argument("--manifest", default=str(REPO_ROOT / "causal_manifest.json"))
    p.add_argument("--out",      default=str(REPO_ROOT / "dataset" / "tier_labels.json"))
    args = p.parse_args()

    tier_dir = Path(args.tier_dir)
    out_path = Path(args.out)

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    # expected keys = abaf file stems (unique per manifest entry)
    expected_keys = {Path(e["abaf"]).stem for e in manifest}
    print(f"Manifest entries:   {len(manifest)}")
    print(f"Unique abaf stems:  {len(expected_keys)}")

    # Collect per-instance files (skip chunk report files)
    instance_files = sorted(
        f for f in tier_dir.glob("*.json")
        if not f.name.startswith("tier_chunk_")
    )
    found_keys = {f.stem for f in instance_files}
    print(f"Tier files found:   {len(found_keys)}")

    missing = expected_keys - found_keys
    extra   = found_keys - expected_keys
    if missing:
        print(f"MISSING ({len(missing)}): {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        print(f"EXTRA   ({len(extra)}):   {sorted(extra)[:5]}")

    # Merge
    merged: dict = {}
    tier_counter: Counter = Counter()
    n_errors = 0

    for f in instance_files:
        if f.stem not in expected_keys:
            continue
        try:
            with open(f) as fh:
                tiers = json.load(fh)
            merged[f.stem] = tiers
            tier_counter.update(tiers.values())
        except Exception as exc:
            print(f"  ERROR reading {f.name}: {exc}")
            n_errors += 1

    print(f"\nMerged:  {len(merged)} instances  ({n_errors} read errors)")
    print(f"Missing: {len(missing)}")
    print()
    print("Tier distribution across all assumptions:")
    total = sum(tier_counter.values())
    for tier in ("skeptical", "credulous", "rejected", "no_ext"):
        n = tier_counter.get(tier, 0)
        pct = 100 * n / total if total else 0
        print(f"  {tier:<12}  {n:>9,}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<12}  {total:>9,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(merged, fh)
    print(f"\nSaved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")

    if missing:
        print(f"\nWARNING: {len(missing)} instances missing — rerun incomplete chunks.")
        sys.exit(1)


if __name__ == "__main__":
    main()

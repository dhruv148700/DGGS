"""
filter_splits.py
────────────────
Remove probe entries whose ABA framework exceeds a computational size threshold
from the train/val/test split manifests.

MOTIVATION:
    n_nodes=6 BA graphs can have up to ~20,000 atoms and ~7,600 assumptions.
    Constructing DGL heterographs for these takes tens of minutes per entry,
    and training GNN forward passes over graphs of that size is prohibitively
    slow.  We cap at n_atoms <= 7,000, which removes ~5.8% of the dataset
    (987 / 17,032 entries), almost entirely from the heavy tail of n_nodes=6
    BA graphs.  The bulk of n_nodes=6 data (~85%) is retained.

WHAT CHANGES:
    splits/train_manifest.json  — filtered in place
    splits/val_manifest.json    — filtered in place
    splits/test_manifest.json   — filtered in place
    splits/dropped_entries.json — written: full manifest dicts for every
                                  dropped entry, so configs can be audited

WHAT DOES NOT CHANGE:
    causal_manifest.json        — permanent record of all generated data;
                                  intentionally left unfiltered

USAGE:
    python filter_splits.py                    # apply default threshold (7000)
    python filter_splits.py --threshold 10000  # different threshold
    python filter_splits.py --check            # dry run, print counts only
"""

import argparse
import json
import os

SPLITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")
DEFAULT_THRESHOLD = 7000


def filter_manifest(path: str, threshold: int, check: bool) -> tuple:
    """Load a manifest, split into kept/dropped by n_atoms threshold.

    Args:
        path:      path to the manifest JSON file
        threshold: entries with n_atoms > threshold are dropped
        check:     if True, print results but do not write

    Returns:
        (n_original, n_kept, n_dropped, dropped_entries)
    """
    with open(path) as fh:
        entries = json.load(fh)

    kept    = [e for e in entries if e["n_atoms"] <= threshold]
    dropped = [e for e in entries if e["n_atoms"] >  threshold]

    if not check:
        with open(path, "w") as fh:
            json.dump(kept, fh, indent=2)

    return len(entries), len(kept), len(dropped), dropped


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold",  type=int, default=DEFAULT_THRESHOLD,
                   help=f"Drop entries with n_atoms > threshold (default: {DEFAULT_THRESHOLD})")
    p.add_argument("--splits-dir", default=SPLITS_DIR,
                   help="Directory containing split manifests (default: splits/)")
    p.add_argument("--check",      action="store_true",
                   help="Dry run: print counts without modifying any files")
    args = p.parse_args()

    all_dropped = []
    total_orig = total_kept = total_dropped = 0

    for split in ("train", "val", "test"):
        path = os.path.join(args.splits_dir, f"{split}_manifest.json")
        n_orig, n_kept, n_dropped, dropped = filter_manifest(
            path, args.threshold, args.check
        )

        # Summarise which graph_type / n_nodes configs were dropped
        from collections import Counter
        by_gt    = Counter(e["graph_type"] for e in dropped)
        by_nodes = Counter(e["n_nodes"]    for e in dropped)

        status = "[DRY RUN] " if args.check else ""
        print(
            f"{status}{split}: {n_orig} -> {n_kept} kept, {n_dropped} dropped "
            f"| by graph_type: {dict(by_gt)} | by n_nodes: {dict(by_nodes)}"
        )

        all_dropped.extend(dropped)
        total_orig    += n_orig
        total_kept    += n_kept
        total_dropped += n_dropped

    print(
        f"\nTotal: {total_orig} -> {total_kept} kept, {total_dropped} dropped "
        f"({100 * total_dropped / total_orig:.1f}%)"
    )

    # Write all dropped entries to a single audit file so configs can be
    # documented and cross-referenced against causal_manifest.json later
    if not args.check:
        dropped_path = os.path.join(args.splits_dir, "dropped_entries.json")
        with open(dropped_path, "w") as fh:
            json.dump(
                {
                    "threshold_field":  "n_atoms",
                    "threshold_value":  args.threshold,
                    "total_dropped":    total_dropped,
                    "entries":          all_dropped,
                },
                fh, indent=2,
            )
        print(f"\nDropped entries written to: {dropped_path}")
        print("Split manifests updated in place.")
        print("causal_manifest.json was NOT modified (permanent full record).")
    else:
        print("\n(--check mode: no files were modified)")


if __name__ == "__main__":
    main()

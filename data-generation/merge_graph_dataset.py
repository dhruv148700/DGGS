"""
merge_graph_dataset.py
──────────────────────
Merge the per-chunk .bin / metadata / failures files produced by
build_causal_splits_dataset.py into final per-split datasets.

Run this once all Condor jobs have completed.

    python merge_graph_dataset.py                # merge all three splits
    python merge_graph_dataset.py --split train  # merge one split
    python merge_graph_dataset.py --check        # report counts, don't write

OUTPUT (written to --out-dir, default: splits/):
    {split}.bin               — merged DGL heterograph list
    {split}_metadata.json     — parallel list of manifest dicts (same order)
    {split}_failures.json     — union of all chunk failure records

NOTES:
  • Chunks are sorted by (split-order, start_idx) to preserve the original
    manifest order, which matters for the metadata/graph alignment guarantee.
  • The script verifies len(graphs) == len(metadata) before writing.
  • Missing chunks are reported with a clear warning; the merge continues
    with what is present so you can inspect partial results.
"""

import argparse
import glob
import json
import logging
import os
import re
import sys

REPO_ROOT  = os.path.abspath(os.path.dirname(__file__))
SPLITS_DIR = os.path.join(REPO_ROOT, "splits")
CHUNKS_DIR = os.path.join(SPLITS_DIR, "chunks")

sys.path.insert(0, REPO_ROOT)

import dgl

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

_SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def _chunk_sort_key(path: str) -> tuple:
    """Sort key: (split_order, start_idx).  e.g. 'train_00000_00200.bin' → (0, 0)."""
    name = os.path.basename(path)
    m = re.match(r"(train|val|test)_(\d+)_(\d+)", name)
    if not m:
        return (99, 0)
    return (_SPLIT_ORDER.get(m.group(1), 99), int(m.group(2)))


def _expected_chunk_count(split: str, manifest_dir: str, chunk_size: int) -> int:
    path = os.path.join(manifest_dir, f"{split}_manifest.json")
    if not os.path.exists(path):
        return -1
    with open(path) as fh:
        n = len(json.load(fh))
    import math
    return math.ceil(n / chunk_size)


def merge_split(
    split: str,
    chunks_dir: str,
    out_dir: str,
    manifest_dir: str,
    chunk_size: int,
    check: bool = False,
) -> None:
    bin_chunks  = sorted(
        glob.glob(os.path.join(chunks_dir, f"{split}_*.bin")),
        key=_chunk_sort_key,
    )
    meta_chunks = sorted(
        glob.glob(os.path.join(chunks_dir, f"{split}_*_metadata.json")),
        key=_chunk_sort_key,
    )
    fail_chunks = sorted(
        glob.glob(os.path.join(chunks_dir, f"{split}_*_failures.json")),
        key=_chunk_sort_key,
    )

    if not bin_chunks:
        log.warning("No chunk .bin files found for split '%s' in %s", split, chunks_dir)
        return

    # Warn about missing chunks
    expected = _expected_chunk_count(split, manifest_dir, chunk_size)
    if expected > 0 and len(bin_chunks) < expected:
        log.warning(
            "%s: only %d / %d chunks present — %d jobs may not have completed yet",
            split, len(bin_chunks), expected, expected - len(bin_chunks),
        )

    if len(bin_chunks) != len(meta_chunks):
        log.warning(
            "%s: .bin count (%d) != metadata count (%d) — chunk set may be incomplete",
            split, len(bin_chunks), len(meta_chunks),
        )

    log.info("Merging %d chunks for split '%s' ...", len(bin_chunks), split)

    all_graphs:   list = []
    all_metadata: list = []
    all_failures: list = []

    for bin_path, meta_path in zip(bin_chunks, meta_chunks):
        graphs, _ = dgl.load_graphs(bin_path)
        all_graphs.extend(graphs)

        with open(meta_path) as fh:
            chunk_meta = json.load(fh)
        all_metadata.extend(chunk_meta)

        log.info("  %-45s  %d graphs", os.path.basename(bin_path), len(graphs))

        if len(graphs) != len(chunk_meta):
            log.error(
                "  Alignment error in %s: %d graphs but %d metadata entries",
                os.path.basename(bin_path), len(graphs), len(chunk_meta),
            )

    for fail_path in fail_chunks:
        with open(fail_path) as fh:
            all_failures.extend(json.load(fh))

    log.info(
        "%s: %d graphs total, %d failures across %d chunks",
        split, len(all_graphs), len(all_failures), len(bin_chunks),
    )

    if len(all_graphs) != len(all_metadata):
        log.error(
            "ALIGNMENT MISMATCH for %s: %d graphs vs %d metadata entries — "
            "NOT writing output; investigate chunk failures first",
            split, len(all_graphs), len(all_metadata),
        )
        return

    if check:
        log.info("(--check mode: skipping write for %s)", split)
        return

    os.makedirs(out_dir, exist_ok=True)
    bin_out  = os.path.join(out_dir, f"{split}.bin")
    meta_out = os.path.join(out_dir, f"{split}_metadata.json")
    fail_out = os.path.join(out_dir, f"{split}_failures.json")

    dgl.save_graphs(bin_out, all_graphs)
    log.info("Written: %s  (%d graphs)", bin_out, len(all_graphs))

    with open(meta_out, "w") as fh:
        json.dump(all_metadata, fh)
    log.info("Written: %s  (%d entries)", meta_out, len(all_metadata))

    with open(fail_out, "w") as fh:
        json.dump(all_failures, fh, indent=2)
    if all_failures:
        log.warning("Written: %s  (%d failures)", fail_out, len(all_failures))
    else:
        log.info("Written: %s  (0 failures)", fail_out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--split",        choices=["train", "val", "test", "all"],
                   default="all",
                   help="Which split to merge (default: all)")
    p.add_argument("--chunks-dir",   default=CHUNKS_DIR,
                   help="Directory containing chunk files (default: splits/chunks/)")
    p.add_argument("--out-dir",      default=SPLITS_DIR,
                   help="Output directory for merged files (default: splits/)")
    p.add_argument("--manifest-dir", default=SPLITS_DIR,
                   help="Directory containing split manifests (default: splits/)")
    p.add_argument("--chunk-size",   type=int, default=200,
                   help="Chunk size used during build, for completeness check (default: 200)")
    p.add_argument("--check",        action="store_true",
                   help="Report counts without writing output files")
    args = p.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        merge_split(
            split,
            chunks_dir=args.chunks_dir,
            out_dir=args.out_dir,
            manifest_dir=args.manifest_dir,
            chunk_size=args.chunk_size,
            check=args.check,
        )


if __name__ == "__main__":
    main()

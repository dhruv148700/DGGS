"""
build_causal_splits_dataset.py
──────────────────────────────
Build DGL heterograph datasets from the pre-computed train/val/test split
manifests.  Designed to run as independent Condor jobs (one chunk each) and
then be merged by merge_graph_dataset.py.

Each successful graph is saved alongside its manifest metadata dict so that
training and evaluation scripts can slice results by n_nodes, graph_type,
probe_role, etc. without rebuilding the dataset.

── MODES ────────────────────────────────────────────────────────────────────

  Single-machine (all splits, useful for testing on a small dataset):
      python build_causal_splits_dataset.py

  Single split on one machine:
      python build_causal_splits_dataset.py --split train

  One Condor chunk (called by scripts-condor/run_build_chunk.sh):
      python build_causal_splits_dataset.py \\
          --split train --start-idx 0 --end-idx 200

  Print total Condor job count (paste into build_dataset.cmd queue line):
      python build_causal_splits_dataset.py --print-n-jobs

── OUTPUT LAYOUT ────────────────────────────────────────────────────────────

  splits/chunks/
      train_00000_00200.bin
      train_00000_00200_metadata.json
      train_00000_00200_failures.json
      train_00200_00400.bin
      ...

  After merge_graph_dataset.py:
      splits/train.bin
      splits/train_metadata.json
      splits/train_failures.json   (union of all chunk failures)
"""

import argparse
import json
import logging
import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

import dgl

from scr.data_utils import load_causal_dataset_from_manifest

SPLITS_DIR = os.path.join(REPO_ROOT, "splits")
CHUNKS_DIR = os.path.join(SPLITS_DIR, "chunks")
CHUNK_SIZE = 200

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job enumeration (shared between build script and Condor shell wrapper)
# ---------------------------------------------------------------------------

def iter_jobs(manifest_dir: str = SPLITS_DIR, chunk_size: int = CHUNK_SIZE):
    """Yield one job dict per Condor process across all three splits.

    Splits are processed in train → val → test order so Process IDs are stable
    as long as the manifests and chunk_size are unchanged.
    """
    for split in ("train", "val", "test"):
        manifest_path = os.path.join(manifest_dir, f"{split}_manifest.json")
        with open(manifest_path) as fh:
            n = len(json.load(fh))
        chunk_id = 0
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            yield {
                "split":    split,
                "chunk_id": chunk_id,
                "start_idx": start,
                "end_idx":   end,
            }
            chunk_id += 1


# ---------------------------------------------------------------------------
# Core build function
# ---------------------------------------------------------------------------

def build_chunk(
    split: str,
    start_idx: int,
    end_idx: int,
    manifest_dir: str,
    out_dir: str,
    skip_existing: bool = False,
    use_scores: bool = False,
) -> tuple:
    """Build graphs for entries[start_idx:end_idx] of the given split.

    Writes three files into out_dir:
        {split}_{start:05d}_{end:05d}.bin
        {split}_{start:05d}_{end:05d}_metadata.json
        {split}_{start:05d}_{end:05d}_failures.json

    Args:
        skip_existing: if True and the .bin already exists on disk, skip this
                       chunk entirely.  Safe to use on resubmits after partial
                       runs — already-built chunks are not re-processed.

    Returns (n_graphs, n_failures).  Returns (0, 0) if chunk was skipped.
    """
    os.makedirs(out_dir, exist_ok=True)

    tag      = f"{split}_{start_idx:05d}_{end_idx:05d}"
    bin_path = os.path.join(out_dir, f"{tag}.bin")

    # Skip if .bin already exists — all three output files were written
    # atomically at the end of the previous run, so presence of the .bin
    # implies the metadata and failures files are also complete.
    if skip_existing and os.path.exists(bin_path):
        log.info("Skipping %s[%d:%d] — output already exists", split, start_idx, end_idx)
        return 0, 0

    manifest_path = os.path.join(manifest_dir, f"{split}_manifest.json")
    with open(manifest_path) as fh:
        all_entries = json.load(fh)

    entries = all_entries[start_idx:end_idx]
    log.info(
        "Building %s[%d:%d]  (%d entries)", split, start_idx, end_idx, len(entries)
    )

    graphs, metadata, failed = load_causal_dataset_from_manifest(
        entries, base_dir=REPO_ROOT, use_scores=use_scores,
    )

    tag = f"{split}_{start_idx:05d}_{end_idx:05d}"
    bin_path      = os.path.join(out_dir, f"{tag}.bin")
    meta_path     = os.path.join(out_dir, f"{tag}_metadata.json")
    failures_path = os.path.join(out_dir, f"{tag}_failures.json")

    dgl.save_graphs(bin_path, graphs)
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh)
    with open(failures_path, "w") as fh:
        json.dump(failed, fh, indent=2)

    log.info("Saved %d graphs → %s", len(graphs), bin_path)
    if failed:
        log.warning("%d failures → %s", len(failed), failures_path)

    return len(graphs), len(failed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split",       choices=["train", "val", "test", "all"],
                   default="all",
                   help="Which split to build (default: all)")
    p.add_argument("--start-idx",   type=int, default=None,
                   help="Start index within the manifest (chunk mode)")
    p.add_argument("--end-idx",     type=int, default=None,
                   help="End index within the manifest (chunk mode)")
    p.add_argument("--chunk-size",  type=int, default=CHUNK_SIZE,
                   help=f"Entries per Condor chunk (default: {CHUNK_SIZE})")
    p.add_argument("--manifest-dir", default=SPLITS_DIR,
                   help="Directory containing split manifests (default: splits/)")
    p.add_argument("--out-dir",     default=CHUNKS_DIR,
                   help="Output directory for chunk files (default: splits/chunks/)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip any chunk whose .bin already exists on disk "
                        "(safe for resubmits after partial runs)")
    p.add_argument("--use-scores", action="store_true",
                   help="Add CI-test reliability score as a 3rd node feature "
                        "(produces in_features=3 graphs; default: 2-feature baseline)")
    p.add_argument("--print-n-jobs", action="store_true",
                   help="Print total Condor job count and exit")
    args = p.parse_args()

    # ── Print job count for the Condor submit file ───────────────────────────
    if args.print_n_jobs:
        n = sum(1 for _ in iter_jobs(args.manifest_dir, args.chunk_size))
        print(n)
        return

    # ── Single Condor chunk mode ─────────────────────────────────────────────
    if args.start_idx is not None or args.end_idx is not None:
        if args.split == "all":
            p.error("--split must be train/val/test in chunk mode, not 'all'")
        if args.start_idx is None or args.end_idx is None:
            p.error("both --start-idx and --end-idx are required in chunk mode")
        build_chunk(
            args.split, args.start_idx, args.end_idx,
            args.manifest_dir, args.out_dir,
            skip_existing=args.skip_existing,
            use_scores=args.use_scores,
        )
        return

    # ── Local single-machine mode ────────────────────────────────────────────
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        manifest_path = os.path.join(args.manifest_dir, f"{split}_manifest.json")
        with open(manifest_path) as fh:
            n = len(json.load(fh))
        total_g, total_f = 0, 0
        for start in range(0, n, args.chunk_size):
            ng, nf = build_chunk(
                split, start, min(start + args.chunk_size, n),
                args.manifest_dir, args.out_dir,
                skip_existing=args.skip_existing,
                use_scores=args.use_scores,
            )
            total_g += ng
            total_f += nf
        log.info(
            "%s complete: %d graphs, %d failures (%.1f%% loss)",
            split, total_g, total_f,
            100 * total_f / max(1, total_g + total_f),
        )


if __name__ == "__main__":
    main()

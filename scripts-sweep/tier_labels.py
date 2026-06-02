"""
tier_labels.py — Pre-compute classical tier labels for every ABAF in the manifest.

Tier labels are derived from stable-extension files and are independent of any
DGGS kernel configuration.  Run this ONCE as a pre-step before the kernel sweep,
then merge with merge_tier_labels.py.

Tier definitions (per assumption, per ABAF)
-------------------------------------------
  skeptical  — in the intersection of ALL stable extensions
  credulous  — in the union but NOT the intersection (some but not all)
  rejected   — in no extension  (all_assumptions − union)
  no_ext     — extension file is empty; assumption has no evidence either way

Round-robin chunk assignment: entry i belongs to chunk C iff (i % n_chunks) == chunk_id.
This interleaves n_nodes sizes so no chunk is handed all large (slow) frameworks.

Output
------
  Per ABAF:    dataset/tier_labels/{instance_id}.json  — {assumption: tier_str}
  Per chunk:   dataset/tier_labels/tier_chunk_{chunk_id}_report.json

Usage (Condor array job, one process per chunk)
-----------------------------------------------
    python scripts-sweep/tier_labels.py --chunk-id 0  --n-chunks 16
    python scripts-sweep/tier_labels.py --chunk-id 3  --n-chunks 16 \\
        --manifest causal_manifest.json --tier-dir dataset/tier_labels/

Python API (used by run_config.py via merge_tier_labels.py output)
-------------------------------------------------------------------
    from scripts_sweep.tier_labels import load
    tier = load("causal_er_n3_d0.3_mna_a0.01_i0", tier_dir="dataset/tier_labels")
    # → {"arr_0_1": "skeptical", "arr_1_0": "rejected", ...}
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_SKEPTICAL = "skeptical"
_CREDULOUS = "credulous"
_REJECTED  = "rejected"
_NO_EXT    = "no_ext"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _parse_assumptions(aba_path: Path) -> set:
    """Extract assumption names from a .aba file ('a <name>' lines)."""
    assumptions = set()
    with open(aba_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2 and parts[0] == "a":
                assumptions.add(parts[1])
    return assumptions


def compute_tier_labels(aba_path: Path, labels_path: Path) -> Dict[str, str]:
    """
    Compute tier label for every assumption in one ABAF.

    Streams the extension file line-by-line so memory usage is O(n_assumptions)
    regardless of how many stable extensions exist (some n=6 files exceed 900 MB).
    Once the running intersection becomes empty it stops being updated, saving time
    on instances with many extensions.

    Returns a dict {assumption_name: tier_str} where tier_str is one of
    "skeptical", "credulous", "rejected", "no_ext".
    """
    all_assumptions = _parse_assumptions(aba_path)

    union: set = set()
    intersection: set | None = None  # None = not yet initialised
    n_exts = 0
    union_saturated = False           # True once union covers all_assumptions

    with open(labels_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ext = set(line.split(","))
            n_exts += 1
            if not union_saturated:
                union |= ext
                union_saturated = union >= all_assumptions
            if intersection is None:
                intersection = ext.copy()
            elif intersection:          # skip update once empty — can never recover
                intersection &= ext
            # Early exit: union is full and intersection is empty → all tiers are fixed
            if union_saturated and intersection is not None and not intersection:
                break

    if n_exts == 0:
        return {a: _NO_EXT for a in all_assumptions}

    if intersection is None:
        intersection = set()

    result: Dict[str, str] = {}
    for a in all_assumptions:
        if a in intersection:
            result[a] = _SKEPTICAL
        elif a in union:
            result[a] = _CREDULOUS
        else:
            result[a] = _REJECTED
    return result


# ---------------------------------------------------------------------------
# Python load API (used downstream by run_config.py)
# ---------------------------------------------------------------------------

def abaf_key(entry: dict) -> str:
    """Unique cache key for a manifest entry: stem of the abaf file path.

    instance_id is NOT unique — the same base graph appears with multiple
    probe roles (initial_full, easy_sat, boundary_sat, boundary_unsat), each
    pointing to a different .aba file.  The file stem is always unique.
    """
    return Path(entry["abaf"]).stem


def load(key: str, tier_dir: str = None) -> Dict[str, str]:
    """Load cached tier labels for one entry (key = abaf_key(entry))."""
    if tier_dir is None:
        tier_dir = str(REPO_ROOT / "dataset" / "tier_labels")
    path = Path(tier_dir) / f"{key}.json"
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Chunk build
# ---------------------------------------------------------------------------

def _out_path(key: str, tier_dir: Path) -> Path:
    return tier_dir / f"{key}.json"


def _already_done(key: str, tier_dir: Path) -> bool:
    return _out_path(key, tier_dir).exists()


def process_entry(entry: dict, tier_dir: Path) -> dict:
    key = abaf_key(entry)
    record = {
        "key":          key,
        "instance_id":  entry["instance_id"],
        "probe_role":   entry.get("probe_role", ""),
        "n_assumptions": entry["n_assumptions"],
        "skipped": False,
        "error": None,
        "time_s": None,
        "n_skeptical": None,
        "n_credulous": None,
        "n_rejected": None,
        "n_no_ext": None,
    }

    if _already_done(key, tier_dir):
        record["skipped"] = True
        return record

    aba_path    = REPO_ROOT / entry["abaf"]
    labels_path = REPO_ROOT / entry["labels"]

    try:
        t0     = time.perf_counter()
        tiers  = compute_tier_labels(aba_path, labels_path)
        elapsed = time.perf_counter() - t0

        with open(_out_path(key, tier_dir), "w") as fh:
            json.dump(tiers, fh)

        counts = {t: 0 for t in (_SKEPTICAL, _CREDULOUS, _REJECTED, _NO_EXT)}
        for v in tiers.values():
            counts[v] += 1

        record.update({
            "time_s":      round(elapsed, 4),
            "n_skeptical": counts[_SKEPTICAL],
            "n_credulous": counts[_CREDULOUS],
            "n_rejected":  counts[_REJECTED],
            "n_no_ext":    counts[_NO_EXT],
        })
    except Exception as exc:
        record["error"] = str(exc)

    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunk-id",  type=int, required=True)
    p.add_argument("--n-chunks",  type=int, required=True)
    p.add_argument("--manifest",  default=str(REPO_ROOT / "causal_manifest.json"))
    p.add_argument("--tier-dir",  default=str(REPO_ROOT / "dataset" / "tier_labels"))
    args = p.parse_args()

    if args.chunk_id >= args.n_chunks:
        raise ValueError(f"chunk_id={args.chunk_id} >= n_chunks={args.n_chunks}")

    tier_dir = Path(args.tier_dir)
    tier_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    chunk_entries = [e for i, e in enumerate(manifest) if i % args.n_chunks == args.chunk_id]

    print(
        f"Chunk {args.chunk_id}/{args.n_chunks}  "
        f"entries={len(chunk_entries)}  "
        f"tier_dir={tier_dir}"
    )

    records = []
    n_done = n_skipped = n_errors = 0
    n_total = len(chunk_entries)

    for entry in chunk_entries:
        rec = process_entry(entry, tier_dir)
        records.append(rec)
        n_done += 1

        if rec["skipped"]:
            n_skipped += 1
            status = "SKIP"
        elif rec["error"]:
            n_errors += 1
            status = f"ERROR: {rec['error']}"
        else:
            status = (
                f"OK  sk={rec['n_skeptical']}  cr={rec['n_credulous']}  "
                f"rej={rec['n_rejected']}  no_ext={rec['n_no_ext']}  "
                f"t={rec['time_s']:.3f}s"
            )

        if n_done % 200 == 0 or rec["error"]:
            print(f"  [{n_done}/{n_total}]  {rec['key']}  {status}")

    ok    = [r for r in records if not r["skipped"] and not r["error"]]
    times = sorted(r["time_s"] for r in ok)
    n_ok  = len(ok)

    report = {
        "chunk_id":    args.chunk_id,
        "n_chunks":    args.n_chunks,
        "n_total":     n_total,
        "n_skipped":   n_skipped,
        "n_errors":    n_errors,
        "n_ok":        n_ok,
        "time_med_s":  times[n_ok // 2]    if times else None,
        "time_p90_s":  times[int(n_ok * 0.9)] if times else None,
        "time_max_s":  times[-1]            if times else None,
        "time_total_s": round(sum(times), 2) if times else None,
        "records":     records,
    }
    report_path = tier_dir / f"tier_chunk_{args.chunk_id}_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    if times:
        print(
            f"\nChunk {args.chunk_id} done  "
            f"ok={n_ok}  skipped={n_skipped}  errors={n_errors}\n"
            f"  time: med={times[n_ok//2]:.3f}s  "
            f"p90={times[int(n_ok*0.9)]:.3f}s  "
            f"max={times[-1]:.3f}s  "
            f"total={sum(times):.1f}s"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
sanity_check.py — Degenerate behaviour and non-convergence audit.

Loads the PKL files from test_convergence.py and checks each entry for:

  Degenerate final states (final_strengths all within THRESH of a fixed value):
    - collapse_zero   : all assumptions → 0
    - collapse_one    : all assumptions → 1
    - stuck_init      : all assumptions within THRESH of their initial value
                        (no movement; more informative than "stuck at 0.5")
    - flat_line       : all assumptions within THRESH of each other but not
                        at 0, 1, or initial (collapsed to some arbitrary constant)

  Non-convergence:
    - did_not_converge : usable (no timeout/OOM) but global_converged is False
    - timeout / oom    : subprocess hit the time/memory limit

Outputs
-------
  Console  — summary table per (init × degenerate category)
  figures/non_converged_random.txt  — filenames that did not converge (random init)
  figures/non_converged_fixed.txt   — filenames that did not converge (fixed init)
  figures/degenerate_random.txt     — degenerate filenames + reason (random init)
  figures/degenerate_fixed.txt      — degenerate filenames + reason (fixed init)

Usage
-----
    python scripts/sanity_check.py
"""

import pickle
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

# ─── Config ──────────────────────────────────────────────────────────────────
PKL_RANDOM = Path("convergence_results_dggs/dggs_e3_d5_s5000_randinit.pkl")
PKL_FIXED  = Path("convergence_results_dggs/dggs_e3_d5_s5000.pkl")
OUTPUT_DIR = Path("figures/")
THRESH     = 0.01   # tolerance for "all values near X"
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)


# ─── Load ─────────────────────────────────────────────────────────────────────

def load_pkl(path: Path, label: str):
    if not path.exists():
        print(f"[WARN] {path} not found — skipping '{label}' init.")
        return []
    with open(path, "rb") as f:
        records = pickle.load(f)
    for r in records:
        r["init"] = label
    print(f"Loaded {len(records)} entries from {path.name}")
    return records


# ─── Degenerate checks ────────────────────────────────────────────────────────

def _all_near(values, target, thresh):
    return all(abs(v - target) < thresh for v in values)


def _all_near_each_other(values, thresh):
    return (max(values) - min(values)) < thresh


def classify(entry) -> str:
    """Return a degenerate category label or None if the entry looks healthy."""
    fs = entry.get("final_strengths")
    if not fs:
        return None

    vals = list(fs.values())

    if _all_near(vals, 0.0, THRESH):
        return "collapse_zero"

    if _all_near(vals, 1.0, THRESH):
        return "collapse_one"

    # check if nothing moved from initial
    init_s = entry.get("initial_strengths", {})
    if init_s:
        moved = any(
            abs(fs.get(k, 0) - init_s[k]) >= THRESH
            for k in init_s
        )
        if not moved:
            return "stuck_init"

    # all assumptions collapsed to same value (not 0, 1, or init)
    if _all_near_each_other(vals, THRESH):
        return "flat_line"

    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    all_records = []
    for path, label in [(PKL_RANDOM, "random"), (PKL_FIXED, "fixed")]:
        all_records.extend(load_pkl(path, label))

    if not all_records:
        print("No records found.")
        return

    # ── Categorise every entry ────────────────────────────────────────────────
    degenerate   = defaultdict(list)   # {init: [(filename, reason)]}
    non_converged = defaultdict(list)  # {init: [filename]}
    timed_out    = defaultdict(list)
    oom_entries  = defaultdict(list)

    category_counts = defaultdict(lambda: defaultdict(int))

    for r in all_records:
        init     = r["init"]
        filename = r["file"]
        timeout  = r.get("timeout", False)
        oom      = r.get("oom", False)

        if timeout:
            timed_out[init].append(filename)
            category_counts[init]["timeout"] += 1
            continue

        if oom:
            oom_entries[init].append(filename)
            category_counts[init]["oom"] += 1
            continue

        # usable entry
        degen = classify(r)
        if degen:
            degenerate[init].append((filename, degen))
            category_counts[init][degen] += 1

        if not r.get("global_converged", False):
            non_converged[init].append(filename)
            category_counts[init]["did_not_converge"] += 1
        else:
            category_counts[init]["ok"] += 1

    # ── Console summary ───────────────────────────────────────────────────────
    categories = [
        "ok", "did_not_converge", "timeout", "oom",
        "collapse_zero", "collapse_one", "stuck_init", "flat_line",
    ]
    inits = sorted({r["init"] for r in all_records})

    rows = []
    for init in inits:
        row = {"init": init}
        for cat in categories:
            row[cat] = category_counts[init].get(cat, 0)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("init")

    print("\n── Sanity check summary ─────────────────────────────────────────")
    print(df.to_string())
    print()

    # ── Non-converged .txt files ───────────────────────────────────────────────
    for init in inits:
        nc = non_converged[init]
        out = OUTPUT_DIR / f"non_converged_{init}.txt"
        with open(out, "w") as f:
            f.write(f"# Did not converge — {init} init — {len(nc)} frameworks\n")
            for name in sorted(nc):
                f.write(name + "\n")
        print(f"Non-converged ({init}): {len(nc)} → {out}")

    # ── Degenerate .txt files ──────────────────────────────────────────────────
    for init in inits:
        deg = degenerate[init]
        out = OUTPUT_DIR / f"degenerate_{init}.txt"
        with open(out, "w") as f:
            f.write(f"# Degenerate final states — {init} init — {len(deg)} frameworks\n")
            f.write("# filename | reason\n")
            for name, reason in sorted(deg):
                f.write(f"{name} | {reason}\n")
        print(f"Degenerate  ({init}): {len(deg)} → {out}")

    # ── Per-category detail ────────────────────────────────────────────────────
    print("\n── Degenerate category key ──────────────────────────────────────")
    print(f"  collapse_zero    : all final strengths within {THRESH} of 0")
    print(f"  collapse_one     : all final strengths within {THRESH} of 1")
    print(f"  stuck_init       : no assumption moved more than {THRESH} from its initial value")
    print(f"  flat_line        : all assumptions collapsed to same value (not 0, 1, or init)")
    print(f"  did_not_converge : usable but global_converged=False (includes degenerate)")
    print(f"  timeout/oom      : subprocess hit resource limit")


if __name__ == "__main__":
    run()

"""
split_dataset.py — Stratified train/val/test split for the Causal ABA GNN dataset.

USAGE:
    python split_dataset.py [--manifest PATH] [--output_dir DIR] [--seed INT]

    Defaults:
        --manifest   causal_manifest.json
        --output_dir splits
        --seed       42

STRATIFICATION RATIONALE:
    Entries are grouped by instance_id to prevent data leakage (all probes from
    the same instance land in the same split).  Each instance is assigned a
    stratum based on (n_nodes, graph_type, n_probes_category), producing up to
    16 strata (4 node-counts × 2 graph-types × 2 probe-count buckets: "single"
    when n_probes == 1, "multi" otherwise).  Splitting is performed independently
    within each stratum:
        outer: 75 % → train_pool, 25 % → test
        inner: 80 % of train_pool → train, 20 % → val
    A single seeded random.Random instance guarantees full reproducibility.
    Strata are processed in sorted order to eliminate any ordering artefacts.
"""

import argparse
import csv
import json
import logging
import os
import random
from collections import defaultdict

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Load
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> list[dict]:
    log.info("Loading manifest from %s", path)
    with open(path) as fh:
        data = json.load(fh)
    log.info("Loaded %d probe entries", len(data))
    return data


# ---------------------------------------------------------------------------
# Step 2: Aggregate probes → instances
# ---------------------------------------------------------------------------

def aggregate_to_instances(manifest: list[dict]) -> dict[str, dict]:
    """Return {instance_id: instance_record}.

    instance_record keys:
        instance_id, n_nodes, graph_type, n_probes, n_probes_category
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest:
        groups[entry["instance_id"]].append(entry)

    instances: dict[str, dict] = {}
    for iid, probes in groups.items():
        # Assert constant fields across probes of the same instance
        n_nodes_vals = {p["n_nodes"] for p in probes}
        gt_vals = {p["graph_type"] for p in probes}
        assert len(n_nodes_vals) == 1, (
            f"instance {iid}: n_nodes is not constant across probes: {n_nodes_vals}"
        )
        assert len(gt_vals) == 1, (
            f"instance {iid}: graph_type is not constant across probes: {gt_vals}"
        )

        n_probes = len(probes)
        instances[iid] = {
            "instance_id": iid,
            "n_nodes": n_nodes_vals.pop(),
            "graph_type": gt_vals.pop(),
            "n_probes": n_probes,
            "n_probes_category": "single" if n_probes == 1 else "multi",
        }

    log.info("Aggregated to %d instances", len(instances))
    return instances


# ---------------------------------------------------------------------------
# Step 3: Compute strata
# ---------------------------------------------------------------------------

def compute_strata(
    instances: dict[str, dict],
) -> dict[tuple, list[dict]]:
    """Return {(n_nodes, graph_type, n_probes_category): [instance_records]}."""
    stratified: dict[tuple, list[dict]] = defaultdict(list)
    for inst in instances.values():
        key = (inst["n_nodes"], inst["graph_type"], inst["n_probes_category"])
        stratified[key].append(inst)
    log.info("Computed %d strata", len(stratified))
    return dict(stratified)


# ---------------------------------------------------------------------------
# Step 4: Stratum size check
# ---------------------------------------------------------------------------

def check_stratum_sizes(
    stratified_instances: dict[tuple, list[dict]],
    min_size: int = 10,
) -> None:
    header = f"{'stratum':<40} {'n_instances':>12}"
    log.info("Stratum size table:")
    log.info(header)
    log.info("-" * len(header))
    for key in sorted(stratified_instances):
        n = len(stratified_instances[key])
        label = str(key)
        flag = "  *** WARN: < 10" if n < min_size else ""
        log.info("  %-38s %6d%s", label, n, flag)


# ---------------------------------------------------------------------------
# Step 5a: Outer split — train_pool / test
# ---------------------------------------------------------------------------

def _split_one_stratum(
    instances: list[dict],
    ratio: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Shuffle and split: first group gets (1 - ratio), second gets ratio.

    Enforces min 1 per partition when len >= 3.
    """
    n = len(instances)
    shuffled = instances[:]
    rng.shuffle(shuffled)
    n_second = round(n * ratio)
    if n >= 3:
        n_second = max(1, min(n_second, n - 1))
    first = shuffled[: n - n_second]
    second = shuffled[n - n_second :]
    return first, second


def outer_split(
    stratified_instances: dict[tuple, list[dict]],
    ratio: float = 0.25,
    *,
    rng: random.Random,
) -> tuple[dict[tuple, list[dict]], dict[tuple, list[dict]]]:
    """Split each stratum 75/25 into train_pool / test."""
    train_pool: dict[tuple, list[dict]] = {}
    test_split: dict[tuple, list[dict]] = {}
    for key in sorted(stratified_instances):
        tp, ts = _split_one_stratum(stratified_instances[key], ratio, rng)
        train_pool[key] = tp
        test_split[key] = ts
        log.info(
            "  outer stratum %s: train_pool=%d  test=%d",
            key, len(tp), len(ts),
        )
    return train_pool, test_split


# ---------------------------------------------------------------------------
# Step 5b: Inner split — train / val
# ---------------------------------------------------------------------------

def inner_split(
    train_pool_stratified: dict[tuple, list[dict]],
    ratio: float = 0.20,
    *,
    rng: random.Random,
) -> tuple[dict[tuple, list[dict]], dict[tuple, list[dict]]]:
    """Split each stratum of train_pool 80/20 into train / val."""
    train: dict[tuple, list[dict]] = {}
    val: dict[tuple, list[dict]] = {}
    for key in sorted(train_pool_stratified):
        tr, vl = _split_one_stratum(train_pool_stratified[key], ratio, rng)
        train[key] = tr
        val[key] = vl
        log.info(
            "  inner stratum %s: train=%d  val=%d",
            key, len(tr), len(vl),
        )
    return train, val


# ---------------------------------------------------------------------------
# Step 6: Expand instance IDs back to probe entries
# ---------------------------------------------------------------------------

def expand_to_probes(
    instance_ids: set[str],
    manifest: list[dict],
) -> list[dict]:
    return [e for e in manifest if e["instance_id"] in instance_ids]


# ---------------------------------------------------------------------------
# Step 7: Verification
# ---------------------------------------------------------------------------

def verify_split(
    train_entries: list[dict],
    val_entries: list[dict],
    test_entries: list[dict],
    original_manifest: list[dict],
    *,
    instances: dict[str, dict],
) -> None:
    """Run all verification checks (hard asserts + soft warnings)."""

    # ------------------------------------------------------------------ a
    train_ids = {e["instance_id"] for e in train_entries}
    val_ids = {e["instance_id"] for e in val_entries}
    test_ids = {e["instance_id"] for e in test_entries}

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids
    assert not tv, f"[HARD FAIL] train/val instance overlap: {tv}"
    assert not tt, f"[HARD FAIL] train/test instance overlap: {tt}"
    assert not vt, f"[HARD FAIL] val/test instance overlap: {vt}"
    log.info("[ASSERT OK] No instance_id overlap across splits")

    # ------------------------------------------------------------------ b
    orig_ids = {e["instance_id"] for e in original_manifest}
    split_ids = train_ids | val_ids | test_ids
    assert split_ids == orig_ids, (
        f"[HARD FAIL] Missing instances: {orig_ids - split_ids}; "
        f"extra: {split_ids - orig_ids}"
    )
    total_probes = len(train_entries) + len(val_entries) + len(test_entries)
    assert total_probes == len(original_manifest), (
        f"[HARD FAIL] Probe count mismatch: {total_probes} vs {len(original_manifest)}"
    )
    log.info("[ASSERT OK] All %d probes accounted for in exactly one split", total_probes)

    # ------------------------------------------------------------------ c  stratum proportions
    THRESHOLD = 0.05
    log.info("[SOFT CHECK] Stratum-level instance proportions (expected ~0.60/0.15/0.25):")
    by_stratum: dict[tuple, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        for iid in ids:
            key = (
                instances[iid]["n_nodes"],
                instances[iid]["graph_type"],
                instances[iid]["n_probes_category"],
            )
            by_stratum[key][split_name] += 1

    for key in sorted(by_stratum):
        counts = by_stratum[key]
        total = sum(counts.values())
        if total == 0:
            continue
        p_tr = counts["train"] / total
        p_vl = counts["val"] / total
        p_ts = counts["test"] / total
        flags = []
        if abs(p_tr - 0.60) > THRESHOLD:
            flags.append(f"train {p_tr:.2f} vs 0.60")
        if abs(p_vl - 0.15) > THRESHOLD:
            flags.append(f"val {p_vl:.2f} vs 0.15")
        if abs(p_ts - 0.25) > THRESHOLD:
            flags.append(f"test {p_ts:.2f} vs 0.25")
        flag_str = "  *** " + "; ".join(flags) if flags else ""
        log.info(
            "  %s  train=%.2f  val=%.2f  test=%.2f  (n=%d)%s",
            str(key), p_tr, p_vl, p_ts, total, flag_str,
        )

    # ------------------------------------------------------------------ d  probe_role distribution
    log.info("[SOFT CHECK] probe_role distribution across splits:")
    global_role_count: dict[str, int] = defaultdict(int)
    for e in original_manifest:
        global_role_count[e["probe_role"]] += 1
    global_total = len(original_manifest)

    split_role: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }
    split_totals = {
        "train": len(train_entries),
        "val": len(val_entries),
        "test": len(test_entries),
    }
    for split_name, entries in [
        ("train", train_entries), ("val", val_entries), ("test", test_entries)
    ]:
        for e in entries:
            split_role[split_name][e["probe_role"]] += 1

    all_roles = sorted(global_role_count)
    for role in all_roles:
        global_share = global_role_count[role] / global_total
        flags = []
        for split_name in ("train", "val", "test"):
            n_split = split_totals[split_name]
            if n_split == 0:
                continue
            share = split_role[split_name][role] / n_split
            if abs(share - global_share) > THRESHOLD:
                flags.append(f"{split_name} {share:.2f} vs global {global_share:.2f}")
        flag_str = "  *** " + "; ".join(flags) if flags else ""
        log.info(
            "  role=%-22s  global=%.2f  train=%.2f  val=%.2f  test=%.2f%s",
            role,
            global_share,
            split_role["train"][role] / max(1, split_totals["train"]),
            split_role["val"][role] / max(1, split_totals["val"]),
            split_role["test"][role] / max(1, split_totals["test"]),
            flag_str,
        )

    # ------------------------------------------------------------------ e  numeric balance
    log.info("[SOFT CHECK] Mean n_assumptions and n_atoms per split:")
    for split_name, entries in [
        ("train", train_entries), ("val", val_entries), ("test", test_entries)
    ]:
        if not entries:
            continue
        mean_assump = sum(e["n_assumptions"] for e in entries) / len(entries)
        mean_atoms = sum(e["n_atoms"] for e in entries) / len(entries)
        log.info(
            "  %-6s  n_probes=%6d  mean_n_assumptions=%.2f  mean_n_atoms=%.2f",
            split_name, len(entries), mean_assump, mean_atoms,
        )


# ---------------------------------------------------------------------------
# Step 8 & 9: Write outputs
# ---------------------------------------------------------------------------

def write_outputs(
    train_entries: list[dict],
    val_entries: list[dict],
    test_entries: list[dict],
    output_dir: str,
    *,
    by_stratum: dict[tuple, dict[str, int]],
    split_metadata: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for name, entries in [
        ("train_manifest.json", train_entries),
        ("val_manifest.json", val_entries),
        ("test_manifest.json", test_entries),
    ]:
        path = os.path.join(output_dir, name)
        with open(path, "w") as fh:
            json.dump(entries, fh, indent=2)
        log.info("Wrote %s  (%d entries)", path, len(entries))

    # split_summary.csv
    csv_path = os.path.join(output_dir, "split_summary.csv")
    fieldnames = [
        "stratum",
        "n_total_instances",
        "n_train",
        "n_val",
        "n_test",
        "n_train_probes",
        "n_val_probes",
        "n_test_probes",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(by_stratum):
            counts = by_stratum[key]
            writer.writerow({
                "stratum": str(key),
                "n_total_instances": counts["n_total_instances"],
                "n_train": counts["n_train"],
                "n_val": counts["n_val"],
                "n_test": counts["n_test"],
                "n_train_probes": counts["n_train_probes"],
                "n_val_probes": counts["n_val_probes"],
                "n_test_probes": counts["n_test_probes"],
            })
    log.info("Wrote %s", csv_path)

    # split_metadata.json
    meta_path = os.path.join(output_dir, "split_metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(split_metadata, fh, indent=2)
    log.info("Wrote %s", meta_path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def _flatten_stratified(stratified: dict[tuple, list[dict]]) -> set[str]:
    ids: set[str] = set()
    for lst in stratified.values():
        for inst in lst:
            ids.add(inst["instance_id"])
    return ids


def main(
    manifest_path: str = "causal_manifest.json",
    output_dir: str = ".",
    seed: int = 42,
) -> None:
    rng = random.Random(seed)

    # 1. Load
    manifest = load_manifest(manifest_path)

    # 2. Aggregate
    instances = aggregate_to_instances(manifest)

    # 3. Strata
    stratified = compute_strata(instances)

    # 4. Size check
    check_stratum_sizes(stratified)

    # 5a. Outer split
    log.info("Performing outer split (train_pool=75%%, test=25%%)...")
    train_pool_strat, test_strat = outer_split(stratified, ratio=0.25, rng=rng)

    # 5b. Inner split
    log.info("Performing inner split (train=80%%, val=20%% of train_pool)...")
    train_strat, val_strat = inner_split(train_pool_strat, ratio=0.20, rng=rng)

    # 6. Expand to probes
    train_ids = _flatten_stratified(train_strat)
    val_ids = _flatten_stratified(val_strat)
    test_ids = _flatten_stratified(test_strat)

    train_entries = expand_to_probes(train_ids, manifest)
    val_entries = expand_to_probes(val_ids, manifest)
    test_entries = expand_to_probes(test_ids, manifest)

    log.info(
        "Expanded: train=%d probes  val=%d probes  test=%d probes",
        len(train_entries), len(val_entries), len(test_entries),
    )

    # 7. Verify
    log.info("Running verification checks...")
    verify_split(
        train_entries, val_entries, test_entries, manifest,
        instances=instances,
    )

    # Build by_stratum summary for CSV
    by_stratum: dict[tuple, dict] = {}
    probe_index: dict[str, list[dict]] = defaultdict(list)
    for e in manifest:
        probe_index[e["instance_id"]].append(e)

    for key in sorted(stratified):
        n_total = len(stratified[key])
        n_tr = len(train_strat.get(key, []))
        n_vl = len(val_strat.get(key, []))
        n_ts = len(test_strat.get(key, []))

        def probe_count(strat_dict, k):
            return sum(
                len(probe_index[inst["instance_id"]])
                for inst in strat_dict.get(k, [])
            )

        by_stratum[key] = {
            "n_total_instances": n_total,
            "n_train": n_tr,
            "n_val": n_vl,
            "n_test": n_ts,
            "n_train_probes": probe_count(train_strat, key),
            "n_val_probes": probe_count(val_strat, key),
            "n_test_probes": probe_count(test_strat, key),
        }

    # split_metadata.json
    split_metadata = {
        "random_seed": seed,
        "outer_ratio": 0.25,
        "inner_ratio": 0.20,
        "n_strata": len(stratified),
        "total_instances": len(instances),
        "total_probes": len(manifest),
        "train_instances": len(train_ids),
        "val_instances": len(val_ids),
        "test_instances": len(test_ids),
        "train_probes": len(train_entries),
        "val_probes": len(val_entries),
        "test_probes": len(test_entries),
    }

    # 8. Write outputs
    write_outputs(
        train_entries, val_entries, test_entries,
        output_dir,
        by_stratum=by_stratum,
        split_metadata=split_metadata,
    )

    log.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratified train/val/test split for Causal ABA GNN dataset."
    )
    parser.add_argument(
        "--manifest",
        default="causal_manifest.json",
        help="Path to input manifest JSON (default: causal_manifest.json)",
    )
    parser.add_argument(
        "--output_dir",
        default="splits",
        help="Directory for output files (default: splits)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    main(args.manifest, args.output_dir, args.seed)

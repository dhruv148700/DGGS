"""
test_probe_pipeline.py
──────────────────────
Smoke test for the multi-probe data generation pipeline.

Runs a single instance (n=5, ER, density=0.5, alpha=0.05, seed=42),
prints a report of all probes produced, and verifies that the expected
files exist and the manifest entries are well-formed.

    python test_probe_pipeline.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scr"))

from ArgCausalDisco.utils.data_utils import simulate_dag, simulate_data_and_run_PC
from ArgCausalDisco.utils.helpers import random_stability
from scr.causal_aba.abapc import facts_from_sepset, get_probes_from_facts
from scr.causal_aba.enums import SemanticEnum
from scr.causal_aba.lp_to_aba_translator import lp_facts_to_aba_file
import networkx as nx

# ── Parameters ────────────────────────────────────────────────────────────────

N          = 5
GRAPH_TYPE = "er"
DENSITY    = 0.5
ALPHA      = 0.05
SEED       = 42
INSTANCE_I = 0

INPUT_DIR  = "test_probe_input"
OUTPUT_DIR = "test_probe_output"
MANIFEST   = "test_probe_manifest.json"

ROLE_CODES = {
    "initial_full":   "full",
    "easy_sat":       "esat",
    "boundary_sat":   "bsat",
    "boundary_unsat": "busat",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def fname(i, role=None):
    role_str = f"_{ROLE_CODES[role]}" if role else ""
    return f"causal_er_n{N}_d{DENSITY}_mna_a{ALPHA}_i{i}{role_str}.aba"


def instance_id(i):
    return os.path.splitext(fname(i))[0]


def write_label_file(models, path):
    with open(path, "w") as fh:
        for model in models:
            fh.write(",".join(sorted(model.assumptions)) + "\n")


# ── Run ───────────────────────────────────────────────────────────────────────

os.makedirs(INPUT_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Running n={N} ER density={DENSITY} alpha={ALPHA} seed={SEED}")
print()

random_stability(SEED)

s0 = max(N - 1, int(DENSITY * N * (N - 1) / 2))
B_true = simulate_dag(d=N, s0=s0, graph_type="ER")
G_true = nx.from_numpy_array(B_true, create_using=nx.DiGraph)
G_true = nx.relabel_nodes(G_true, {i: f"X{i+1}" for i in range(N)})

_data, cg = simulate_data_and_run_PC(G_true, alpha=ALPHA, seed=SEED)
facts = facts_from_sepset(cg, N, ALPHA)
print(f"Facts extracted from PC: {len(facts)}")

t0 = time.time()
probes = get_probes_from_facts(facts, N, semantics=SemanticEnum.ST)
elapsed = time.time() - t0
print(f"Binary search done in {elapsed:.2f}s  →  {len(probes)} probe(s) selected")
print()

# ── Report probes ─────────────────────────────────────────────────────────────

print(f"{'Role':<16}  {'Facts':>6}  {'SAT':>5}  {'#Models':>8}  {'#Credulous':>10}")
print("-" * 54)
for p in probes:
    n_models = len(p.models) if p.models else 0
    n_cred   = len(p.credulous) if p.credulous else 0
    print(f"{p.role:<16}  {p.fact_count:>6}  {str(p.is_sat):>5}  {n_models:>8}  {n_cred:>10}")
print()

# ── Write files + manifest ────────────────────────────────────────────────────

manifest_entries = []

iid = instance_id(INSTANCE_I)

for probe in probes:
    role = probe.role
    input_path  = os.path.join(INPUT_DIR,  fname(INSTANCE_I, role))
    output_path = os.path.join(OUTPUT_DIR, "output_" + fname(INSTANCE_I, role))

    fw = lp_facts_to_aba_file(probe.fact_subset, n_nodes=N, out_path=input_path)
    write_label_file(probe.models or [], output_path)

    scores_path = input_path.replace(".aba", ".scores.json")

    entry = {
        "graph_type":       GRAPH_TYPE,
        "n_nodes":          N,
        "density_or_m":     DENSITY,
        "alpha":            ALPHA,
        "abaf":             input_path,
        "labels":           output_path,
        "scores":           scores_path,
        "n_atoms":          len(fw.all_elements()),
        "n_assumptions":    len(fw.assumptions),
        "n_credulous":      len(probe.credulous),
        "instance_id":      iid,
        "probe_role":       role,
        "probe_fact_count": probe.fact_count,
        "probe_is_sat":     probe.is_sat,
    }
    manifest_entries.append(entry)

with open(MANIFEST, "w") as fh:
    json.dump(manifest_entries, fh, indent=2)

# ── Verify files exist ────────────────────────────────────────────────────────

print("Files written:")
all_ok = True
for entry in manifest_entries:
    for key in ("abaf", "labels", "scores"):
        path = entry[key]
        exists = os.path.exists(path)
        size   = os.path.getsize(path) if exists else -1
        status = f"OK ({size:>5} bytes)" if exists else "MISSING"
        if not exists:
            all_ok = False
        role = entry["probe_role"]
        print(f"  [{role:<14}] {key:<7}: {status}  {path}")

print()
print(f"Manifest: {MANIFEST}  ({len(manifest_entries)} entries)")
print()

# ── Validate manifest schema ──────────────────────────────────────────────────

REQUIRED_KEYS = {
    "graph_type", "n_nodes", "density_or_m", "alpha",
    "abaf", "labels", "scores",
    "n_atoms", "n_assumptions", "n_credulous",
    "instance_id", "probe_role", "probe_fact_count", "probe_is_sat",
}

schema_ok = True
for entry in manifest_entries:
    missing = REQUIRED_KEYS - entry.keys()
    if missing:
        print(f"SCHEMA ERROR in {entry['probe_role']}: missing keys {missing}")
        schema_ok = False

    # SAT probes must have at least n_atoms > 0
    if entry["probe_is_sat"] and entry["n_atoms"] == 0:
        print(f"LOGIC ERROR in {entry['probe_role']}: SAT probe but n_atoms=0")
        schema_ok = False

    # UNSAT probes must have n_credulous == 0
    if not entry["probe_is_sat"] and entry["n_credulous"] > 0:
        print(f"LOGIC ERROR in {entry['probe_role']}: UNSAT probe but n_credulous={entry['n_credulous']}")
        schema_ok = False

    # boundary_sat must have the largest fact_count among SAT probes
    # (checked below after collecting all)

sat_entries  = [e for e in manifest_entries if e["probe_is_sat"]]
unsat_entries = [e for e in manifest_entries if not e["probe_is_sat"]]

if sat_entries:
    bsat = next((e for e in manifest_entries if e["probe_role"] == "boundary_sat"), None)
    if bsat:
        max_sat_count = max(e["probe_fact_count"] for e in sat_entries)
        if bsat["probe_fact_count"] != max_sat_count:
            print(f"LOGIC ERROR: boundary_sat has fact_count={bsat['probe_fact_count']} but max SAT is {max_sat_count}")
            schema_ok = False

if unsat_entries:
    bunsat = next((e for e in manifest_entries if e["probe_role"] == "boundary_unsat"), None)
    if bunsat:
        min_unsat_count = min(e["probe_fact_count"] for e in unsat_entries)
        if bunsat["probe_fact_count"] != min_unsat_count:
            print(f"LOGIC ERROR: boundary_unsat has fact_count={bunsat['probe_fact_count']} but min UNSAT is {min_unsat_count}")
            schema_ok = False

# All probe_fact_counts must be unique (dedup guarantee)
counts = [e["probe_fact_count"] for e in manifest_entries]
if len(counts) != len(set(counts)):
    print(f"DEDUP ERROR: duplicate fact_counts in manifest: {counts}")
    schema_ok = False

print("Schema validation:", "PASS" if schema_ok else "FAIL")
print("File existence:   ", "PASS" if all_ok else "FAIL")
print()
if all_ok and schema_ok:
    print("All checks passed.")

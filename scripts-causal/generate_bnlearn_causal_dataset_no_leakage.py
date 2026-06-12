"""
generate_bnlearn_causal_dataset_no_leakage.py
─────────────────────────────────────────────
Same pipeline as generate_bnlearn_causal_dataset.py but with no leakage:

  - No binary search or probing — full raw fact set from PC is used as-is
  - optimise_remove_edges=False — no edges pruned from the graph before
    path enumeration (no indep facts leak skeleton information)

Seed generation and data loading are identical to experiments_bnlearn.py.

Outputs:
  no_leakage_data_bnlearn_causal/aba/causal_bnlearn_{dataset}_n{n}_a0.01_s{seed}_full.aba
  no_leakage_data_bnlearn_causal/aba/causal_bnlearn_{dataset}_n{n}_a0.01_s{seed}_full.scores.json
  no_leakage_data_bnlearn_causal/dag/dag_bnlearn_{dataset}_n{n}.npy

Run from repo root:
    python scripts-causal/generate_bnlearn_causal_dataset_no_leakage.py
"""

import os
import sys
import logging
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scr"))

from ArgCausalDisco.utils.data_utils import load_bnlearn_data_dag
from ArgCausalDisco.utils.helpers import random_stability
from scr.causal_aba.abapc import get_cg_and_facts
from scr.causal_aba.lp_to_aba_translator import lp_facts_to_aba_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(REPO_ROOT, "scripts-causal", "generate_bnlearn_causal_dataset_no_leakage.log")),
    ],
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_PATH   = os.path.join(REPO_ROOT, "ArgCausalDisco", "datasets")
ABA_OUT_DIR = os.path.join(REPO_ROOT, "no_leakage_data_bnlearn_causal", "aba")
DAG_OUT_DIR = os.path.join(REPO_ROOT, "no_leakage_data_bnlearn_causal", "dag")

ALPHA       = 0.01
SAMPLE_SIZE = 5000
N_RUNS      = 50

DATASET_LIST = [
    "cancer",
    "earthquake",
    "survey",
    "asia",
]

os.makedirs(ABA_OUT_DIR, exist_ok=True)
os.makedirs(DAG_OUT_DIR, exist_ok=True)

# ─── Seed generation: identical to experiments_bnlearn.py ─────────────────────
random_stability(2024)
seeds_list = np.random.randint(0, 10000, (N_RUNS,)).tolist()
logger.info(f"Seeds ({N_RUNS}): {seeds_list}")

# ─── Main loop ────────────────────────────────────────────────────────────────

for dataset_name in DATASET_LIST:
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {dataset_name}")

    n_skipped = 0

    for seed_idx, seed in enumerate(seeds_list):

        # ── Load BN data (same call as experiments_bnlearn.py) ────────────────
        X_s, B_true = load_bnlearn_data_dag(
            dataset_name=dataset_name,
            data_path=DATA_PATH,
            sample_size=SAMPLE_SIZE,
            seed=seed,
            standardise=True,
            print_info=(seed_idx == 0),
        )
        n_nodes = B_true.shape[0]

        aba_stem = f"bnlearn_{dataset_name}_n{n_nodes}_a{ALPHA}_s{seed}"
        aba_path = os.path.join(ABA_OUT_DIR, f"causal_{aba_stem}_full.aba")
        dag_path = os.path.join(DAG_OUT_DIR, f"dag_bnlearn_{dataset_name}_n{n_nodes}.npy")

        # ── Write B_true once per dataset (seed-invariant) ────────────────────
        if not os.path.exists(dag_path):
            np.save(dag_path, B_true)
            logger.info(f"  Written DAG: {os.path.basename(dag_path)}")

        # ── Skip if .aba already generated ────────────────────────────────────
        if os.path.exists(aba_path):
            logger.info(f"  [{seed_idx+1}/{N_RUNS}] seed={seed} — already exists, skipping")
            n_skipped += 1
            continue

        # ── Run PC and extract full fact set ──────────────────────────────────
        _, facts = get_cg_and_facts(X_s, alpha=ALPHA)
        logger.info(f"  [{seed_idx+1}/{N_RUNS}] seed={seed} — {len(facts)} facts from PC")

        # ── Translate full fact set directly — no probing, no edge removal ────
        lp_facts_to_aba_file(facts, n_nodes=n_nodes, out_path=aba_path, optimise_remove_edges=False)
        logger.info(f"    Written: {os.path.basename(aba_path)}")

    logger.info(
        f"Dataset {dataset_name} complete — "
        f"skipped={n_skipped}  written={N_RUNS - n_skipped}"
    )

logger.info("All datasets done.")

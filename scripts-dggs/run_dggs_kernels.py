"""
run_dggs_kernels.py — DGGS convergence experiments across kernel configurations.

Mirrors the structure of test_convergence.py but exposes body aggregation and
influence function as CLI arguments, so all four kernel combos can be run:

  body: prod  → ProductBody   (product of body atom strengths)
        min   → MinBody       (minimum of body atom strengths)

  influence: lin  → LinearInfluence(k=1.0)   (DF-QuAD, Baroni et al. 2019)
             quad → QuadraticInfluence(k=1.0) (QE semantics, Rago et al. 2016)

Output pkl names:
  dggs_e<exp>_d<DELTA>_s<MAX_STEPS>_body-<body>_inf-<inf>[_randinit].pkl

Record schema is identical to test_convergence.py plus two extra fields:
  body_kernel  — kernel name string (e.g. "ProductBody")
  inf_kernel   — influence name string (e.g. "LinearInfluence")

Usage:
  python scripts-dggs/run_dggs_kernels.py --body min --inf lin --init fixed
  python scripts-dggs/run_dggs_kernels.py --body prod --inf quad --init random
  python scripts-dggs/run_dggs_kernels.py --body min --inf quad --init fixed
"""

import argparse
import hashlib
import re
import pickle
import random
import signal
import sys
import traceback
from collections import defaultdict
from decimal import Decimal
from multiprocessing import Process, Queue
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from tqdm import tqdm

from scr.dependency_graph import DependencyGraph
from scr.ABAF import ABAF
from scr.dggs import DGGSRunner
from scr.dggs.kernels import (
    ProductBody, MinBody,
    MaxAggregation,
    LinearInfluence, QuadraticInfluence,
)

# ─── Config ──────────────────────────────────────────────────────────────────
INPUT_DIR       = Path("data/abaf/").resolve()
OUTPUT_DIR      = Path("convergence_results_dggs/")

MAX_FILES       = 0        # 0 = no limit
MIN_SENTENCES   = 0
MAX_SENTENCES   = 100
TIMEOUT_SECONDS = 600

EPSILON         = 1e-3
DELTA           = 5
MAX_STEPS       = 5000

RESULT_OVERRIDE = False
SEED            = 42
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

BODY_KERNELS = {
    "prod": ProductBody,
    "min":  MinBody,
}

INF_KERNELS = {
    "lin":  lambda: LinearInfluence(conservativeness=1.0),
    "quad": lambda: QuadraticInfluence(conservativeness=1.0),
}


def _output_pkl(body: str, inf: str, init: str) -> Path:
    e_digit = str("%.e" % Decimal(EPSILON))[-1]
    suffix = "_randinit" if init == "random" else ""
    fname = f"dggs_e{e_digit}_d{DELTA}_s{MAX_STEPS}_body-{body}_inf-{inf}{suffix}.pkl"
    return OUTPUT_DIR / fname


def _load_existing(out_path: Path) -> list:
    if out_path.exists():
        try:
            with open(out_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return []
    return []


def stable_hash(name: str) -> int:
    digest = hashlib.sha256(name.encode()).digest()
    return (digest[0] << 8) | digest[1]


# ─── File selection ───────────────────────────────────────────────────────────

pattern_s = re.compile(r"_s(\d+)_")
param_pat = re.compile(
    r"_s(?P<s>\d+)_"
    r"n(?P<n>[\d.]+)_"
    r"a(?P<a>[\d.]+)_"
    r"r(?P<r>\d+)_"
    r"b(?P<b>\d+)"
)

TIMEOUT_RECORD = INPUT_DIR / f"00_timed_out_{TIMEOUT_SECONDS}s.txt"
try:
    with open(TIMEOUT_RECORD) as f:
        timed_out_stems = {ln.strip() for ln in f if ln.strip()}
except FileNotFoundError:
    timed_out_stems = set()


def _select_aba_paths() -> list:
    all_aba = sorted(INPUT_DIR.glob("*.aba"))
    paths = [
        p for p in all_aba
        if (m := pattern_s.search(p.name))
           and MIN_SENTENCES <= int(m.group(1)) <= MAX_SENTENCES
           and p.stem not in timed_out_stems
    ]
    if MAX_FILES > 0:
        paths = paths[:MAX_FILES]
    return paths


def should_skip(path: Path, by_file: dict) -> bool:
    if RESULT_OVERRIDE:
        return False
    return len(by_file.get(path.stem, [])) >= 1


# ─── Non-flat check ──────────────────────────────────────────────────────────

def is_disk_flat(p: Path) -> bool:
    lines = p.read_text().splitlines()
    assumps = {
        parts[1]
        for ln in lines if ln.startswith("a ")
        for parts in [ln.split()] if len(parts) > 1
    }
    for ln in lines:
        if not ln.startswith("r "):
            continue
        parts = ln.split()
        if len(parts) >= 2 and parts[1] in assumps:
            return False
    return True


# ─── DGGS runner ─────────────────────────────────────────────────────────────

def run_dggs_collect(abaf: ABAF, body_kernel, inf_kernel) -> dict:
    runner = DGGSRunner(
        abaf,
        body_agg=body_kernel,
        influence=inf_kernel,
        max_iter=MAX_STEPS,
        epsilon=EPSILON,
        window=DELTA,
    )

    initial_strengths = {a.name: a.tau for a in abaf.assumptions}
    state  = runner.initialise()
    result = runner.run(initial_state=state)

    final_strengths = result.final_state.assumptions
    conv_time = result.n_iterations if result.converged else None

    per_arg = {name: result.converged for name in final_strengths}
    total = len(per_arg)
    prop_conv = sum(per_arg.values()) / total if total else 0.0

    return {
        "initial_strengths": initial_strengths,
        "final_strengths":   final_strengths,
        "global_converged":  result.converged,
        "prop_converged":    prop_conv,
        "per_arg":           per_arg,
        "convergence_time":  conv_time,
        "num_assumptions":   len(abaf.assumptions),
        "num_rules":         len(abaf.rules),
        "num_sentences":     len(abaf.sentences),
    }


# ─── Worker ──────────────────────────────────────────────────────────────────

def worker_file(aba_path_str: str, params: dict, body: str, inf: str,
                init: str, queue: Queue):
    try:
        aba_path = Path(aba_path_str)

        dg = DependencyGraph()
        dg.create_from_file(str(aba_path))

        tau_a = {}
        if init == "random":
            for name in sorted(dg.assumptions):
                random.seed(SEED + stable_hash(name) % 1000)
                tau_a[name] = random.uniform(0.0, 1.0)

        abaf = ABAF.from_dependency_graph(dg, tau_a=tau_a)
        is_non_flat = not is_disk_flat(aba_path)

        body_kernel = BODY_KERNELS[body]()
        inf_kernel  = INF_KERNELS[inf]()

        metrics = run_dggs_collect(abaf, body_kernel, inf_kernel)

        entry = {
            "file":        aba_path.name,
            "file_path":   str(aba_path),
            "model":       "DGGS",
            "body_kernel": body_kernel.name,
            "inf_kernel":  inf_kernel.name,
            **params,
            "non_flat":    is_non_flat,
            "timeout":     False,
            **metrics,
        }
        queue.put(entry)

    except Exception:
        queue.put({"__error__": traceback.format_exc()})


def run_file_with_timeout(aba_path: Path, params: dict, body: str, inf: str,
                          init: str, timeout: int) -> dict:
    q = Queue()
    p = Process(target=worker_file,
                args=(str(aba_path), params, body, inf, init, q))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        print(f"⚠️  Timeout: {aba_path.name}", flush=True)
        return {
            "file":      aba_path.name,
            "file_path": str(aba_path),
            "model":     "DGGS",
            **params,
            "non_flat":  not is_disk_flat(aba_path),
            "timeout":   True,
        }

    if not q.empty():
        msg = q.get()
        if isinstance(msg, dict) and "__error__" in msg:
            print(f"\n⚠️  Worker crashed on {aba_path.name}:\n", flush=True)
            print(msg["__error__"], file=sys.stderr)
            raise RuntimeError(f"Worker for {aba_path.name} raised an exception")
        return msg

    if p.exitcode == -signal.SIGTERM:
        print(f"⚠️  OOM: {aba_path.name}", flush=True)
        return {
            "file":      aba_path.name,
            "file_path": str(aba_path),
            "model":     "DGGS",
            **params,
            "non_flat":  not is_disk_flat(aba_path),
            "timeout":   False,
            "oom":       True,
        }

    if p.exitcode != 0:
        raise RuntimeError(
            f"Worker for {aba_path.name} exited with code {p.exitcode} and no traceback."
        )

    raise RuntimeError(f"Worker for {aba_path.name} returned no result.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def process_once(body: str, inf: str, init: str):
    out_path = _output_pkl(body, inf, init)
    results  = _load_existing(out_path)
    by_file  = defaultdict(list)
    for r in results:
        by_file[Path(r["file"]).stem].append(r)

    aba_paths = _select_aba_paths()

    body_name = BODY_KERNELS[body]().name
    inf_name  = INF_KERNELS[inf]().name
    print(
        f"\n=== DGGS convergence  |  ε={EPSILON}  δ={DELTA}  steps={MAX_STEPS}"
        f"  body={body_name}  influence={inf_name}  init={init}  files={len(aba_paths)} ==="
    )
    print(f"    Output → {out_path}\n", flush=True)

    for aba_path in tqdm(aba_paths, desc="Files", unit="file"):
        if should_skip(aba_path, by_file):
            continue

        m = param_pat.search(aba_path.name)
        params = (
            dict(s=int(m.group("s")), n=float(m.group("n")),
                 a=float(m.group("a")), r=int(m.group("r")), b=int(m.group("b")))
            if m else dict(s=None, n=None, a=None, r=None, b=None)
        )

        print(f"\n--- {aba_path.name} ---", flush=True)
        entry = run_file_with_timeout(aba_path, params, body, inf, init, TIMEOUT_SECONDS)
        results.append(entry)

        with open(out_path, "wb") as pf:
            pickle.dump(results, pf)

    print(f"\n✅  Done — {len(results)} entries in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DGGS convergence experiments with configurable kernels."
    )
    parser.add_argument(
        "--body", choices=["prod", "min"], default="prod",
        help="Body aggregation kernel: prod=ProductBody, min=MinBody (default: prod)"
    )
    parser.add_argument(
        "--inf", choices=["lin", "quad"], default="lin",
        help="Influence function: lin=LinearInfluence, quad=QuadraticInfluence (default: lin)"
    )
    parser.add_argument(
        "--init", choices=["fixed", "random"], default="fixed",
        help="Assumption base score initialisation: fixed=0.5, random∈[0,1] (default: fixed)"
    )
    args = parser.parse_args()
    process_once(args.body, args.inf, args.init)
"""
test_convergence.py — Convergence experiments for DGGS semantics.

For each .aba file in INPUT_DIR, runs the DGGS gradual-semantics iteration
(defined in run_dggs.py) and records:

  Entry fields (one entry per file):
    file, file_path          — filename and full path
    model                    — always "DGGS"
    s, n, a, r, b            — parameters extracted from the filename
    num_assumptions          — number of assumptions in the framework
    num_rules                — number of rules
    num_sentences            — number of non-assumption claim atoms (sentences)
    non_flat                 — True if any rule derives an assumption directly
    initial_strengths        — {assumption_name: tau} at t=0
    final_strengths          — {assumption_name: strength} at the last iteration
    global_converged         — True if ALL assumptions converged (max change
                               below EPSILON over the last DELTA iterations)
    prop_converged           — fraction of assumptions that individually converged
    per_arg                  — {assumption_name: bool} per-assumption convergence
    convergence_time         — iteration index when global convergence was
                               first reached, or None if it was not reached
    timeout                  — True if the subprocess hit TIMEOUT_SECONDS

Results are pickled to OUTPUT_DIR as:
  dggs_e<exp>_d<DELTA>_s<MAX_STEPS>[_randinit].pkl

No intermediate caching is performed: DGGS operates directly on parsed .aba
files, so the DependencyGraph→ABAF construction is fast (milliseconds per
file) and does not warrant disk caching.
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
sys.path.insert(0, str(_HERE.parent))  # project root → scr/

# Role: batch convergence experiment — runs DGGS over data/abaf/*.aba, records per-file metrics.
# Uses DGGSRunner from scr.dggs. Handles random vs fixed tau_a init; subprocess isolation with timeout.

from tqdm import tqdm

from scr.dependency_graph import DependencyGraph
from scr.ABAF import ABAF
from scr.dggs import DGGSRunner

# ─── Config ──────────────────────────────────────────────────────────────────
INPUT_DIR       = Path("data/abaf/").resolve()
OUTPUT_DIR      = Path("convergence_results_dggs/")

MAX_FILES       = 0        # 0 = no limit
MIN_SENTENCES   = 0        # minimum s-parameter in filename filter
MAX_SENTENCES   = 100      # maximum s-parameter in filename filter
TIMEOUT_SECONDS = 600      # per-file subprocess timeout (seconds)

EPSILON         = 1e-3     # convergence tolerance
DELTA           = 5        # stability window (consecutive iterations below EPSILON)
MAX_STEPS       = 5000     # hard iteration cap

BASE_SCORES     = 'random' # 'random' → uniform tau_a in [0,1];
                           # ''       → default tau_a = 0.5 for every assumption

RESULT_OVERRIDE = False    # set True to re-run files that already have results
SEED            = 42       # global base seed for reproducible tau draws
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)


def _output_pkl() -> Path:
    e_digit = str('%.e' % Decimal(EPSILON))[-1]
    base_init = '_randinit' if BASE_SCORES == 'random' else ''
    fname = f"dggs_e{e_digit}_d{DELTA}_s{MAX_STEPS}{base_init}.pkl"
    return OUTPUT_DIR / fname


def _load_existing_results(out_path: Path):
    if out_path.exists():
        try:
            with open(out_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return []
    return []


# ─── Stable hash for reproducible per-assumption seeds ───────────────────────

def stable_hash(name: str) -> int:
    """First two bytes of SHA-256(name) combined into a 16-bit int."""
    digest = hashlib.sha256(name.encode()).digest()
    return (digest[0] << 8) | digest[1]


# ─── File selection ───────────────────────────────────────────────────────────

pattern_s = re.compile(r"_s(\d+)_")

TIMEOUT_RECORD = INPUT_DIR / f"00_timed_out_{TIMEOUT_SECONDS}s.txt"
try:
    with open(TIMEOUT_RECORD, "r") as f:
        timed_out_stems = {line.strip() for line in f if line.strip()}
except FileNotFoundError:
    timed_out_stems = set()


def _select_aba_paths():
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


param_pat = re.compile(
    r"_s(?P<s>\d+)_"
    r"n(?P<n>[\d.]+)_"
    r"a(?P<a>[\d.]+)_"
    r"r(?P<r>\d+)_"
    r"b(?P<b>\d+)"
)


def should_skip(path: Path, by_file: dict) -> bool:
    if RESULT_OVERRIDE:
        return False
    return len(by_file.get(path.stem, [])) >= 1


# ─── Non-flat check ──────────────────────────────────────────────────────────

def is_disk_flat(p: Path) -> bool:
    """Return True if no rule has an assumption as its head."""
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

def run_dggs_collect(
    abaf: ABAF,
    max_iterations: int,
    epsilon: float,
    stability_window: int,
) -> dict:
    """Run DGGS via DGGSRunner and collect convergence metrics for PKL storage."""
    runner = DGGSRunner(abaf, max_iter=max_iterations, epsilon=epsilon, window=stability_window)

    initial_strengths = {a.name: a.tau for a in abaf.assumptions}

    state  = runner.initialise()
    result = runner.run(initial_state=state)

    final_strengths = result.final_state.assumptions
    conv_time = result.n_iterations if result.converged else None

    # per_arg: all True when converged (global criterion), all False otherwise.
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

def worker_file(aba_path_str: str, params: dict, queue: Queue):
    """
    Subprocess entry point: parse one .aba file, run DGGS, push result entry.
    """
    try:
        aba_path = Path(aba_path_str)

        dg = DependencyGraph()
        dg.create_from_file(str(aba_path))

        tau_a = {}
        if BASE_SCORES == 'random':
            for name in sorted(dg.assumptions):
                random.seed(SEED + stable_hash(name) % 1000)
                tau_a[name] = random.uniform(0.0, 1.0)
        abaf = ABAF.from_dependency_graph(dg, tau_a=tau_a)

        is_non_flat = not is_disk_flat(aba_path)

        metrics = run_dggs_collect(abaf, MAX_STEPS, EPSILON, DELTA)

        entry = {
            "file":      aba_path.name,
            "file_path": str(aba_path),
            "model":     "DGGS",
            **params,
            "non_flat":  is_non_flat,
            "timeout":   False,
            **metrics,
        }
        queue.put(entry)

    except Exception:
        queue.put({"__error__": traceback.format_exc()})


def run_file_with_timeout(aba_path: Path, params: dict, timeout: int):
    q = Queue()
    p = Process(target=worker_file, args=(str(aba_path), params, q))
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


# ─── Main experiment ─────────────────────────────────────────────────────────

def process_once():
    out_path = _output_pkl()
    results  = _load_existing_results(out_path)
    by_file  = defaultdict(list)
    for r in results:
        by_file[Path(r["file"]).stem].append(r)

    aba_paths = _select_aba_paths()

    print(
        f"\n=== DGGS convergence  |  ε={EPSILON}  δ={DELTA}  steps={MAX_STEPS}"
        f"  init={BASE_SCORES or 'default(0.5)'}  files={len(aba_paths)} ==="
    )
    print(f"    Output → {out_path}\n", flush=True)

    for aba_path in tqdm(aba_paths, desc="Files", unit="file"):
        if should_skip(aba_path, by_file):
            print(f"⏭   Skipping {aba_path.name} (already run)", flush=True)
            continue

        m = param_pat.search(aba_path.name)
        params = (
            dict(s=int(m.group("s")), n=float(m.group("n")),
                 a=float(m.group("a")), r=int(m.group("r")), b=int(m.group("b")))
            if m else dict(s=None, n=None, a=None, r=None, b=None)
        )

        print(f"\n--- {aba_path.name} ---", flush=True)
        entry = run_file_with_timeout(aba_path, params, TIMEOUT_SECONDS)
        results.append(entry)

        with open(out_path, "wb") as pf:
            pickle.dump(results, pf)

    print(f"\n✅  Done — {len(results)} entries in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGGS convergence experiment.")
    parser.add_argument(
        "--init", choices=["random", "fixed"], default="random",
        help="Assumption base score initialisation: random ∈ [0,1] or fixed 0.5 (default: random)"
    )
    args = parser.parse_args()
    BASE_SCORES = 'random' if args.init == 'random' else ''
    process_once()

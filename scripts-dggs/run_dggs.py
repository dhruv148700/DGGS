#!/usr/bin/env python3
"""
run_dggs.py — CLI wrapper: run DGGS gradual semantics on a single .aba file.

Usage
-----
    python run_dggs.py framework.aba [OPTIONS]

Options
-------
    --tau-a    JSON dict of assumption base scores  (default 0.5 each)
               e.g. '{"a": 0.6, "b": 0.5, "c": 0.7}'
    --tau-r    JSON dict of rule reliabilities keyed by rule index int (default 1.0 each)
               e.g. '{"1": 1.0, "2": 0.8}'
    --body     body aggregation kernel: product | min  (default: product)
    --claim    claim aggregation kernel: max | sum | mean  (default: max)
    --support  support aggregation kernel: max | sum | mean  (default: max)
    --iota     influence kernel: linear | quadratic  (default: linear)
    --max-iter hard iteration cap  (default 200)
    --epsilon  convergence tolerance  (default 1e-3)
    --window   stability window size  (default 5)
    --init     fixed | random  (default: fixed — 0.5 for unspecified assumptions)
    --seed     base random seed for --init random  (default 42)
    --verbose  print per-iteration delta
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scr.dependency_graph import DependencyGraph
from scr.ABAF import ABAF
from scr.dggs import DGGSRunner
from scr.dggs.kernels import (
    ProductBody, MinBody,
    MaxAggregation, SumAggregation, MeanAggregation,
    LinearInfluence, QuadraticInfluence,
)

_BODY_KERNELS = {"product": ProductBody, "min": MinBody}
_AGG_KERNELS  = {"max": MaxAggregation, "sum": SumAggregation, "mean": MeanAggregation}
_IOTA_KERNELS = {"linear": LinearInfluence, "quadratic": QuadraticInfluence}


def _stable_hash(name: str) -> int:
    digest = hashlib.sha256(name.encode()).digest()
    return (digest[0] << 8) | digest[1]


def main():
    parser = argparse.ArgumentParser(
        description="Run DGGS gradual semantics on a .aba framework file."
    )
    parser.add_argument("aba_file")
    parser.add_argument("--tau-a",   default="{}")
    parser.add_argument("--tau-r",   default="{}")
    parser.add_argument("--body",    choices=list(_BODY_KERNELS), default="product")
    parser.add_argument("--claim",   choices=list(_AGG_KERNELS),  default="max")
    parser.add_argument("--support", choices=list(_AGG_KERNELS),  default="max")
    parser.add_argument("--iota",    choices=list(_IOTA_KERNELS), default="linear")
    parser.add_argument("--max-iter", type=int,   default=200, dest="max_iter")
    parser.add_argument("--epsilon",  type=float, default=1e-3)
    parser.add_argument("--window",   type=int,   default=5)
    parser.add_argument("--init",    choices=["fixed", "random"], default="fixed")
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tau_a_override = json.loads(args.tau_a)
    tau_r          = {int(k): v for k, v in json.loads(args.tau_r).items()}

    dg = DependencyGraph()
    dg.create_from_file(args.aba_file)

    tau_a: dict = {}
    for name in sorted(dg.assumptions):
        if name in tau_a_override:
            tau_a[name] = tau_a_override[name]
        elif args.init == "random":
            random.seed(args.seed + _stable_hash(name) % 1000)
            tau_a[name] = random.uniform(0.0, 1.0)

    abaf = ABAF.from_dependency_graph(dg, tau_a=tau_a, tau_r=tau_r)
    init_label = f"random (seed={args.seed})" if args.init == "random" else "fixed 0.5"
    print(f"Loaded: {abaf}")
    print(f"  {len(abaf.assumptions)} assumptions, "
          f"{len(abaf.sentences)} claims, "
          f"{len(abaf.rules)} rules  [init={init_label}]\n")

    runner = DGGSRunner(
        abaf,
        body_agg    = _BODY_KERNELS[args.body](),
        claim_agg   = _AGG_KERNELS[args.claim](),
        support_agg = _AGG_KERNELS[args.support](),
        influence   = _IOTA_KERNELS[args.iota](),
        max_iter    = args.max_iter,
        epsilon     = args.epsilon,
        window      = args.window,
    )

    if args.verbose:
        state = runner.initialise()
        print(f"{'t':>5} | {'Δmax':>12}")
        print("-" * 22)
        print(f"{'0':>5} | {'—':>12}")
        from scr.dggs.runner import RunResult
        recent = []
        for t in range(1, args.max_iter + 1):
            prev = state.assumptions
            state = runner.iterate(state)
            delta = max(abs(state.assumptions[k] - prev[k]) for k in prev)
            print(f"{t:>5} | {delta:12.8f}")
            recent.append(delta)
            if len(recent) > args.window:
                recent.pop(0)
            if len(recent) == args.window and max(recent) < args.epsilon:
                result = RunResult(state, t, True,
                                   f"converged (max delta < {args.epsilon} "
                                   f"over last {args.window} steps)")
                break
        else:
            result = RunResult(state, args.max_iter, False,
                               f"max iterations reached ({args.max_iter})")
    else:
        result = runner.run()

    print(f"\nStopped after {result.n_iterations} iteration(s) — {result.stop_reason}")
    print("\nFinal assumption strengths:")
    for name in sorted(result.final_state.assumptions):
        print(f"  sigma({name}) = {result.final_state.assumptions[name]:.6f}")


if __name__ == "__main__":
    main()

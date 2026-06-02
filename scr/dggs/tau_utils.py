"""
tau_utils.py — helpers for building tau_a / tau_r from CI score files.

CI scores are stored as flat JSON dicts:
  plain keys  ("a", "b", ...)              → assumption base scores (tau_a)
  pipe keys   ("head|body1|body2", ...)    → rule reliability scores (tau_r)
"""

import json
from typing import Dict, Tuple

from scr.dependency_graph import DependencyGraph


def load_scores(scores_path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Split a .scores.json file into plain (assumption) and pipe (rule) scores."""
    with open(scores_path) as fh:
        raw = json.load(fh)
    plain = {k: v for k, v in raw.items() if "|" not in k}
    rules = {k: v for k, v in raw.items() if "|" in k}
    return plain, rules


def build_tau_a(dg: DependencyGraph, plain_scores: Dict[str, float]) -> Dict[str, float]:
    """Map assumption names to base scores; default 0.5 for anything absent from scores."""
    return {name: plain_scores.get(name, 0.5) for name in dg.assumptions}


def build_tau_r(dg: DependencyGraph, rule_scores: Dict[str, float]) -> Dict[int, float]:
    """Map rule indices to reliability scores; absent rules default to 1.0 in ABAF."""
    tau_r: Dict[int, float] = {}
    for idx, (head, body) in dg.rules.items():
        key = head + "|" + "|".join(body)
        if key in rule_scores:
            tau_r[idx] = rule_scores[key]
    return tau_r

"""Sanity-check: annotate an .aba file with scores from its companion .scores.json."""

import json
import sys
from pathlib import Path


def aba_line_to_score_key(line: str) -> str | None:
    """Return the scores-JSON key for a parsed .aba line, or None if not scoreable."""
    parts = line.split()
    if not parts:
        return None
    tag = parts[0]
    if tag == "a":
        # assumption: key is the name
        return parts[1]
    if tag == "c":
        # contrary: key is the assumption name (first token)
        return parts[1]
    if tag == "r":
        # rule: "head|body1|body2" or "head|" for empty-body rules
        head = parts[1]
        body = parts[2:]
        return head + "|" + "|".join(body)
    return None  # header line "p aba N" etc.


def annotate(aba_path: Path, scores_path: Path, out_path: Path) -> None:
    with open(scores_path) as f:
        scores: dict[str, float] = json.load(f)

    lines = aba_path.read_text().splitlines()
    out_lines = []
    for line in lines:
        key = aba_line_to_score_key(line)
        if key is not None and key in scores:
            out_lines.append(f"{line}  # score={scores[key]:.6f}")
        else:
            out_lines.append(line)

    out_path.write_text("\n".join(out_lines))
    print(f"Written {len(out_lines)} lines → {out_path}")

    matched = sum(1 for l in lines if (k := aba_line_to_score_key(l)) and k in scores)
    print(f"Scored {matched}/{len(lines)} lines  ({len(scores)} score entries total)")


if __name__ == "__main__":
    pairs = [
        ("synthetic_er.aba", "synthetic_er.scores.json", "synthetic_er.annotated.txt"),
        ("synthetic_sf.aba", "synthetic_sf.scores.json", "synthetic_sf.annotated.txt"),
    ]
    root = Path(__file__).parent
    for aba, scores, out in pairs:
        print(f"\n--- {aba} ---")
        annotate(root / aba, root / scores, root / out)

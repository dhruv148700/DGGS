"""
analyse_manifest.py
───────────────────
Analysis of causal_manifest.json (probe-based schema).

  3. Mean complexity by n_nodes        — assumptions, atoms
  A. Probe count distribution          — how many probes per instance
  B. probe_is_sat rate                 — overall and by probe_role
  C. Mean probe_fact_count by role     — fact-set coverage per role
  D. Mean complexity by probe_role     — assumptions / atoms across roles
  E. n_credulous by probe_role         — credulous set size across roles

    python scripts-causal/analyse_manifest.py [path/to/manifest.json]
"""

import json
import os
import sys
from collections import defaultdict

REPO_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "causal_manifest.json")

ROLES = ["initial_full", "easy_sat", "boundary_sat", "boundary_unsat"]

# ─── Load manifest ─────────────────────────────────────────────────────────────

with open(MANIFEST_PATH) as fh:
    manifest = json.load(fh)

print(f"Loaded {len(manifest)} entries from {MANIFEST_PATH}")

def sep(title=""):
    width = 72
    if title:
        pad = max(0, width - len(title) - 3)
        print(f"\n── {title} {'─' * pad}")
    else:
        print("─" * width)

node_counts = sorted({e["n_nodes"] for e in manifest})

# Group by instance_id for probe-count analysis
by_instance = defaultdict(list)
for entry in manifest:
    by_instance[entry["instance_id"]].append(entry)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MEAN COMPLEXITY BY n_nodes
# ═══════════════════════════════════════════════════════════════════════════════
sep("3  MEAN COMPLEXITY BY n_nodes  (assumptions / atoms)")

print(f"  {'n':>3}  {'entries':>7}  {'mean_asms':>10}  {'mean_atoms':>11}")
print(f"  {'─'*3}  {'─'*7}  {'─'*10}  {'─'*11}")
for n in node_counts:
    sub = [e for e in manifest if e["n_nodes"] == n]
    print(
        f"  {n:>3}  {len(sub):>7}"
        f"  {sum(e['n_assumptions'] for e in sub)/len(sub):>10.1f}"
        f"  {sum(e['n_atoms']       for e in sub)/len(sub):>11.1f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# A. PROBE COUNT DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
sep("A  PROBE COUNT DISTRIBUTION  (probes generated per instance)")

n_instances = len(by_instance)
count_dist = defaultdict(int)
for probes in by_instance.values():
    count_dist[len(probes)] += 1

print(f"  Total instances : {n_instances}")
print(f"  Total entries   : {len(manifest)}")
print()
print(f"  {'#probes':>8}  {'instances':>10}  {'%':>7}")
print(f"  {'─'*8}  {'─'*10}  {'─'*7}")
for k in sorted(count_dist):
    print(f"  {k:>8}  {count_dist[k]:>10}  {count_dist[k]/n_instances:>7.1%}")

# Also break down by n_nodes
print()
print(f"  mean probes per instance by n_nodes:")
print(f"  {'n':>3}  {'instances':>10}  {'mean_probes':>12}")
print(f"  {'─'*3}  {'─'*10}  {'─'*12}")
for n in node_counts:
    inst = {iid: ps for iid, ps in by_instance.items()
            if ps[0]["n_nodes"] == n}
    if not inst:
        continue
    mean_p = sum(len(ps) for ps in inst.values()) / len(inst)
    print(f"  {n:>3}  {len(inst):>10}  {mean_p:>12.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# B. probe_is_sat RATE
# ═══════════════════════════════════════════════════════════════════════════════
sep("B  probe_is_sat RATE  (fraction of entries / instances with stable extensions)")

n_sat = sum(1 for e in manifest if e["probe_is_sat"])
print(f"  Overall (all entries)    : {n_sat}/{len(manifest)}  ({n_sat/len(manifest):.1%})")

# Per instance: does it have any SAT probe? (always yes by construction, shown for completeness)
n_inst_sat = sum(1 for ps in by_instance.values() if any(e["probe_is_sat"] for e in ps))
print(f"  Instances with any SAT   : {n_inst_sat}/{n_instances}  ({n_inst_sat/n_instances:.1%})")

# initial_full SAT = full fact set already consistent (no binary search needed)
full_entries = [e for e in manifest if e["probe_role"] == "initial_full"]
n_full_sat   = sum(1 for e in full_entries if e["probe_is_sat"])
if full_entries:
    print(f"  Full set SAT (no search) : {n_full_sat}/{len(full_entries)}  ({n_full_sat/len(full_entries):.1%})")

print()
print(f"  By probe_role:")
print(f"  {'role':<16}  {'entries':>7}  {'sat':>6}  {'sat%':>7}")
print(f"  {'─'*16}  {'─'*7}  {'─'*6}  {'─'*7}")
for role in ROLES:
    sub = [e for e in manifest if e["probe_role"] == role]
    if not sub:
        continue
    n_s = sum(1 for e in sub if e["probe_is_sat"])
    print(f"  {role:<16}  {len(sub):>7}  {n_s:>6}  {n_s/len(sub):>7.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# C. MEAN probe_fact_count BY ROLE
# ═══════════════════════════════════════════════════════════════════════════════
sep("C  MEAN probe_fact_count BY ROLE  (fact-set coverage per role)")

# Also show as % of full probe's fact count for the same instance
full_fact_count = {e["instance_id"]: e["probe_fact_count"]
                   for e in manifest if e["probe_role"] == "initial_full"}

print(f"  {'role':<16}  {'entries':>7}  {'mean_facts':>11}  {'% of full':>10}")
print(f"  {'─'*16}  {'─'*7}  {'─'*11}  {'─'*10}")
for role in ROLES:
    sub = [e for e in manifest if e["probe_role"] == role]
    if not sub:
        continue
    mean_fc = sum(e["probe_fact_count"] for e in sub) / len(sub)
    # % of full: only for entries whose instance also has a full probe
    pct_vals = [
        e["probe_fact_count"] / full_fact_count[e["instance_id"]]
        for e in sub
        if e["instance_id"] in full_fact_count and full_fact_count[e["instance_id"]] > 0
    ]
    pct_str = f"{sum(pct_vals)/len(pct_vals):>9.1%}" if pct_vals else "       n/a"
    print(f"  {role:<16}  {len(sub):>7}  {mean_fc:>11.1f}  {pct_str:>10}")

print()
print(f"  By n_nodes (mean probe_fact_count):")
print(f"  {'n':>3}  " + "  ".join(f"{r:<14}" for r in ROLES))
print(f"  {'─'*3}  " + "  ".join("─"*14 for _ in ROLES))
for n in node_counts:
    row = f"  {n:>3}  "
    for role in ROLES:
        sub = [e for e in manifest if e["n_nodes"] == n and e["probe_role"] == role]
        if sub:
            row += f"{sum(e['probe_fact_count'] for e in sub)/len(sub):>14.1f}  "
        else:
            row += f"{'—':>14}  "
    print(row)


# ═══════════════════════════════════════════════════════════════════════════════
# D. MEAN COMPLEXITY BY PROBE ROLE
# ═══════════════════════════════════════════════════════════════════════════════
sep("D  MEAN COMPLEXITY BY PROBE ROLE  (assumptions / atoms)")

print(f"  {'role':<16}  {'entries':>7}  {'mean_asms':>10}  {'mean_atoms':>11}")
print(f"  {'─'*16}  {'─'*7}  {'─'*10}  {'─'*11}")
for role in ROLES:
    sub = [e for e in manifest if e["probe_role"] == role]
    if not sub:
        continue
    print(
        f"  {role:<16}  {len(sub):>7}"
        f"  {sum(e['n_assumptions'] for e in sub)/len(sub):>10.1f}"
        f"  {sum(e['n_atoms']       for e in sub)/len(sub):>11.1f}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# E. n_credulous BY PROBE ROLE
# ═══════════════════════════════════════════════════════════════════════════════
sep("E  n_credulous BY PROBE ROLE  (credulous set size)")

print(f"  {'role':<16}  {'entries':>7}  {'mean_cred':>10}  {'% with cred>0':>14}")
print(f"  {'─'*16}  {'─'*7}  {'─'*10}  {'─'*14}")
for role in ROLES:
    sub = [e for e in manifest if e["probe_role"] == role]
    if not sub:
        continue
    mean_c  = sum(e["n_credulous"] for e in sub) / len(sub)
    has_any = sum(1 for e in sub if e["n_credulous"] > 0)
    print(f"  {role:<16}  {len(sub):>7}  {mean_c:>10.1f}  {has_any/len(sub):>14.1%}")

# ═══════════════════════════════════════════════════════════════════════════════
# F. SANITY CHECK  n_credulous <= n_assumptions
# ═══════════════════════════════════════════════════════════════════════════════
sep("F  SANITY CHECK  n_credulous <= n_assumptions")

violations = [e for e in manifest if e["n_credulous"] > e["n_assumptions"]]
if violations:
    print(f"  FAIL — {len(violations)} violation(s) found:")
    for e in violations[:20]:
        print(f"    {e['abaf']}  role={e['probe_role']}"
              f"  n_credulous={e['n_credulous']}  n_assumptions={e['n_assumptions']}")
    if len(violations) > 20:
        print(f"    ... and {len(violations) - 20} more")
else:
    print(f"  OK — n_credulous <= n_assumptions for all {len(manifest)} entries")

sep()
print(f"Done.  {len(manifest)} entries,  {n_instances} instances analysed.\n")

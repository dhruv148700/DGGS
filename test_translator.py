#!/usr/bin/env python3
"""Test script for the LP to ABA translator."""

from scr.causal_aba.lp_to_aba_translator import lp_facts_to_aba_file
from scr.causal_aba.utils import facts_from_file

# Load facts from the five-node example
facts = facts_from_file("data/two_independent_edges.lp")

print(f"Loaded {len(facts)} facts from data/two_independent_edges.lp")
print(f"First few facts: {facts[:5]}")

# Run the translator
lp_facts_to_aba_file(
    facts,
    n_nodes=4,
    out_path="out.aba",
    optimise_remove_edges=True,
)

print("Translation complete! Output written to out.aba")

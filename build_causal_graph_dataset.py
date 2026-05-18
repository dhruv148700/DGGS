"""
build_causal_graph_dataset.py
─────────────────────────────
Load the paired (input/output) .aba files under input_data_causal/ and
output_data_causal/ via scr.data_utils.load_dataset and serialise the
resulting heterograph list as causal_all.bin.

load_dataset() already handles the input_*/output_<name>.aba pairing,
so this script just adds 'input_data_causal' / 'output_data_causal' as
a new directory pair alongside the existing iccma / generated ones.

    python build_causal_graph_dataset.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scr"))

import dgl

from scr.data_utils import load_dataset


INPUT_DIR  = "./input_data_causal"
OUTPUT_DIR = "./output_data_causal"
BIN_PATH   = "causal_all.bin"


def main():
    print("generating causal dataset graphs")
    graphs = load_dataset(input_directory=INPUT_DIR, output_directory=OUTPUT_DIR)
    dgl.save_graphs(BIN_PATH, graphs)
    print(f"wrote {len(graphs)} graphs to {BIN_PATH}")


if __name__ == "__main__":
    main()

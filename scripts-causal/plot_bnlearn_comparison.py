"""
Plot causal discovery baselines vs DGGS variants on bnlearn datasets.
Datasets : cancer, earthquake, survey
Baselines: Random, FGS, MPC, ABAPC
Extras   : DGGS (reject_off), DGGS (opt) (reject_on)

Outputs (JPEG only)
-------------------
- results/causal_recovery/bnlearn_NSID.jpeg
- results/causal_recovery/bnlearn_NSHD_F1.jpeg
- results/causal_recovery/bnlearn_prec_rec.jpeg
- results/causal_recovery/bnlearn_runtime.jpeg
- results/causal_recovery/bnlearn_summary.json   <- mean ± std for DGGS / DGGS (opt) only
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent
ACD_DIR     = REPO_ROOT / 'ArgCausalDisco'
RESULTS_DIR = ACD_DIR / 'results'
FIGS_DIR    = REPO_ROOT / 'results' / 'causal_recovery'
JSON_DIR    = REPO_ROOT / 'merged_results_bnlearn_eval'
SUMMARY_OUT = FIGS_DIR / 'bnlearn_summary.json'

sys.path.insert(0, str(ACD_DIR))
sys.path.insert(0, str(ACD_DIR / 'utils'))

from plotting import (
    double_bar_chart_plotly,
    sec_blue, sec_orange, main_purple, main_green,
)

# ── Dataset metadata ──────────────────────────────────────────────────────────
DATASETS  = ['cancer', 'earthquake', 'survey']
ARCS_MAP  = {'cancer': 4, 'earthquake': 4, 'survey': 6}
NODES_MAP = {'cancer': 5, 'earthquake': 5, 'survey': 6}

# ── Pluggable extra methods ───────────────────────────────────────────────────
EXTRA_METHODS = [
    {'path': JSON_DIR / 'merged_reject_off.json', 'label': 'DGGS'},
    {'path': JSON_DIR / 'merged_reject_on.json',  'label': 'DGGS (opt)'},
]

# ── Style ─────────────────────────────────────────────────────────────────────
NAMES_DICT = {
    'random':   'Random',
    'fgs':      'FGS',
    'nt':       'NOTEARS-MLP',
    'mpc':      'MPC',
    'abapc':    'ABAPC',
    'dggs':     'DGGS',
    'dggs_opt': 'DGGS (opt)',
}

COLORS_DICT = {
    'random':   '#7f7f7f',
    'fgs':      sec_orange,
    'nt':       main_purple,
    'mpc':      main_green,
    'abapc':    sec_blue,
    'dggs':     '#e377c2',
    'dggs_opt': '#d62728',
}

METHODS_ORDER = ['Random', 'FGS', 'NOTEARS-MLP', 'MPC', 'ABAPC', 'DGGS', 'DGGS (opt)']
FONT_SIZE     = 23

# ── Load .npy baselines ───────────────────────────────────────────────────────
CPDAG_COLS = [
    'dataset', 'model', 'elapsed_mean', 'elapsed_std', 'nnz_mean', 'nnz_std',
    'fdr_mean', 'fdr_std', 'tpr_mean', 'tpr_std', 'fpr_mean', 'fpr_std',
    'precision_mean', 'precision_std', 'recall_mean', 'recall_std',
    'F1_mean', 'F1_std', 'shd_mean', 'shd_std',
    'SID_low_mean', 'SID_low_std', 'SID_high_mean', 'SID_high_std',
]

def load_npy(fname):
    df = pd.DataFrame(
        np.load(RESULTS_DIR / fname, allow_pickle=True),
        columns=CPDAG_COLS,
    )
    df['dataset'] = df['dataset'].astype(str)
    df['model']   = df['model'].astype(str)
    return df

main_npy = load_npy('stored_results_bnlearn_50rep_cpdag.npy')
main_npy = main_npy[
    main_npy['dataset'].isin(DATASETS) &
    main_npy['model'].isin(['Random', 'FGS', 'NOTEARS-MLP', 'ABAPC (Ours)'])
].copy()
main_npy['model'] = main_npy['model'].replace({'ABAPC (Ours)': 'ABAPC'})

mpc_npy = load_npy('stored_results_bnlearn_50rep_mpc_cpdag.npy')
mpc_npy = mpc_npy[mpc_npy['dataset'].isin(DATASETS)].copy()

baselines = pd.concat([main_npy, mpc_npy], ignore_index=True)

baselines['n_edges'] = baselines['dataset'].map(ARCS_MAP).astype(float)
baselines['n_nodes'] = baselines['dataset'].map(NODES_MAP).astype(float)
for col in ['shd', 'SID_low', 'SID_high']:
    baselines[f'p_{col}_mean'] = baselines[f'{col}_mean'].astype(float) / baselines['n_edges']
    baselines[f'p_{col}_std']  = baselines[f'{col}_std'].astype(float)  / baselines['n_edges']

# ── Load and aggregate JSON extras ────────────────────────────────────────────
METRICS_JSON = ['elapsed', 'nnz', 'fdr', 'tpr', 'fpr', 'precision', 'recall', 'f1', 'shd']

def load_json_method(path, label):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for ds in DATASETS:
        runs    = [r for r in data[ds] if r['status'] == 'ok']
        n_edges = ARCS_MAP[ds]
        n_nodes = NODES_MAP[ds]
        row = {'dataset': ds, 'model': label, 'n_edges': float(n_edges), 'n_nodes': float(n_nodes)}
        for metric in METRICS_JSON:
            vals = np.array([r[metric] for r in runs], dtype=float)
            row[f'{metric}_mean'] = float(np.mean(vals))
            row[f'{metric}_std']  = float(np.std(vals, ddof=1))
        row['F1_mean'], row['F1_std'] = row.pop('f1_mean'), row.pop('f1_std')
        for sid in ['sid_low_n', 'sid_high_n']:
            vals = np.array([r[sid] for r in runs], dtype=float)
            row[f'{sid}_mean'] = float(np.mean(vals))
            row[f'{sid}_std']  = float(np.std(vals, ddof=1))
        row['p_shd_mean']      = row['shd_mean'] / n_edges
        row['p_shd_std']       = row['shd_std']  / n_edges
        row['p_SID_low_mean']  = row['sid_low_n_mean']
        row['p_SID_low_std']   = row['sid_low_n_std']
        row['p_SID_high_mean'] = row['sid_high_n_mean']
        row['p_SID_high_std']  = row['sid_high_n_std']
        rows.append(row)
    return pd.DataFrame(rows)

extra_frames = [load_json_method(spec['path'], spec['label']) for spec in EXTRA_METHODS]

# ── Summary: DGGS / DGGS (opt) only ──────────────────────────────────────────
SUMMARY_METRICS = ['elapsed', 'p_shd', 'F1', 'precision', 'recall', 'p_SID_low', 'p_SID_high']
summary = {}

for df in extra_frames:
    for _, row in df.iterrows():
        key = f"{row['model']} | {row['dataset']}"
        summary[key] = {}
        for m in SUMMARY_METRICS:
            if f'{m}_mean' in row.index:
                summary[key][m] = {
                    'mean': round(float(row[f'{m}_mean']), 4),
                    'std':  round(float(row[f'{m}_std']),  4),
                }

FIGS_DIR.mkdir(parents=True, exist_ok=True)
with open(SUMMARY_OUT, 'w') as f:
    json.dump(summary, f, indent=2)
print(f'Summary saved → {SUMMARY_OUT}')

# ── Prettify dataset labels ───────────────────────────────────────────────────
def prettify_dataset(df):
    df = df.copy()
    df['dataset'] = (
        df['dataset'].str.upper()
        + '<br> |V|=' + df['n_nodes'].astype(int).astype(str)
        + ', |E|='    + df['n_edges'].astype(int).astype(str)
    )
    return df

baselines    = prettify_dataset(baselines)
extra_frames = [prettify_dataset(df) for df in extra_frames]

all_sum = pd.concat([baselines] + extra_frames, ignore_index=True)
all_sum['p_SID_low_mean']  = all_sum['p_SID_low_mean'].replace(0, 0.03)
all_sum['p_SID_high_mean'] = all_sum['p_SID_high_mean'].replace(0, 0.03)

methods = [m for m in METHODS_ORDER if m in all_sum['model'].unique()]

# ── Helper: colour lookup ─────────────────────────────────────────────────────
def colour(method):
    key = list(NAMES_DICT.keys())[list(NAMES_DICT.values()).index(method)]
    return COLORS_DICT[key]

# ── Custom single-axis double bar (fixes zero-alignment issue) ────────────────
def plot_double_bar_single_axis(df, var1, var2, label1, label2,
                                methods, range_y, save_path, font_size=23,
                                rect_exp=0.01, total_width_frac=0.63, gap_frac=0.16):
    """
    Both variables share one y-axis so their zeros are guaranteed to align.
    Offsetgroup layout mirrors double_bar_chart_plotly exactly:
      [0..n-1] var1 bars  |  [n] white spacer  |  [n+1..2n] var2 bars
    """
    n = len(methods)
    fig = go.Figure()

    for i, (var, show_legend) in enumerate([(var1, True), (var2, False)]):
        for m, method in enumerate(methods):
            df_m = df[df['model'] == method]
            fig.add_trace(go.Bar(
                x=df_m['dataset'],
                y=df_m[f'{var}_mean'],
                error_y=dict(type='data', array=df_m[f'{var}_std'].tolist(),
                             visible=True, thickness=2),
                name=method,
                offsetgroup=m + n * i + i,
                marker_color=colour(method),
                opacity=0.6,
                showlegend=show_legend,
            ))
        # Invisible spacer after the first group — same pattern as double_bar_chart_plotly
        if i == 0:
            ref_x = df[df['model'] == methods[-1]]['dataset']
            fig.add_trace(go.Bar(
                x=ref_x,
                y=np.zeros(len(ref_x)),
                name='', offsetgroup=n,          # sits between the two clusters
                marker_color='white', opacity=1, showlegend=False,
            ))

    # ── Label tiles: exact same positioning logic as double_bar_chart_plotly ──
    unique_datasets = list(dict.fromkeys(df['dataset'].tolist()))
    n_x_cat     = max(len(unique_datasets), 1)
    cluster_w   = 1.0 / n_x_cat
    total_tw    = min(cluster_w * total_width_frac, cluster_w * 0.9)
    tile_w      = total_tw / 2
    max_gap_av  = max(cluster_w - total_tw, 0.0)
    tile_gap    = min(cluster_w * gap_frac, max_gap_av)
    clust_pad   = max((cluster_w - (tile_w * 2 + tile_gap)) / 2, 0.0)

    top_y0, top_y1 = 1.04, 1.10
    text_y = (top_y0 + top_y1) / 2

    for xi in range(n_x_cat):
        cluster_left = xi * cluster_w + clust_pad
        for li, lbl in enumerate([label1, label2]):
            left  = cluster_left + li * (tile_w + tile_gap)
            right = left + tile_w
            fig.add_shape(type='rect', xref='x domain', yref='y domain',
                          x0=max(0, left - rect_exp), x1=min(1, right + rect_exp),
                          y0=top_y0, y1=top_y1,
                          line=dict(color='#E5ECF6', width=2), fillcolor='#E5ECF6', layer='below')
            fig.add_annotation(xref='x domain', yref='y domain',
                               x=(left + right) / 2, y=text_y,
                               xanchor='center', yanchor='middle',
                               text=f"{' ' * 9}{lbl}{' ' * 9}",
                               showarrow=False,
                               font=dict(size=font_size, color='black'))

    top_margin = max(80, int(font_size * 2.8))
    fig.update_layout(
        barmode='group', bargap=0.08, bargroupgap=0.05,
        legend=dict(orientation='h', xanchor='center', x=0.5, yanchor='top', y=1.23),
        template='plotly_white', width=2000, height=700,
        margin=dict(l=40, r=40, b=70, t=top_margin),
        hovermode='x unified',
        font=dict(size=font_size, family='Serif', color='black'),
    )
    fig.update_yaxes(range=range_y,
                     title=dict(text='Normalised SHD / F1', font=dict(size=font_size)))
    fig.write_image(str(save_path))
    fig.show()

# ── Plot: NSID best / worst ───────────────────────────────────────────────────
_html = str(FIGS_DIR / 'bnlearn_NSID.html')
double_bar_chart_plotly(
    all_sum, ['p_SID_low', 'p_SID_high'], NAMES_DICT, COLORS_DICT, methods,
    save_figs=True, font_size=FONT_SIZE, output_name=_html,
    range_y1=[0, 6], range_y2=[0, 6], rect_exp=0.01,
)
Path(_html).unlink(missing_ok=True)

# ── Plot: NSHD + F1 (single axis so zeros align) ─────────────────────────────
_html = str(FIGS_DIR / 'bnlearn_NSHD_F1.html')
plot_double_bar_single_axis(
    all_sum, 'p_shd', 'F1', 'NSHD', 'F1',
    methods=methods,
    range_y=[0, 2],
    save_path=FIGS_DIR / 'bnlearn_NSHD_F1.jpeg',
    font_size=FONT_SIZE,
)
Path(_html).unlink(missing_ok=True)

# ── Plot: Precision + Recall ──────────────────────────────────────────────────
_html = str(FIGS_DIR / 'bnlearn_prec_rec.html')
double_bar_chart_plotly(
    all_sum, ['precision', 'recall'], NAMES_DICT, COLORS_DICT, methods,
    save_figs=True, font_size=FONT_SIZE, output_name=_html,
    range_y1=[0, 1.3], range_y2=[0, 1.3], rect_exp=0.01,
)
Path(_html).unlink(missing_ok=True)

# ── Plot: Runtime line chart (log scale) ─────────────────────────────────────
def plot_runtime_lines(df, methods, save_path, font_size=23):
    datasets = list(dict.fromkeys(df['dataset'].tolist()))
    fig = go.Figure()
    for method in methods:
        df_m = df[df['model'] == method]
        x, y, err = [], [], []
        for ds in datasets:
            row = df_m[df_m['dataset'] == ds]
            if not row.empty:
                x.append(ds)
                y.append(float(row['elapsed_mean'].values[0]))
                err.append(float(row['elapsed_std'].values[0]))
        fig.add_trace(go.Scatter(
            x=x, y=y,
            error_y=dict(type='data', array=err, visible=True, thickness=2),
            name=method, mode='lines+markers',
            line=dict(color=colour(method), width=2),
            marker=dict(size=8, color=colour(method)),
        ))
    fig.update_yaxes(type='log',
                     title=dict(text='log(elapsed time [s])', font=dict(size=font_size)))
    fig.update_xaxes(title=dict(text='Dataset', font=dict(size=font_size)))
    fig.update_layout(
        template='plotly_white', width=900, height=500,
        font=dict(size=font_size, family='Serif', color='black'),
        legend=dict(orientation='v', xanchor='left', x=1.02, yanchor='middle', y=0.5),
        margin=dict(l=60, r=200, b=80, t=40),
    )
    fig.write_image(str(save_path))
    fig.show()

plot_runtime_lines(
    all_sum, methods,
    save_path=FIGS_DIR / 'bnlearn_runtime.jpeg',
    font_size=FONT_SIZE,
)

print('Done. Figures saved to', FIGS_DIR)

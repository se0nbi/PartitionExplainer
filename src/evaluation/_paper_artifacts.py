"""Generate the paper's leaderboard table, comparison figure, and dataset-statistics table.

Usage:
  python _paper_artifacts.py board   # Table III leaderboard -> stdout + outputs/hpo/board/overall_winners_FINAL16.csv
  python _paper_artifacts.py fig      # Figure -> figures/fig_leaderboard.pdf (characterization vs sparsity, 6 panels)
  python _paper_artifacts.py stats    # Table I dataset statistics -> outputs/dataset_stats.csv
  python _paper_artifacts.py          # default: board
"""
import os
import sys
from collections import Counter
import numpy as np
import pandas as pd
import os as _os, sys as _sys, glob as _glob
_sys.path[:0] = _glob.glob(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "*", ""))
import _board_core as BC


def main_board():
    df = BC.load_base_df()
    cells = BC.compute_cells(df)
    pw = {k: v["winner"] for k, v in cells.items()}
    o = Counter(pw.values())
    tot = sum(o.values())
    print(f"=== FINAL 15-dataset 4-metric board: {tot} pairs ===")
    for m in ["ours"] + BC.BASE:
        if o.get(m, 0):
            print(f"  {m:14s} {o[m]:3d}  ({100*o[m]/tot:4.1f}%)")

    og = Counter(v["winner"] for v in BC.compute_cells(df, ours_vars=("ours_groups_tuned",)).values())
    tg = sum(og.values())
    print(f"\nours_groups single-output: {og.get('ours', 0)}/{tg} ({100*og.get('ours', 0)/tg:.0f}%)")

    print("\nper-metric (ours / 15 datasets):")
    for metric in BC.METRICS:
        cnt = Counter(v for k, v in pw.items() if k[1] == metric)
        others = "  ".join(f"{m}:{c}" for m, c in cnt.most_common() if m != "ours")
        print(f"  {BC.MLABEL[metric]:10s} ours {cnt.get('ours', 0)}/15   {others}")

    for kind, label in [("REAL", "REAL-WORLD"), ("SYN", "SYNTHETIC")]:
        sub = {k: v for k, v in pw.items() if (k[0] in BC.SYN) == (kind == "SYN")}
        c = Counter(sub.values())
        n = len(sub)
        nwin = sum(1 for ds in set(k[0] for k in sub)
                   if Counter(v for k, v in sub.items() if k[0] == ds).most_common(1)[0][0] == "ours")
        print(f"\n=== {label} ({n} pairs): ours {c.get('ours',0)}/{n} ({100*c.get('ours',0)/n:.0f}%); "
              f"ours #1 on {nwin}/{len(set(k[0] for k in sub))} datasets ===")
        for ds in sorted(set(k[0] for k in sub)):
            dc = Counter(v for k, v in sub.items() if k[0] == ds)
            print(f"    {ds:18s} ours {dc.get('ours',0)}/{sum(dc.values())}  winner={dc.most_common(1)[0][0]}")

    os.makedirs("outputs/hpo/board", exist_ok=True)
    pd.DataFrame([{"dataset": k[0], "metric": k[1], "overall_winner": v} for k, v in pw.items()]
                 ).to_csv("outputs/hpo/board/overall_winners_FINAL16.csv", index=False)
    print("\nwrote outputs/hpo/board/overall_winners_FINAL16.csv")


# Connected line figure: Characterization vs sparsity, one line per method, one panel per dataset.
def main_fig():
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    METRIC = "characterization_prob"
    SPS = [0.5, 0.6, 0.7, 0.8, 0.9]
    PANELS = [("mutag", "MUTAG"), ("graph_sst2", "Graph-SST2"), ("benzene", "Benzene"),
              ("spmotif_0.7", "SPMotif-0.7"), ("ba_house_or_grid", "BA-House-OR-Grid"),
              ("spmotif_0.9", "SPMotif-0.9")]
    STYLE = {
        "ours":        ("PARSE",        "#d62728", "o"),
        "gstarx":      ("GStarX",       "#1f77b4", "s"),
        "graphext":    ("GraphEXT",     "#2ca02c", "^"),
        "graphshapiq": ("GraphSHAP-IQ", "#9467bd", "D"),
        "pgexplainer": ("PGExplainer",  "#ff7f0e", "v"),
        "gnnexplainer": ("GNNExplainer", "#8c564b", "P"),
    }

    def load_ds(ds):
        parts = [pd.read_csv(f"outputs/hpo/eval_records/main/eval_records_{ds}.csv")]
        for b in ["graphext", "same", "subgraphx", "graphsvx", "mage"]:
            p = f"outputs/hpo/eval_records/baselines/{b}/eval_records_{b}_{ds}.csv"
            if os.path.exists(p):
                parts.append(pd.read_csv(p))
        g = f"outputs/hpo/eval_records/gsiq25/eval_records_gsiq25_{ds}.csv"
        df = pd.concat(parts, ignore_index=True)
        if os.path.exists(g):
            df = df[df.method != "graphshapiq"]
            df = pd.concat([df, pd.read_csv(g)], ignore_index=True)
        return df

    def series(df, method):
        if method == "ours":
            out = []
            for sp in SPS:
                vs = [df[(df.method == m) & (abs(df.target_sparsity - sp) < 0.02)][METRIC].mean()
                      for m in ["ours_tuned", "ours_groups_tuned"]]
                vs = [v for v in vs if v == v]
                out.append(max(vs) if vs else np.nan)
            return out
        return [df[(df.method == method) & (abs(df.target_sparsity - sp) < 0.02)][METRIC].mean() for sp in SPS]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.85), sharex=True)
    for ax, (ds, title) in zip(axes.flat, PANELS):
        df = load_ds(ds)
        for m, (lab, col, mk) in STYLE.items():
            y = series(df, m)
            if any(v == v for v in y):
                ax.plot(SPS, y, marker=mk, ms=4, lw=1.6 if m == "ours" else 1.1,
                        color=col, label=lab, zorder=5 if m == "ours" else 2,
                        alpha=1.0 if m == "ours" else 0.85)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("Sparsity", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Characterization", fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("figures/fig_leaderboard.pdf", bbox_inches="tight", dpi=300)
    print("wrote figures/fig_leaderboard.pdf")


CONTROLLED = ["AND", "OR", "DIST"]
SYN_BENCH = ["ba_2motifs", "ba_house_grid", "ba_house_or_grid",
             "spmotif_0.5", "spmotif_0.7", "spmotif_0.9"]
REAL_BENCH = ["benzene", "bace", "bbbp", "mutag", "proteins", "enzymes",
              "graph_sst2", "graph_sst5", "graph_twitter"]


def _graph_stats(graphs, n_feat=None, n_cls=None):
    nodes = np.array([int(g.num_nodes) for g in graphs], dtype=float)
    edges = np.array([int(g.edge_index.size(1)) // 2 for g in graphs], dtype=float)  # undirected
    if n_feat is None:
        n_feat = int(graphs[0].x.size(1))
    if n_cls is None:
        ys = [int(g.y.view(-1)[0]) for g in graphs]
        n_cls = int(max(ys)) + 1
    return dict(n_graphs=len(graphs), avg_nodes=nodes.mean(), min_nodes=int(nodes.min()),
                max_nodes=int(nodes.max()), avg_edges=edges.mean(),
                n_features=int(n_feat), n_classes=int(n_cls))


def main_stats():
    import time
    import traceback
    import torch
    import _hpo_engine as E

    E.patch_torch_load()
    try:
        import _newds_loaders
    except Exception as e:
        print(f"(newds loaders: {e})", flush=True)

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DIG_ROOT = os.path.join(ROOT, "outputs", "dig_data")

    ns = E.build_namespace(verbose=False)
    E.inject_overrides(ns)
    E.install_runtime_hooks(ns, robust=False)
    E.install_extra_loaders(ns)

    _base = ns["load_dig_dataset"]
    def loader(name):
        if str(name).lower() in ("graph_sst5", "graphsst5"):
            from dig.xgraph.dataset import SentiGraphDataset
            ds = SentiGraphDataset(DIG_ROOT, name="Graph-SST5")
            gs = [ds[i].clone() for i in range(len(ds))]
            for g in gs:
                g.x = g.x.float()
            return gs, {"name": name, "input_dim": int(ds.num_node_features),
                        "num_classes": int(ds.num_classes)}
        return _base(name)

    rows = {}

    # controlled tasks: generate full N and measure
    N = int(getattr(ns["CFG"], "N_TOTAL_PER_TASK", 8000))
    noise = float(getattr(ns["CFG"], "LABEL_NOISE_RATE", 0.03))
    gens = {"AND": ("generate_dataset_AND", 42), "OR": ("generate_dataset_OR", 43),
            "DIST": ("generate_dataset_DIST", 44)}
    for tag, (fn, seed) in gens.items():
        t = time.time()
        try:
            graphs, df = ns[fn](N, seed, noise)
            st = _graph_stats(graphs, n_feat=int(ns["CFG"].INPUT_DIM))
            st["group"] = "Controlled"
            rows[tag] = st
            print(f"[{tag:13s}] {st['n_graphs']:5d} graphs  avg_n={st['avg_nodes']:.1f} "
                  f"({st['min_nodes']}-{st['max_nodes']})  avg_e={st['avg_edges']:.1f}  "
                  f"feat={st['n_features']} cls={st['n_classes']}  {time.time()-t:.0f}s", flush=True)
        except Exception:
            print(f"[{tag}] FAILED\n{traceback.format_exc()}", flush=True)

    # benchmark datasets
    for group, names in [("Synthetic", SYN_BENCH), ("Real-world", REAL_BENCH)]:
        for name in names:
            t = time.time()
            try:
                graphs, info = loader(name)
                st = _graph_stats(graphs, n_feat=info.get("input_dim"), n_cls=info.get("num_classes"))
                st["group"] = group
                rows[name] = st
                print(f"[{name:13s}] {st['n_graphs']:5d} graphs  avg_n={st['avg_nodes']:.1f} "
                      f"({st['min_nodes']}-{st['max_nodes']})  avg_e={st['avg_edges']:.1f}  "
                      f"feat={st['n_features']} cls={st['n_classes']}  {time.time()-t:.0f}s", flush=True)
            except Exception:
                print(f"[{name}] FAILED\n{traceback.format_exc()}", flush=True)

    order = CONTROLLED + SYN_BENCH + REAL_BENCH
    df = pd.DataFrame([{**{"dataset": k}, **rows[k]} for k in order if k in rows])
    df.to_csv("outputs/dataset_stats.csv", index=False)
    print("\nwrote outputs/dataset_stats.csv")
    print(df.to_string(index=False))


_DISPATCH = {"board": main_board, "fig": main_fig, "stats": main_stats}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "board"
    if cmd not in _DISPATCH:
        print(f"usage: python _paper_artifacts.py [{' | '.join(_DISPATCH)}]")
        sys.exit(2)
    _DISPATCH[cmd]()

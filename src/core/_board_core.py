"""Leaderboard scoring: ranks explainers per (dataset, metric, sparsity) cell on the common
graph_idx intersection and reports the per-(dataset, metric) winner."""
import glob
import os
from collections import Counter
import numpy as np
import pandas as pd

PROD = "outputs/hpo/eval_records"
SYN = {"ba_2motifs", "ba_house_grid", "ba_house_or_grid",
       "spmotif_0.5", "spmotif_0.7", "spmotif_0.9"}
DROP = set()
GSIQ25 = ["ba_2motifs", "bace", "benzene", "proteins"]
BASE = ["gnnexplainer", "pgexplainer", "graphshapiq", "gstarx", "sme", "subgraphx",
        "graphsvx", "mage", "same", "graphext"]
OURS = ["ours_tuned", "ours_groups_tuned"]
# fid_minus_prob is scored closest-to-0 ('abs_low'); winner() routes every non-'higher'
# direction through abs(), so 'abs_low' and 'lower' behave identically.
METRICS = {"harmonic_fidelity": "higher", "fid_plus_prob": "higher",
           "fid_minus_prob": "abs_low", "characterization_prob": "higher"}
MLABEL = {"harmonic_fidelity": "H-Fidelity", "fid_plus_prob": "Fid+",
          "fid_minus_prob": "Fid-", "characterization_prob": "Char"}
SPS = [0.5, 0.6, 0.7, 0.8, 0.9]

# filename substrings whose CSVs are excluded from the board
_EXCLUDE = ("gsiq", "PRE", "pretweak", "stage", "overnight", "_faith")


def _prod_files(prod_dir=PROD):
    # Recursive: records may be grouped into method subfolders (main/, baselines/<m>/)
    # or sit flat in prod_dir. gsiq25/ is matched here too but dropped by _EXCLUDE.
    return [f for f in glob.glob(os.path.join(prod_dir, "**", "eval_records_*.csv"), recursive=True)
            if all(t not in f for t in _EXCLUDE)]


def load_base_df(prod_dir=PROD, ours_override_dir=None, apply_drop=True):
    """Load baseline records plus the GraphSHAP-IQ-25 swap, optionally overriding ours rows per
    dataset. apply_drop=False keeps every dataset instead of removing those in DROP."""
    df = pd.concat([pd.read_csv(f) for f in _prod_files(prod_dir)], ignore_index=True)
    for ds in GSIQ25:
        g = os.path.join(prod_dir, "gsiq25", f"eval_records_gsiq25_{ds}.csv")
        if os.path.exists(g):
            df = df[~((df.dataset == ds) & (df.method == "graphshapiq"))]
            df = pd.concat([df, pd.read_csv(g).assign(dataset=ds)], ignore_index=True)
    if ours_override_dir:
        for p in glob.glob(os.path.join(ours_override_dir, "eval_records_*.csv")):
            ds = os.path.basename(p)[len("eval_records_"):-4]
            ov = pd.read_csv(p)
            ov = ov[ov.method.isin(OURS)]
            if ov.empty:
                continue
            df = df[~((df.dataset == ds) & (df.method.isin(OURS)))]
            df = pd.concat([df, ov.assign(dataset=ds)], ignore_index=True)
    return df[~df.dataset.isin(DROP)] if apply_drop else df


def winner(comp, direction):
    """Argmax for 'higher' metrics; argmin-by-|value| for everything else (Fid- closest-to-0)."""
    return max(comp, key=comp.get) if direction == "higher" else min(comp, key=lambda k: abs(comp[k]))


def compute_cells(df, ours_vars=OURS):
    """{(dataset, metric): {winner, ours, comp_best, direction}} via common-graph averaging.
    ours = best-of(ours_vars)."""
    cells = {}
    for ds in sorted(df.dataset.unique()):
        dd = df[df.dataset == ds]
        for metric, direction in METRICS.items():
            if metric not in dd.columns:
                continue
            per = {}
            for sp in SPS:
                sub = dd[abs(dd.target_sparsity - sp) < 0.02]
                if sub.empty:
                    continue
                present = [m for m in (list(ours_vars) + BASE)
                           if sub[sub.method == m][metric].notna().any()]
                common = (set.intersection(*[set(sub[sub.method == m].dropna(subset=[metric]).graph_idx)
                                             for m in present]) if present else set())
                if not common:
                    continue

                def mean_on(m):
                    s = sub[(sub.method == m) & (sub.graph_idx.isin(common))][metric].dropna()
                    return float(s.mean()) if len(s) else np.nan
                oc = [mean_on(m) for m in ours_vars]
                oc = [v for v in oc if v == v]
                if not oc:
                    continue
                per.setdefault("ours", []).append(max(oc) if direction == "higher" else min(oc, key=abs))
                for b in BASE:
                    v = mean_on(b)
                    if v == v:
                        per.setdefault(b, []).append(v)
            if "ours" not in per:
                continue
            ov = {k: float(np.mean(v)) for k, v in per.items()}
            w = winner(ov, direction)
            comp = {k: v for k, v in ov.items() if k != "ours"}
            cb = (max(comp.values()) if direction == "higher"
                  else min(comp.values(), key=abs)) if comp else None
            cells[(ds, metric)] = {"winner": w, "ours": ov["ours"], "comp_best": cb, "direction": direction}
    return cells


def board_summary(cells):
    c = Counter(v["winner"] for v in cells.values())
    tot = sum(c.values())
    return {"ours": c.get("ours", 0), "total": tot,
            "pct": round(100 * c.get("ours", 0) / tot, 1) if tot else 0.0,
            "by_method": dict(c)}


def pareto_diff(base_cells, cand_cells):
    """gains: loss->win, regressions: win->loss. Only cells present in BOTH are compared."""
    gains, regressions = [], []
    for k in base_cells:
        if k not in cand_cells:
            continue
        bw = base_cells[k]["winner"] == "ours"
        cw = cand_cells[k]["winner"] == "ours"
        if cw and not bw:
            gains.append(k)
        elif bw and not cw:
            regressions.append(k)
    return gains, regressions

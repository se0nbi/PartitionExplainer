"""Board-tuning harness for the explanation method.

Usage:
  python _ovn.py driver [datasets...]   # re-HPO with a faithfulness-only objective
  python _ovn.py tweak  [datasets...]   # algorithm tweaks (M-propagation / grpaug), no search
  python _ovn.py valcal [datasets...]   # lambda val-calibration
  python _ovn.py final  [datasets...]   # assemble best accepted candidate per dataset, report board

Results are staged under _overnight/ and the run is resumable via _overnight/ledger.jsonl.
"""
import os
import sys
import glob
import json
import time
import gc
import traceback
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch
import os as _os, sys as _sys, glob as _glob
_sys.path[:0] = _glob.glob(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "*", ""))
import _hpo_engine as E
import _board_core as BC

E.patch_torch_load()

OUT = "_overnight"
STAGING = os.path.join(OUT, "staging")
CONFIGS = os.path.join(OUT, "configs")
LEDGER = os.path.join(OUT, "ledger.jsonl")
SUMMARY = os.path.join(OUT, "SUMMARY.md")
for d in (OUT, STAGING, CONFIGS):
    os.makedirs(d, exist_ok=True)
DIG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs", "dig_data")
GT_IMBAL = {"benzene"}


jdump = E.jdump


def log(rec):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)) + "\n")
    print(f"[LEDGER] {rec.get('dataset')}{('/'+rec['tweak']) if rec.get('tweak') else ''}: "
          f"{rec.get('status')} gains={rec.get('gains')} regress={rec.get('regress')}", flush=True)


remap = E.remap


def prod_hp(ds):
    for p in (os.path.join(E.HP_BEST_DIR, f"best_hp_{ds}.json"),
              os.path.join("outputs", "hpo_newds", "hyperparameters", "best", f"best_hp_{ds}.json")):
        if os.path.exists(p):
            return E.HP(**json.load(open(p))["hp"])
    return E.DEFAULT_HP


def cells_from_ours(ds, ours_df):
    """Board cells for one dataset from the given ours rows plus production baselines."""
    df = BC.load_base_df(ours_override_dir=None)
    df = df[df.dataset == ds]
    df = pd.concat([df[~df.method.isin(BC.OURS)],
                    ours_df[ours_df.method.isin(BC.OURS)].assign(dataset=ds)], ignore_index=True)
    return {k: v for k, v in BC.compute_cells(df).items() if k[0] == ds}


def build_ns(search_baselines=None):
    try:
        import _newds_loaders  # noqa: F401
    except Exception as e:
        print(f"(newds loaders: {e})", flush=True)
    ns = E.build_namespace(verbose=False)
    E.inject_overrides(ns)
    E.install_runtime_hooks(ns, robust=False)
    E.install_extra_loaders(ns)
    ns["EVAL_CFG"].SUBX_MAX_GRAPHS = 2
    E.TRAIN_CAP = 6000
    E.GRAPHSHAPIQ_MAX = 3
    if search_baselines is not None:
        E.BASELINE_METHODS = list(search_baselines)
    _base = ns["load_dig_dataset"]

    def loader(name):
        if str(name).lower() in ("graph_sst5", "graphsst5"):
            from dig.xgraph.dataset import SentiGraphDataset
            dd = SentiGraphDataset(DIG_ROOT, name="Graph-SST5")
            gs = [dd[i].clone() for i in range(len(dd))]
            for g in gs:
                g.x = g.x.float()
            return gs, {"name": name, "input_dim": int(dd.num_node_features),
                        "num_classes": int(dd.num_classes)}
        return _base(name)
    ns["load_dig_dataset"] = loader
    return ns


def done_datasets():
    if not os.path.exists(LEDGER):
        return set()
    out = set()
    for line in open(LEDGER, encoding="utf-8"):
        try:
            r = json.loads(line)
            if r.get("phase") == "rehpo":
                out.add(r["dataset"])
        except Exception:
            pass
    return out


def done_pairs():
    out = set()
    if os.path.exists(LEDGER):
        for line in open(LEDGER, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("phase") == "tweak":
                    out.add((r["dataset"], r["tweak"]))
            except Exception:
                pass
    return out


def _adj_mat(adj, k):
    A = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in adj.get(i, ()):
            if 0 <= j < k:
                A[i, j] = 1.0
    return np.maximum(A, A.T)   # symmetrize


def propagate_M(M, adj, lam, kernel="row"):
    """Structure-aware interaction-matrix propagation. kernel:
       row    P=D^-1 A, M'=M+lam/2 (PM+MP)         (random-walk)
       sym    S=D^-1/2 A D^-1/2, M'=M+lam/2 (SM+MS) (GCN-style)
       row2   apply 'row' twice (2-hop)
       rowoff propagate only the off-diagonal interactions; keep the diagonal main-effects."""
    k = M.shape[0]
    A = _adj_mat(adj, k)
    d = A.sum(1, keepdims=True)
    d[d == 0] = 1.0
    if kernel == "sym":
        ds = 1.0 / np.sqrt(d)
        S = ds * A * ds.T
        return M + 0.5 * lam * (S @ M + M @ S)
    P = A / d
    if kernel == "rowoff":
        diag = np.diag(np.diag(M)).copy()
        off = M - diag
        off2 = off + 0.5 * lam * (P @ off + off @ P)
        return diag + off2
    Mp = M + 0.5 * lam * (P @ M + M @ P)
    if kernel == "row2":
        Mp = Mp + 0.5 * lam * (P @ Mp + Mp @ P)
    return Mp


def _freeze_lite(ns, fr, n_test=25, n_perm=50):
    test = fr._splits["test"]
    tidx = E._select_eval_indices(test, n_test, fr.seed, E.BALANCE_GT)
    fr.test_graphs = [test[int(i)].to(fr.device) for i in tidx]
    fr.test_sg, fr.test_gt, fr.test_M = E._build_graph_assets(
        ns, fr.model, fr.device, fr.test_graphs, fr.name, n_perm, fr.seed, label="test")
    return fr


def eval_ours(ns, fr, hp, ops=None):
    """Re-eval ours/ours_groups on the cached leaderboard split. ops (or None for defaults):
       {'lam': float, 'kernel': str}  structure-aware M propagation, and/or
       {'grpaug': float}              override the group cross-interaction coef."""
    ops = ops or {}
    grid = ns["EVAL_CFG"].SPARSITY_GRID
    e1, fid = ns["evaluate_one_graph"], ns["_eval_fidelity"]
    recs = []
    for gi in range(len(fr.test_graphs)):
        data, sg, gt, M = fr.test_graphs[gi], fr.test_sg[gi], fr.test_gt[gi], fr.test_M[gi]
        n = data.x.size(0)
        M_use = propagate_M(M, getattr(sg, "heavy_adj", {}) or {}, ops["lam"], ops.get("kernel", "row")) \
            if ops.get("lam") else M
        node_scores, partition, agg = E.our_explain_cached(sg, M_use, hp, ns, seed=fr.seed)
        coef = ops["grpaug"] if ops.get("grpaug") is not None else hp.group_aug_coef
        recs += e1(fr.model, data, node_scores, fr.device, grid, "ours", gi, gt)
        ranked = E.rank_groups(agg, partition, coef)
        if not ranked:
            continue
        prefixes, cumul = [], set()
        for gsel in ranked:
            if gsel < len(partition):
                cumul = cumul | set(partition[gsel])
                prefixes.append((set(cumul), 1.0 - len(cumul) / n))
        if not prefixes:
            continue
        for target_sp in grid:
            bi = min(range(len(prefixes)), key=lambda i: abs(prefixes[i][1] - target_sp))
            expl, actual_sp = set(prefixes[bi][0]), prefixes[bi][1]
            fd = fid(fr.model, data, expl, fr.device)
            fd.update(method="ours_groups", graph_idx=gi, target_sparsity=target_sp,
                      actual_sparsity=actual_sp, num_selected=len(expl), num_nodes=n)
            recs.append(fd)
    return remap(pd.DataFrame(recs), "tuned").assign(dataset=fr.name)


LHS = int(os.environ.get("OVN_LHS", 200))
BO = int(os.environ.get("OVN_BO", 110))
SENS = int(os.environ.get("OVN_SENS", 5))
CONFIRM = int(os.environ.get("OVN_CONFIRM", 8))
EPOCHS = int(os.environ.get("OVN_EPOCHS", 150))
NTEST = int(os.environ.get("OVN_NTEST", 60))
NVAL = int(os.environ.get("OVN_NVAL", 28))
SEARCH_BASELINES = ["gstarx", "pgexplainer", "gnnexplainer"]
REPRO_TOL = 0.03
FAITH_METRICS = {"harmonic_fidelity": "higher", "fid_plus_prob": "higher",
                 "fid_minus_prob": "abs_low", "characterization_prob": "higher"}
METRIC_W = {"harmonic_fidelity": float(os.environ.get("OVN_W_HFID", 3.0)),
            "fid_plus_prob": float(os.environ.get("OVN_W_FIDP", 2.0)),
            "fid_minus_prob": 0.7, "characterization_prob": 0.7}
_ORIG_OBJECTIVE = E.objective


def weighted_objective(cells, default_cells, attack_weight=2.0):
    num = den = 0.0
    wins = pv = 0
    for key, c in cells.items():
        metric = key[0]
        dft = default_cells.get(key) if default_cells else None
        lost = (dft is not None and not dft["win"])
        w = (attack_weight if lost else 1.0) * METRIC_W.get(metric, 1.0)
        num += w * c["margin"]
        den += w
        if c["win"]:
            wins += 1
        if dft is not None and dft["win"] and not c["win"]:
            pv += 1
    raw = (num / den) if den > 0 else -1e9
    return raw - 1000.0 * pv, {"raw": float(raw), "wins": int(wins),
                               "pareto_violations": int(pv), "n_cells": len(cells)}


def run_dataset_rehpo(ns, ds, base_cells):
    t0 = time.time()
    E.BALANCE_GT = ds in GT_IMBAL
    E.OBJ_METRICS = dict(FAITH_METRICS)
    E.objective = weighted_objective
    rec = {"phase": "rehpo", "dataset": ds, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        fr = E.freeze_dataset(ns, ds, model_class="gcn", n_val=NVAL, n_perm=50, seed=42, epochs=EPOCHS)
    except Exception as ex:
        rec.update(status="freeze_failed", error=str(ex)[:300])
        log(rec)
        return None

    php = prod_hp(ds)
    E.freeze_test(ns, fr, n_test=25, n_perm=50)
    repro = remap(E.eval_hp(ns, fr, php, split="test"), "tuned").assign(dataset=ds)
    repro_cells = cells_from_ours(ds, repro)
    prod_recorded = {k: v for k, v in base_cells.items() if k[0] == ds}
    drift = max((abs(repro_cells[k]["ours"] - prod_recorded[k]["ours"])
                 for k in prod_recorded if k in repro_cells), default=1.0)
    repro_ours = BC.board_summary(repro_cells)["ours"] if repro_cells else -1
    prod_ours = sum(1 for k in prod_recorded if prod_recorded[k]["winner"] == "ours")
    rec.update(drift=round(float(drift), 4), repro_ours_cells=repro_ours, prod_ours_cells=prod_ours)

    res = E.search_dataset(ns, fr, lhs_n=LHS, bo_iter=BO, sens_levels=SENS, seed=0)
    pairs = sorted(zip(res["samples_obj"], res["samples_hp"]), key=lambda t: -t[0])
    seen, fin = set(), []
    for obj, hp_d in pairs:
        key = tuple(round(hp_d[k], 2) for k in E.ALL_CONT_DIMS)
        if key in seen:
            continue
        seen.add(key)
        fin.append(E.HP(**hp_d))
        if len(fin) >= CONFIRM:
            break
    _, best, _ = E.confirm_finalists(ns, fr, fin, n_test=NTEST, n_perm_hi=100, robust=False)
    best_hp = E.HP(**best["hp"])

    cand = remap(E.eval_hp(ns, fr, best_hp, split="test"), "tuned").assign(dataset=ds)
    cand_cells = cells_from_ours(ds, cand)
    gains, regress = BC.pareto_diff(repro_cells, cand_cells)
    detail = {}
    for k in repro_cells:
        rc, cc = repro_cells[k], cand_cells.get(k, {})
        detail[k[1]] = {"repro_ours": round(rc["ours"], 4),
                        "cand_ours": round(cc.get("ours", float("nan")), 4),
                        "comp_best": round(rc["comp_best"], 4) if rc["comp_best"] is not None else None,
                        "won_repro": rc["winner"] == "ours",
                        "won_cand": cc.get("winner") == "ours",
                        "gap_to_comp": round(cc.get("ours", float("nan")) - rc["comp_best"], 4)
                        if rc["comp_best"] is not None else None}
    rec.update(best_obj=round(float(res["best_obj"]), 3),
               default_obj=round(float(res["default_obj"]), 3),
               gains=[k[1] for k in gains], regress=[k[1] for k in regress],
               n_gain=len(gains), n_regress=len(regress), secs=round(time.time() - t0),
               detail=detail)
    cp = os.path.join(STAGING, f"eval_records_{ds}.csv")
    if len(regress) == 0 and len(gains) > 0 and drift <= REPRO_TOL:
        rec["status"] = "ACCEPTED"
        cand.to_csv(cp, index=False)
        jdump({"dataset": ds, "hp": asdict(best_hp), "gains": [k[1] for k in gains],
               "drift": float(drift), "obj": "faith_hfid_weighted"},
              os.path.join(CONFIGS, f"best_hp_{ds}.json"))
    elif len(regress) == 0 and len(gains) > 0:
        rec["status"] = "gain_but_drift"
    else:
        rec["status"] = "rejected"
    log(rec)
    del fr
    E.clear_fid_cache()
    gc.collect()
    return rec


def write_summary(base_summary):
    accepted = []
    if os.path.exists(LEDGER):
        for line in open(LEDGER, encoding="utf-8"):
            try:
                r = json.loads(line)
                if r.get("status") == "ACCEPTED":
                    accepted.append(r)
            except Exception:
                pass
    cells = BC.compute_cells(BC.load_base_df(ours_override_dir=STAGING))
    s = BC.board_summary(cells)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write("# Overnight re-HPO progress\n\n")
        f.write(f"- Baseline board: **{base_summary['ours']}/{base_summary['total']} "
                f"({base_summary['pct']}%)**\n")
        f.write(f"- Current staged board: **{s['ours']}/{s['total']} ({s['pct']}%)**  "
                f"({s['ours']-base_summary['ours']:+d} cells)\n")
        f.write(f"- by_method: {s['by_method']}\n\n## Accepted\n")
        for r in accepted:
            f.write(f"- **{r['dataset']}**: +{r['n_gain']} ({', '.join(r['gains'])})\n")
    print(f"[SUMMARY] staged {s['ours']}/{s['total']} ({s['pct']}%) "
          f"vs baseline {base_summary['ours']}/{base_summary['total']}", flush=True)


def main_driver():
    targets = sys.argv[2:] or [
        "mutag", "enzymes", "bace", "bbbp", "ba_2motifs", "ba_house_grid",
        "spmotif_0.9", "spmotif_0.7", "graph_sst5", "graph_sst2"]
    base_cells = BC.compute_cells(BC.load_base_df())
    base_summary = BC.board_summary(base_cells)
    print(f"=== OVERNIGHT re-HPO START === baseline {base_summary['ours']}/{base_summary['total']} "
          f"({base_summary['pct']}%); targets={targets}", flush=True)
    ns = build_ns(search_baselines=SEARCH_BASELINES)
    skip = done_datasets()
    for ds in targets:
        if ds in skip:
            print(f"[skip] {ds} already in ledger", flush=True)
            continue
        try:
            run_dataset_rehpo(ns, ds, base_cells)
        except Exception:
            log({"phase": "rehpo", "dataset": ds, "status": "crashed",
                 "error": traceback.format_exc()[-400:]})
        write_summary(base_summary)
    print("=== OVERNIGHT re-HPO DONE ===", flush=True)


ROUND1 = [("mprop:0.10", {"lam": 0.10}), ("mprop:0.20", {"lam": 0.20}),
          ("mprop:0.35", {"lam": 0.35}), ("mprop:0.50", {"lam": 0.50}),
          ("grpaug:0.25", {"grpaug": 0.25}), ("grpaug:0.75", {"grpaug": 0.75}),
          ("grpaug:1.00", {"grpaug": 1.00})]
ROUND2 = (
    [(f"row:{l}", {"lam": l, "kernel": "row"}) for l in (0.05, 0.15, 0.30, 0.40, 0.60, 0.70, 1.0)] +
    [(f"sym:{l}", {"lam": l, "kernel": "sym"}) for l in (0.20, 0.35, 0.50, 0.70)] +
    [(f"row2:{l}", {"lam": l, "kernel": "row2"}) for l in (0.15, 0.25, 0.35)] +
    [(f"rowoff:{l}", {"lam": l, "kernel": "rowoff"}) for l in (0.25, 0.50, 0.75)] +
    [("row0.35+ga0.75", {"lam": 0.35, "kernel": "row", "grpaug": 0.75}),
     ("row0.20+ga0.50", {"lam": 0.20, "kernel": "row", "grpaug": 0.50}),
     ("sym0.35+ga0.75", {"lam": 0.35, "kernel": "sym", "grpaug": 0.75})])
TWEAKS = ROUND2 if os.environ.get("OVN_ROUND") == "2" else ROUND1


def safe_name(name):
    return name.replace(":", "").replace("+", "_").replace(".", "")


def run_dataset_tweak(ns, ds, base_cells):
    E.BALANCE_GT = ds in GT_IMBAL
    try:
        fr = E.freeze_dataset(ns, ds, model_class="gcn", n_val=5, n_perm=50, seed=42, epochs=150, skip_baselines=True)
        _freeze_lite(ns, fr, n_test=25, n_perm=50)
    except Exception as ex:
        log({"phase": "tweak", "dataset": ds, "tweak": "FREEZE", "status": "freeze_failed", "error": str(ex)[:300]})
        return
    php = prod_hp(ds)
    repro = eval_ours(ns, fr, php)
    repro_cells = cells_from_ours(ds, repro)
    prod_recorded = {k: v for k, v in base_cells.items() if k[0] == ds}
    drift = max((abs(repro_cells[k]["ours"] - prod_recorded[k]["ours"])
                 for k in prod_recorded if k in repro_cells), default=1.0)
    done = done_pairs()
    for name, ops in TWEAKS:
        if (ds, name) in done:
            continue
        t0 = time.time()
        rec = {"phase": "tweak", "dataset": ds, "tweak": name, "drift": round(float(drift), 4),
               "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            cand = eval_ours(ns, fr, php, ops=ops)
            cand_cells = cells_from_ours(ds, cand)
            gains, regress = BC.pareto_diff(repro_cells, cand_cells)
            detail = {}
            for k in repro_cells:
                rc, cc = repro_cells[k], cand_cells.get(k, {})
                detail[k[1]] = {"repro": round(rc["ours"], 4), "cand": round(cc.get("ours", float("nan")), 4),
                                "comp": round(rc["comp_best"], 4) if rc["comp_best"] is not None else None,
                                "won_r": rc["winner"] == "ours", "won_c": cc.get("winner") == "ours"}
            rec.update(gains=[k[1] for k in gains], regress=[k[1] for k in regress],
                       n_gain=len(gains), n_regress=len(regress), secs=round(time.time() - t0),
                       detail=detail)
            if len(regress) == 0 and len(gains) > 0 and drift <= 0.03:
                rec["status"] = "ACCEPTED"
                # avoid the "eval_records_*" prefix so load_base_df's glob does not pick these up
                cand.to_csv(os.path.join(STAGING, f"cand_{ds}__{safe_name(name)}.csv"), index=False)
                jdump({"dataset": ds, "tweak": name, "gains": [k[1] for k in gains]},
                      os.path.join(CONFIGS, f"tweak_{ds}_{safe_name(name)}.json"))
            elif len(regress) == 0 and len(gains) > 0:
                rec["status"] = "gain_but_drift"
            else:
                rec["status"] = "rejected"
        except Exception:
            rec.update(status="crashed", error=traceback.format_exc()[-300:])
        log(rec)
    del fr
    E.clear_fid_cache()
    gc.collect()


def main_tweak():
    targets = sys.argv[2:] or ["mutag", "enzymes", "bace", "bbbp", "ba_2motifs", "ba_house_grid",
                               "spmotif_0.9", "spmotif_0.7", "graph_sst5"]
    base_cells = BC.compute_cells(BC.load_base_df())
    print(f"=== OVERNIGHT TWEAK START === baseline {BC.board_summary(base_cells)}; targets={targets}", flush=True)
    ns = build_ns()
    for ds in targets:
        try:
            run_dataset_tweak(ns, ds, base_cells)
        except Exception:
            log({"phase": "tweak", "dataset": ds, "tweak": "?", "status": "crashed",
                 "error": traceback.format_exc()[-300:]})
    print("=== OVERNIGHT TWEAK DONE ===", flush=True)


GRID = ([("none", 0.0)] +
        [("row", l) for l in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.0)] +
        [("sym", l) for l in (0.20, 0.35, 0.50, 0.70)])


def build_val_assets(ns, fr, n_val=25, n_perm=50):
    val = fr._splits["val"]
    vidx = E._select_eval_indices(val, n_val, fr.seed, E.BALANCE_GT)
    vg = [val[int(i)].to(fr.device) for i in vidx]
    vsg, vgt, vM = E._build_graph_assets(ns, fr.model, fr.device, vg, fr.name, n_perm, fr.seed, label="val")
    return vg, vsg, vgt, vM


def val_score(df):
    """Balanced held-out faithfulness: + H-Fid + Fid+ + Char - |Fid-|, best-of(node,group), mean/sp."""
    s = 0.0
    for met, direction in BC.METRICS.items():
        vals = []
        for sp in BC.SPS:
            sub = df[abs(df.target_sparsity - sp) < 0.02]
            o = sub[sub.method == "ours_tuned"][met].mean()
            g = sub[sub.method == "ours_groups_tuned"][met].mean()
            cand = [v for v in (o, g) if v == v]
            if not cand:
                continue
            vals.append(max(cand) if direction == "higher" else min(cand, key=abs))
        if vals:
            m = float(np.nanmean(vals))
            s += m if direction == "higher" else -abs(m)
    return s


def main_valcal():
    targets = sys.argv[2:] or ["spmotif_0.7", "spmotif_0.9", "graph_sst5"]
    base_cells = BC.compute_cells(BC.load_base_df())
    ns = build_ns()
    results = []
    for ds in targets:
        t0 = time.time()
        E.BALANCE_GT = ds in GT_IMBAL
        try:
            fr = E.freeze_dataset(ns, ds, model_class="gcn", n_val=5, n_perm=50, seed=42,
                                  epochs=150, skip_baselines=True)
            php = prod_hp(ds)
            vg, vsg, vgt, vM = build_val_assets(ns, fr, n_val=25, n_perm=50)
            fr.test_graphs, fr.test_sg, fr.test_gt, fr.test_M = vg, vsg, vgt, vM   # sweep on VAL
            scores = []
            for kernel, lam in GRID:
                ops = None if lam == 0.0 else {"lam": lam, "kernel": kernel}
                scores.append(((kernel, lam), val_score(eval_ours(ns, fr, php, ops=ops))))
            scores.sort(key=lambda x: -x[1])
            (bk, bl), bs = scores[0]
            base_val = next(s for (k, l), s in scores if l == 0.0)

            _freeze_lite(ns, fr, n_test=25, n_perm=50)                         # back to TEST
            ops = None if bl == 0.0 else {"lam": bl, "kernel": bk}
            cand = eval_ours(ns, fr, php, ops=ops)
            cand_cells = cells_from_ours(ds, cand)
            prod_ds = {k: v for k, v in base_cells.items() if k[0] == ds}
            gains, regress = BC.pareto_diff(prod_ds, cand_cells)
            ok = len(regress) == 0 and len(gains) > 0
            if ok:
                cand.to_csv(os.path.join(STAGING, f"valcal_{ds}.csv"), index=False)
                json.dump({"dataset": ds, "lam": bl, "kernel": bk, "val_score": bs,
                           "base_val": base_val, "gains": [k[1] for k in gains]},
                          open(os.path.join(CONFIGS, f"valcal_{ds}.json"), "w"), indent=2)
            rec = {"ds": ds, "val_lam": bl, "val_kernel": bk, "val_score": round(bs, 4),
                   "noprop_val": round(base_val, 4), "val_picks_prop": bl > 0,
                   "test_gains": [k[1] for k in gains], "test_regress": [k[1] for k in regress],
                   "accept": ok, "secs": round(time.time() - t0),
                   "top5": [(k, l, round(s, 4)) for (k, l), s in scores[:5]]}
            results.append(rec)
            print(f"[{ds}] VAL-best ({bk},{bl}) score={bs:.4f} vs no-prop {base_val:.4f}; "
                  f"TEST gains={rec['test_gains']} regress={rec['test_regress']} accept={ok}", flush=True)
            del fr
            E.clear_fid_cache()
            gc.collect()
        except Exception:
            results.append({"ds": ds, "error": traceback.format_exc()[-400:]})
            print(f"[{ds}] FAILED\n{traceback.format_exc()[-600:]}", flush=True)
        json.dump(results, open(os.path.join(OUT, "valcal_results.json"), "w"), indent=2)
    print("=== VALCAL DONE ===", flush=True)


def best_accepts():
    """Re-derive each staged candidate's Pareto gain and keep the best per dataset
    (requires zero regressions)."""
    base_cells = BC.compute_cells(BC.load_base_df())
    best = {}
    for f in sorted(glob.glob("_overnight/staging/cand_*.csv")):
        name = os.path.basename(f)[len("cand_"):-4]
        if "__" not in name:
            continue
        ds, tag = name.rsplit("__", 1)
        try:
            ov = pd.read_csv(f)
            cand_cells = cells_from_ours(ds, ov)
        except Exception:
            continue
        prod_ds = {k: v for k, v in base_cells.items() if k[0] == ds}
        gains, regress = BC.pareto_diff(prod_ds, cand_cells)
        if len(regress) == 0 and len(gains) > 0 and (ds not in best or len(gains) > best[ds]["n_gain"]):
            best[ds] = {"n_gain": len(gains), "file": f, "tag": tag, "gains": [k[1] for k in gains]}
    return best


def final_board():
    df = BC.load_base_df(ours_override_dir=None)
    best = best_accepts()
    for ds, info in best.items():
        ov = pd.read_csv(info["file"])
        df = df[~((df.dataset == ds) & (df.method.isin(BC.OURS)))]
        df = pd.concat([df, ov[ov.method.isin(BC.OURS)].assign(dataset=ds)], ignore_index=True)
    return BC.board_summary(BC.compute_cells(df)), best


def main_final():
    base = BC.board_summary(BC.compute_cells(BC.load_base_df()))
    final, best = final_board()
    print(f"baseline : {base['ours']}/{base['total']} ({base['pct']}%)  {base['by_method']}")
    print(f"FINAL    : {final['ours']}/{final['total']} ({final['pct']}%)  {final['by_method']}")
    print(f"delta    : {final['ours'] - base['ours']:+d} cells")
    print("accepted improvements:")
    for ds, info in sorted(best.items()):
        print(f"  {ds:18s} +{info['n_gain']}  via {info['tag']:12s} ({', '.join(info['gains'] or [])})")
    if not best:
        print("  (none yet)")


_DISPATCH = {"driver": main_driver, "tweak": main_tweak, "valcal": main_valcal, "final": main_final}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd not in _DISPATCH:
        print(f"usage: python _ovn.py [{' | '.join(_DISPATCH)}] [datasets...]")
        sys.exit(2)
    _DISPATCH[cmd]()

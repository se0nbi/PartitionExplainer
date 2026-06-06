"""Evaluate SPMotif explanations under the DIR OOD protocol: train on the biased split
and evaluate on the balanced (b=1/3) test split.

Usage:  python _spmotif_ood.py spmotif_0.5 [spmotif_0.7 spmotif_0.9]
"""
import os
import sys
import json
import pickle
import gc

os.environ.setdefault("HPO_OUT_DIR", os.path.join("outputs", "spmotif_ood"))

import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data

import os as _os, sys as _sys, glob as _glob
_sys.path[:0] = _glob.glob(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "*", ""))
import _newds_loaders  # noqa: F401
import _hpo_engine as E
import _baseline_extra as BX

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = E.HPO_OUT
os.makedirs(OUT, exist_ok=True)
BEST_HP_DIR = os.path.join("outputs", "hpo_newds", "hyperparameters", "best")
SPB = {"spmotif_0.5": 0.5, "spmotif_0.7": 0.7, "spmotif_0.9": 0.9}
EPOCHS = int(os.environ.get("OOD_EPOCHS", "150"))
N_TEST = int(os.environ.get("OOD_NTEST", "25"))


def _read_split(name, split):
    raw = os.path.join(ROOT, "outputs", "dig_data_newds", name, name, "raw", f"{split}.pkl")
    with open(raw, "rb") as f:
        edge_index_list, label_list, gt_list, role_id_list, pos = pickle.load(f)
    g = torch.Generator().manual_seed(42)
    graphs = []
    for edge_index, y, gt, z, p in zip(edge_index_list, label_list, gt_list, role_id_list, pos):
        ei = torch.from_numpy(edge_index).long()
        n = int(torch.unique(ei).max()) + 1
        x = torch.rand((n, 4), generator=g)
        d = Data(x=x, edge_index=ei, y=torch.tensor(int(y), dtype=torch.long).view(-1))
        nl = torch.tensor(z, dtype=torch.float).view(-1)
        d.gt_node_mask = (nl != 0)
        graphs.append(d)
    return graphs


class _GList(list):
    pass


def _load_ood(name):
    tr, va, te = (_read_split(name, s) for s in ("train", "val", "test"))
    g = _GList(tr + va + te)
    g._native = {"train": tr, "val": va, "test": te}
    info = {"name": name, "task": "graph_classification", "input_dim": 4, "num_classes": 3,
            "has_ground_truth": True, "ground_truth_type": "node_mask",
            "avg_nodes": float(np.mean([x.num_nodes for x in g]))}
    print(f"  [OOD] {name}: train(biased@{SPB[name]})={len(tr)} val(bal)={len(va)} test(bal)={len(te)}",
          flush=True)
    return g, info


def install_ood(ns):
    base_loader = ns["load_dig_dataset"]
    base_split = ns["split_dataset"]

    def loader(name):
        return _load_ood(name.lower()) if name.lower() in SPB else base_loader(name)

    def split(dataset, seed=42):
        return dataset._native if hasattr(dataset, "_native") else base_split(dataset, seed=seed)

    ns["load_dig_dataset"] = loader
    ns["split_dataset"] = split


DATASETS = [d for d in sys.argv[1:] if d in SPB] or list(SPB)

ns = E.build_namespace(verbose=False)
E.inject_overrides(ns)
E.install_runtime_hooks(ns, robust=False)
E.install_extra_loaders(ns)
install_ood(ns)
E.TRAIN_CAP = 6000
E.BALANCE_GT = False
E.GRAPHSHAPIQ_MAX = 3
E.RUN_EXPENSIVE_BASELINES = False
device = ns["EVAL_CFG"].DEVICE
cfg = ns["EVAL_CFG"]
GRID = cfg.SPARSITY_GRID
e1 = ns["evaluate_one_graph"]


_remap = E.remap


def _expensive(baseline, model, data, nc):
    if baseline == "subgraphx":
        return ns["run_subgraphx"](model, data, device, nc, min_atoms=cfg.SUBX_MIN_ATOMS,
                                   rollout=10, sample_num=30)
    if baseline == "graphsvx":
        return E.run_graphsvx(model, data, device, ns, num_classes=nc,
                              num_samples=max(2 * int(data.x.size(0)), 100))
    if baseline == "mage":
        return BX.run_mage(model, data, device, nc)
    if baseline == "same":
        return BX.run_same(model, data, device, nc)
    if baseline == "graphext":
        return BX.run_graphext(model, data, device, nc)
    return None


for name in DATASETS:
    print(f"\n==== {name} (OOD: train@{SPB[name]} -> test@balanced) ====", flush=True)
    try:
        fr = E.freeze_dataset(ns, name, n_val=5, n_perm=50, seed=42, epochs=EPOCHS)
        E.freeze_test(ns, fr, n_test=N_TEST, n_perm=50)
    except Exception as ex:
        import traceback
        print(f"[{name}] freeze FAILED: {ex}", flush=True); traceback.print_exc(); continue
    model = fr.model
    nc = int(fr.info.get("num_classes", 3))
    print(f"[{name}] balanced-val acc (OOD) = {fr.meta.get('val_acc', float('nan')):.3f}", flush=True)

    frames = [fr.baseline_test]
    frames.append(_remap(E.eval_hp(ns, fr, E.DEFAULT_HP, split="test"), "default"))
    bp = os.path.join(BEST_HP_DIR, f"best_hp_{name}.json")
    if os.path.exists(bp):
        b = json.load(open(bp))
        frames.append(_remap(E.eval_hp(ns, fr, E.HP(**b["hp"]), split="test"), "tuned"))
    else:
        frames.append(_remap(E.eval_hp(ns, fr, E.DEFAULT_HP, split="test"), "tuned"))

    for baseline in ("graphsvx", "mage", "same", "graphext", "subgraphx"):
        recs, nf = [], 0
        for gi, data in enumerate(fr.test_graphs):
            gt = fr.test_gt[gi]
            try:
                s = _expensive(baseline, model, data, nc)
            except Exception:
                s = None
            if s is None:
                nf += 1
                continue
            recs += e1(model, data, np.asarray(s, float), device, GRID, baseline, gi, gt)
        if recs:
            frames.append(pd.DataFrame(recs))
        print(f"[{name}] {baseline}: {len(fr.test_graphs)-nf}/{len(fr.test_graphs)} scored", flush=True)

    out = pd.concat(frames, ignore_index=True)
    out["dataset"] = name
    out.to_csv(os.path.join(E.EVAL_MAIN_DIR, f"eval_records_{name}.csv"), index=False)
    print(f"[{name}] wrote {OUT}/eval_records_{name}.csv; methods={sorted(out['method'].unique())}",
          flush=True)
    del fr
    E.clear_fid_cache()
    gc.collect()

print("\n=== SPMOTIF OOD DONE ===", flush=True)

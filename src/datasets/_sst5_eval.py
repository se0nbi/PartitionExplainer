"""Evaluation script for Graph-SST5: trains a model and computes our method plus baselines."""
import os
import time
import numpy as np
import pandas as pd
import torch
import os as _os, sys as _sys, glob as _glob
_sys.path[:0] = _glob.glob(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "*", ""))
import _hpo_engine as E
import _baseline_extra as BX

E.patch_torch_load()

NAME = "graph_sst5"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIG_ROOT = os.path.join(ROOT, "outputs", "dig_data")  # contains Graph-SST5/raw/...

E.ROBUST_SAMPLES = 30
ns = E.build_namespace(verbose=False)
E.inject_overrides(ns)
E.install_runtime_hooks(ns, robust=True)
E.install_extra_loaders(ns)
ns["EVAL_CFG"].SUBX_MAX_GRAPHS = 2
E.TRAIN_CAP = 6000
E.GRAPHSHAPIQ_MAX = 3
E.BALANCE_GT = False

_real = ns["load_dig_dataset"]
def _loader(name):
    if str(name).lower() in ("graph_sst5", "graphsst5"):
        from dig.xgraph.dataset import SentiGraphDataset
        ds = SentiGraphDataset(DIG_ROOT, name="Graph-SST5")
        graphs = [ds[i] for i in range(len(ds))]
        info = {"name": name, "task": "graph_classification",
                "input_dim": int(ds.num_node_features),
                "num_classes": int(ds.num_classes)}
        return graphs, info
    return _real(name)
ns["load_dig_dataset"] = _loader

print("=== Graph-SST5 full eval START ===", flush=True)
t0 = time.time()
fr = E.freeze_dataset(ns, NAME, n_val=5, n_perm=50, seed=42, epochs=150)
E.freeze_test(ns, fr, n_test=25, n_perm=50)
print(f"frozen in {time.time()-t0:.0f}s: val_acc={fr.meta.get('val_acc'):.3f} "
      f"nc={fr.info.get('num_classes')} input_dim={fr.info.get('input_dim')} "
      f"baselines={sorted(fr.baseline_test['method'].unique())}", flush=True)


_remap = E.remap

import json as _json
_bp = os.path.join(E.HP_BEST_DIR, f"best_hp_{NAME}.json")
if os.path.exists(_bp):
    _tuned_hp = E.HP(**_json.load(open(_bp))["hp"])
    print(f"ours TUNED config loaded from best_hp_{NAME}.json", flush=True)
else:
    _tuned_hp = E.DEFAULT_HP
    print(f"WARNING: best_hp_{NAME}.json missing -> ours falls back to DEFAULT "
          f"(run _sst5_hpo.py first)", flush=True)
dft = _remap(E.eval_hp(ns, fr, E.DEFAULT_HP, split="test"), "default")
tun = _remap(E.eval_hp(ns, fr, _tuned_hp, split="test"), "tuned")
recs = pd.concat([fr.baseline_test, dft, tun], ignore_index=True)
recs["dataset"] = NAME
recs.to_csv(os.path.join(E.EVAL_MAIN_DIR, f"eval_records_{NAME}.csv"), index=False)
print(f"wrote eval_records_{NAME}.csv methods={sorted(recs['method'].unique())}", flush=True)

# extra baselines reuse the frozen model + the SAME deterministic 25 test graphs
device = ns["EVAL_CFG"].DEVICE
cfg = ns["EVAL_CFG"]
GRID = cfg.SPARSITY_GRID
e1 = ns["evaluate_one_graph"]
model = fr.model
nc = int(fr.info.get("num_classes", 2))
test = fr._splits["test"]
tidx = E._select_eval_indices(test, 25, 42, False)


def score(bl, data):
    if bl == "subgraphx":
        E.RUN_EXPENSIVE_BASELINES = True
        return ns["run_subgraphx"](model, data, device, nc, min_atoms=cfg.SUBX_MIN_ATOMS,
                                   rollout=10, sample_num=30)
    if bl == "graphsvx":
        return E.run_graphsvx(model, data, device, ns, num_classes=nc,
                              num_samples=max(2 * int(data.x.size(0)), 100))
    if bl == "mage":
        return BX.run_mage(model, data, device, nc)
    if bl == "same":
        return BX.run_same(model, data, device, nc)
    if bl == "graphext":
        return BX.run_graphext(model, data, device, nc)


for bl in ["subgraphx", "graphsvx", "mage", "same", "graphext"]:
    t = time.time()
    brecs, nfail = [], 0
    for gi, i in enumerate(tidx):
        data = test[int(i)].to(device)
        gt = ns["get_ground_truth_nodes"](data, dataset_name=NAME)
        try:
            s = score(bl, data)
        except Exception:
            s = None
        if s is None:
            nfail += 1
            continue
        brecs += e1(model, data, np.asarray(s, float), device, GRID, bl, gi, gt)
    _bdir = os.path.join(E.EVAL_BASELINES_DIR, bl)
    os.makedirs(_bdir, exist_ok=True)
    pd.DataFrame(brecs).assign(dataset=NAME).to_csv(
        os.path.join(_bdir, f"eval_records_{bl}_{NAME}.csv"), index=False)
    print(f"[{bl}] {len(tidx)-nfail}/{len(tidx)} scored, {len(brecs)} rows "
          f"({time.time()-t:.0f}s)", flush=True)

print(f"=== Graph-SST5 DONE in {(time.time()-t0)/60:.1f} min ===", flush=True)

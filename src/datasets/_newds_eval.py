"""Run the full leaderboard evaluation for one or more datasets on a single model freeze, writing per-dataset CSV records for every explanation method."""
import os
import sys
import json
import gc

os.environ.setdefault("HPO_OUT_DIR", os.path.join("outputs", "hpo_newds"))

import numpy as np
import pandas as pd

import os as _os, sys as _sys, glob as _glob
_sys.path[:0] = _glob.glob(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "*", ""))
import _newds_loaders  # noqa: F401
import _hpo_engine as E
import _baseline_extra as BX

OUT = E.HPO_OUT
N_TEST = int(os.environ.get("NEWDS_NTEST", "25"))
EPOCHS = int(os.environ.get("NEWDS_EPOCHS", "150"))
DATASETS = sys.argv[1:]
if not DATASETS:
    print("usage: python _newds_eval.py <dataset> [dataset...]", flush=True)
    sys.exit(1)

ns = E.build_namespace(verbose=False)
E.inject_overrides(ns)
E.install_runtime_hooks(ns, robust=False)
E.install_extra_loaders(ns)
E.TRAIN_CAP = 6000
E.BALANCE_GT = False
ns["EVAL_CFG"].SUBX_MAX_GRAPHS = 2
device = ns["EVAL_CFG"].DEVICE
cfg = ns["EVAL_CFG"]
GRID = cfg.SPARSITY_GRID
e1 = ns["evaluate_one_graph"]


_remap = E.remap


def _expensive(name, baseline, model, data, nc):
    if baseline == "subgraphx":
        return ns["run_subgraphx"](model, data, device, nc,
                                   min_atoms=cfg.SUBX_MIN_ATOMS, rollout=10, sample_num=30)
    if baseline == "graphsvx":
        return E.run_graphsvx(model, data, device, ns, num_classes=nc,
                              num_samples=max(2 * int(data.x.size(0)), 100))
    if baseline == "mage":
        return BX.run_mage(model, data, device, nc)
    if baseline == "same":
        return BX.run_same(model, data, device, nc)
    return None


for name in DATASETS:
    bp = os.path.join(E.HP_BEST_DIR, f"best_hp_{name}.json")
    E.GRAPHSHAPIQ_MAX = 3
    E.RUN_EXPENSIVE_BASELINES = False  # we run subgraphx/mage ourselves below
    try:
        fr = E.freeze_dataset(ns, name, n_val=5, n_perm=50, seed=42, epochs=EPOCHS)
        E.freeze_test(ns, fr, n_test=N_TEST, n_perm=50)
    except Exception as ex:
        import traceback
        print(f"[{name}] freeze FAILED: {ex}", flush=True)
        traceback.print_exc()
        continue
    model = fr.model
    nc = int(fr.info.get("num_classes", 2))

    frames = [fr.baseline_test]
    # ours: default + tuned (if a tuned best_hp exists)
    frames.append(_remap(E.eval_hp(ns, fr, E.DEFAULT_HP, split="test"), "default"))
    if os.path.exists(bp):
        b = json.load(open(bp))
        frames.append(_remap(E.eval_hp(ns, fr, E.HP(**b["hp"]), split="test"), "tuned"))
    else:
        print(f"[{name}] WARN: no best_hp -> tuned == default", flush=True)
        frames.append(_remap(E.eval_hp(ns, fr, E.DEFAULT_HP, split="test"), "tuned"))

    # expensive baselines on the SAME frozen test graphs
    for baseline in ("graphsvx", "mage", "same", "subgraphx"):
        recs, nfail = [], 0
        for gi, data in enumerate(fr.test_graphs):
            gt = fr.test_gt[gi]
            try:
                s = _expensive(name, baseline, model, data, nc)
            except Exception:
                s = None
            if s is None:
                nfail += 1
                continue
            recs += e1(model, data, np.asarray(s, float), device, GRID, baseline, gi, gt)
        if recs:
            frames.append(pd.DataFrame(recs))
        print(f"[{name}] {baseline}: {len(fr.test_graphs) - nfail}/{len(fr.test_graphs)} scored",
              flush=True)

    out = pd.concat(frames, ignore_index=True)
    out["dataset"] = name
    out.to_csv(os.path.join(E.EVAL_MAIN_DIR, f"eval_records_{name}.csv"), index=False)
    print(f"[{name}] wrote eval_records_{name}.csv; methods={sorted(out['method'].unique())}",
          flush=True)
    del fr
    E.clear_fid_cache()
    gc.collect()

print("=== NEWDS EVAL DONE ===", flush=True)

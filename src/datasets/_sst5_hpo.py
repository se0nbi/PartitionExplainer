"""Per-dataset hyperparameter optimization for the graph_sst5 dataset."""
import os
import json
import time
from dataclasses import asdict
import torch
import os as _os, sys as _sys, glob as _glob
_sys.path[:0] = _glob.glob(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "*", ""))
import _hpo_engine as E

E.patch_torch_load()

NAME = "graph_sst5"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIG_ROOT = os.path.join(ROOT, "outputs", "dig_data")
LHS, BO, NVAL, NTEST, EPOCHS, CONFIRM = 400, 250, 40, 60, 150, 12

ns = E.build_namespace(verbose=True)
E.inject_overrides(ns)
E.install_runtime_hooks(ns, robust=False)
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


jdump = E.jdump


print(f"=== SST5 HPO START (lhs={LHS} bo={BO} nval={NVAL} ntest={NTEST}) ===", flush=True)
t0 = time.time()
fr = E.freeze_dataset(ns, NAME, model_class="gcn", n_val=NVAL, n_perm=50, seed=42, epochs=EPOCHS)
print(f"frozen in {time.time()-t0:.0f}s  val_acc={fr.meta['val_acc']:.3f}", flush=True)

res = E.search_dataset(ns, fr, lhs_n=LHS, bo_iter=BO, sens_levels=5, seed=0)
print(f"search done  best_obj={res['best_obj']:.4f}  default_obj={res['default_obj']:.4f}", flush=True)
jdump(res, os.path.join(E.HP_STATE_DIR, f"state_{NAME}.json"))

pairs = sorted(zip(res["samples_obj"], res["samples_hp"]), key=lambda t: -t[0])
seen, finalists = set(), []
for obj, hp_d in pairs:
    key = tuple(round(hp_d[k], 2) for k in E.ALL_CONT_DIMS)
    if key in seen:
        continue
    seen.add(key); finalists.append(E.HP(**hp_d))
    if len(finalists) >= CONFIRM:
        break

rows, best, dinfo = E.confirm_finalists(ns, fr, finalists, n_test=NTEST, n_perm_hi=100, robust=False)
best_hp = E.HP(**best["hp"])
out = {
    "schema_version": 1, "dataset": NAME, "model_class": "gcn",
    "frozen_context": {"split_seed": 42, "n_val_graphs": NVAL, "n_test_graphs": NTEST,
                       "n_perm": 50, "n_perm_confirm": 100, "sa_seed": 42, "epochs": EPOCHS,
                       "obj_sparsities": E.OBJ_SPARSITIES, "baseline_methods": E.BASELINE_METHODS},
    "hp": asdict(best_hp), "is_default": best.get("is_default", False),
    "accepted": best.get("accept", False), "confirm_rows": rows,
    "default_val": dinfo["default_val"], "default_test": dinfo["default_test"], "meta": fr.meta,
}
jdump(out, os.path.join(E.HP_BEST_DIR, f"best_hp_{NAME}.json"))
print(f"=== SST5 HPO DONE in {(time.time()-t0)/60:.1f} min  "
      f"accepted={best.get('accept', 'DEFAULT')}  "
      f"val_wins {dinfo['default_val'].get('wins') if isinstance(dinfo.get('default_val'),dict) else '?'} ===", flush=True)

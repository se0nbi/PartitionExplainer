"""Orchestrator for the per-dataset hyperparameter search and surrogate model.

For each dataset it freezes the model/baselines, searches hyperparameters,
confirms finalists on validation and test, and saves the best configuration.
It then fits a cross-dataset surrogate and writes the report.

Usage:
  python _hpo_run.py --datasets ba_2motifs bbbp bace proteins enzymes \
      --lhs 300 --bo 150 --nval 25 --ntest 40 --epochs 150 --confirm-top 6

Outputs are written to outputs/hpo/.
"""
import os
import sys
import json
import time
import pickle
import argparse
from dataclasses import asdict

import os as _os, sys as _sys, glob as _glob
_sys.path[:0] = _glob.glob(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "*", ""))
import _hpo_engine as E


jdump = E.jdump
jload = E.jload


def pick_finalists(res, top=6):
    """Top-`top` unique configs from the search samples, by objective."""
    pairs = sorted(zip(res["samples_obj"], res["samples_hp"]), key=lambda t: -t[0])
    seen, out = set(), []
    for obj, hp_d in pairs:
        key = tuple(round(hp_d[k], 2) for k in E.ALL_CONT_DIMS)
        if key in seen:
            continue
        seen.add(key); out.append(E.HP(**hp_d))
        if len(out) >= top:
            break
    return out


def run_one_dataset(ns, name, args):
    best_path = os.path.join(E.HP_BEST_DIR, f"best_hp_{name}.json")
    if os.path.exists(best_path) and not args.force:
        print(f"[{name}] best_hp exists -> skipping (resume).", flush=True)
        return

    t0 = time.time()
    fr = E.freeze_dataset(ns, name, model_class="gcn", n_val=args.nval,
                          n_perm=args.nperm, seed=42, epochs=args.epochs)
    print(f"[{name}] frozen in {time.time()-t0:.0f}s "
          f"(val_acc={fr.meta['val_acc']:.3f}, "
          f"baseline methods={sorted(fr.baseline_val['method'].unique())})", flush=True)

    t0 = time.time()
    res = E.search_dataset(ns, fr, lhs_n=args.lhs, bo_iter=args.bo,
                           sens_levels=args.sens_levels, seed=0)
    print(f"[{name}] search in {time.time()-t0:.0f}s  "
          f"best_obj={res['best_obj']:.4f} default_obj={res['default_obj']:.4f}", flush=True)
    jdump(res, os.path.join(E.HP_STATE_DIR, f"state_{name}.json"))

    # confirm finalists on full val + test
    finalists = pick_finalists(res, top=args.confirm_top)
    t0 = time.time()
    rows, best, dinfo = E.confirm_finalists(ns, fr, finalists, n_test=args.ntest,
                                            n_perm_hi=args.nperm_hi, robust=False)
    print(f"[{name}] confirm in {time.time()-t0:.0f}s  accepted={best.get('accept', 'DEFAULT')}",
          flush=True)

    best_hp = E.HP(**best["hp"])
    out = {
        "schema_version": 1, "dataset": name, "model_class": "gcn",
        "frozen_context": {"split_seed": 42, "n_val_graphs": args.nval,
                           "n_test_graphs": args.ntest, "n_perm": args.nperm,
                           "n_perm_confirm": args.nperm_hi,
                           "sa_seed": 42, "epochs": args.epochs,
                           "sparsity_grid": list(ns["EVAL_CFG"].SPARSITY_GRID),
                           "obj_sparsities": E.OBJ_SPARSITIES,
                           "baseline_methods": E.BASELINE_METHODS},
        "hp": asdict(best_hp),
        "is_default": best.get("is_default", False),
        "accepted": best.get("accept", False),
        "confirm_rows": rows,
        "default_val": dinfo["default_val"], "default_test": dinfo["default_test"],
        "meta": fr.meta,
    }
    jdump(out, best_path)
    print(f"[{name}] wrote {best_path}", flush=True)
    # free memory before next dataset
    del fr
    E.clear_fid_cache()
    import gc; gc.collect()


def run_confirm_only(ns, name, args):
    """Re-evaluate the saved best_hp on val+test with Robust alpha-Fidelity enabled
    for both ours and the baselines, and rewrite its confirm_rows. No search."""
    bp = os.path.join(E.HP_BEST_DIR, f"best_hp_{name}.json")
    if not os.path.exists(bp):
        print(f"[{name}] no best_hp -> skip confirm-only", flush=True)
        return
    b = jload(bp)
    best_hp = E.HP(**b["hp"])
    mc = b.get("model_class", "gcn")
    t0 = time.time()
    fr = E.freeze_dataset(ns, name, model_class=mc, n_val=args.nval,
                          n_perm=args.nperm, seed=42, epochs=args.epochs)
    finalists = [] if b.get("is_default") else [best_hp]
    rows, best, dinfo = E.confirm_finalists(ns, fr, finalists, n_test=args.ntest,
                                            n_perm_hi=args.nperm_hi, robust=True)
    b["confirm_rows"] = rows
    b["default_val"] = dinfo["default_val"]
    b["default_test"] = dinfo["default_test"]
    b["hp"] = best["hp"]
    b["is_default"] = best.get("is_default", False)
    b["accepted"] = best.get("accept", False)
    b["robust_confirmed"] = True
    jdump(b, bp)
    print(f"[{name}] confirm-only (robust) in {time.time()-t0:.0f}s  "
          f"accepted={best.get('accept', 'DEFAULT')}", flush=True)
    del fr
    E.clear_fid_cache()
    import gc; gc.collect()


def build_surrogate_and_report(args):
    """Pool all state files, fit the surrogate, run predict_hp, LODO and the
    size law, and write report.csv."""
    results = []
    for name in args.datasets:
        sp = os.path.join(E.HP_STATE_DIR, f"state_{name}.json")
        if os.path.exists(sp):
            results.append(jload(sp))
    if len(results) < 1:
        print("No state files; skipping surrogate.", flush=True)
        return
    print(f"\n=== surrogate over {len(results)} datasets "
          f"({sum(len(r['samples_obj']) for r in results)} samples) ===", flush=True)

    X, yraw, yz, ds = E.build_surrogate_dataset(results)
    surr = E.fit_meta_surrogate(X, yz, seed=0)
    with open(os.path.join(E.HP_CALIB_DIR,"surrogate.pkl"), "wb") as f:
        pickle.dump({"model": surr, "meta_keys": E.META_KEYS,
                     "cont_dims": E.ALL_CONT_DIMS, "search_bounds": E.SEARCH_BOUNDS}, f)
    # feature importances (which dataset properties + HPs drive quality)
    feat_names = E.META_KEYS + E.ALL_CONT_DIMS
    imp = sorted(zip(feat_names, surr.feature_importances_.tolist()),
                 key=lambda t: -t[1])
    jdump({"feature_importance": imp}, os.path.join(E.HP_CALIB_DIR,"surrogate_importance.json"))
    print("  top surrogate features: " + ", ".join(f"{k}={v:.3f}" for k, v in imp[:6]))

    # predict_hp per dataset (from full surrogate) + LODO generalization
    for r in results:
        ph = E.predict_hp(surr, r["meta"], seed=0)
        jdump(asdict(ph), os.path.join(E.HP_CALIB_DIR,f"predicted_hp_{r['dataset']}.json"))
    lodo = E.lodo_validation(results, seed=0)
    jdump(lodo, os.path.join(E.HP_CALIB_DIR,"lodo.json"))
    for d, v in lodo.items():
        print(f"  LODO {d}: heldout_R2={v['heldout_r2']:.3f}  "
              f"theta_dist={v['theta_dist_to_searched']:.3f}")

    jdump(E.size_law_report(results), os.path.join(E.HP_CALIB_DIR,"size_law.json"))

    # report.csv: per dataset default-vs-tuned (from best_hp confirm rows)
    import pandas as pd
    rep = []
    for name in args.datasets:
        bp = os.path.join(E.HP_BEST_DIR, f"best_hp_{name}.json")
        if not os.path.exists(bp):
            continue
        b = jload(bp)
        drow = next((x for x in b["confirm_rows"] if x["is_default"]), None)
        trow = next((x for x in b["confirm_rows"] if not x["is_default"] and x["accept"]), None)
        if trow is None:  # nothing accepted -> tuned == default
            trow = drow
        rep.append({
            "dataset": name, "accepted": b["accepted"],
            "val_wins_default": drow["val_wins"], "val_wins_tuned": trow["val_wins"],
            "val_raw_default": round(drow["val_raw"], 4), "val_raw_tuned": round(trow["val_raw"], 4),
            "test_wins_default": drow["test_wins"], "test_wins_tuned": trow["test_wins"],
            "test_raw_default": round(drow["test_raw"], 4), "test_raw_tuned": round(trow["test_raw"], 4),
            "mu_c": b["hp"]["mu_c"], "mu_beta": b["hp"]["mu_beta"],
            "node_aug": b["hp"]["node_aug_coef"], "group_aug": b["hp"]["group_aug_coef"],
            "gamma_H": b["hp"]["gamma_H"], "gamma_S": b["hp"]["gamma_S"],
        })
    df = pd.DataFrame(rep)
    df.to_csv(os.path.join(E.BOARD_DIR, "report.csv"), index=False)
    print("\n=== REPORT (default -> tuned) ===")
    if not df.empty:
        tv = df["test_wins_tuned"].sum(); td = df["test_wins_default"].sum()
        vv = df["val_wins_tuned"].sum(); vd = df["val_wins_default"].sum()
        print(df.to_string(index=False))
        print(f"\nTOTAL val wins:  {vd} -> {vv}  ({vv-vd:+d})")
        print(f"TOTAL test wins: {td} -> {tv}  ({tv-td:+d})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["ba_2motifs", "bbbp", "bace", "proteins", "enzymes"])
    ap.add_argument("--lhs", type=int, default=300)
    ap.add_argument("--bo", type=int, default=150)
    ap.add_argument("--nval", type=int, default=25)
    ap.add_argument("--ntest", type=int, default=40)
    ap.add_argument("--nperm", type=int, default=50)
    ap.add_argument("--nperm-hi", type=int, default=100, dest="nperm_hi")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--sens-levels", type=int, default=5, dest="sens_levels")
    ap.add_argument("--confirm-top", type=int, default=6, dest="confirm_top")
    ap.add_argument("--train-cap", type=int, default=0, dest="train_cap",
                    help="subsample train split to N graphs (0 = no cap; use for Graph-SST2)")
    ap.add_argument("--gsiq-max", type=int, default=5, dest="gsiq_max",
                    help="cap GraphSHAP-IQ graph count (memory) ")
    ap.add_argument("--balance-gt", action="store_true", dest="balance_gt",
                    help="select val/test balanced by label (for low-positive-rate GT datasets)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-search", action="store_true",
                    help="only (re)build the surrogate + report from existing state files")
    ap.add_argument("--confirm-only", action="store_true", dest="confirm_only",
                    help="re-confirm saved best_hp with Robust alpha-Fidelity on (no search)")
    args = ap.parse_args()

    os.makedirs(E.HP_DIR, exist_ok=True)
    jdump(asdict(E.DEFAULT_HP), os.path.join(E.HP_CALIB_DIR,"initial_hp.json"))

    print(f"=== HPO RUN START  datasets={args.datasets} ===", flush=True)
    print(f"    lhs={args.lhs} bo={args.bo} nval={args.nval} ntest={args.ntest} "
          f"nperm={args.nperm}/{args.nperm_hi} epochs={args.epochs}", flush=True)

    t_all = time.time()
    if args.confirm_only or not args.skip_search:
        ns = E.build_namespace(verbose=True)
        E.inject_overrides(ns)
        E.install_runtime_hooks(ns, robust=bool(args.confirm_only))
        E.install_extra_loaders(ns)  # adds MUTAG/ENZYMES/BZR/COX2/GraphXAI
        E.TRAIN_CAP = args.train_cap or None
        E.GRAPHSHAPIQ_MAX = args.gsiq_max
        E.BALANCE_GT = args.balance_gt
        ns["EVAL_CFG"].SUBX_MAX_GRAPHS = 2
        for name in args.datasets:
            try:
                if args.confirm_only:
                    run_confirm_only(ns, name, args)
                else:
                    run_one_dataset(ns, name, args)
            except Exception as e:
                import traceback
                print(f"[{name}] FAILED: {e}", flush=True); traceback.print_exc()
            E.clear_fid_cache()

    build_surrogate_and_report(args)
    print(f"\n=== HPO RUN DONE in {(time.time()-t_all)/60:.1f} min ===", flush=True)


if __name__ == "__main__":
    main()

"""Per-dataset hyperparameter optimization driver for the extra datasets."""
import os
import argparse

os.environ.setdefault("HPO_OUT_DIR", os.path.join("outputs", "hpo_newds"))

import os as _os, sys as _sys, glob as _glob
_sys.path[:0] = _glob.glob(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "*", ""))
import _newds_loaders  # noqa: F401
import _hpo_engine as E
import _hpo_run as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--lhs", type=int, default=250)
    ap.add_argument("--bo", type=int, default=150)
    ap.add_argument("--nval", type=int, default=25)
    ap.add_argument("--ntest", type=int, default=40)
    ap.add_argument("--nperm", type=int, default=50)
    ap.add_argument("--nperm-hi", type=int, default=100, dest="nperm_hi")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--sens-levels", type=int, default=5, dest="sens_levels")
    ap.add_argument("--confirm-top", type=int, default=6, dest="confirm_top")
    ap.add_argument("--train-cap", type=int, default=6000, dest="train_cap")
    ap.add_argument("--gsiq-max", type=int, default=3, dest="gsiq_max")
    ap.add_argument("--balance-gt", action="store_true", dest="balance_gt")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(E.HPO_OUT, exist_ok=True)
    print(f"=== NEWDS HPO START datasets={args.datasets} out={E.HPO_OUT} ===", flush=True)
    ns = E.build_namespace(verbose=True)
    E.inject_overrides(ns)
    E.install_runtime_hooks(ns, robust=False)
    E.install_extra_loaders(ns)
    E.TRAIN_CAP = args.train_cap or None
    E.GRAPHSHAPIQ_MAX = args.gsiq_max
    E.BALANCE_GT = args.balance_gt
    ns["EVAL_CFG"].SUBX_MAX_GRAPHS = 2

    for name in args.datasets:
        try:
            R.run_one_dataset(ns, name, args)
        except Exception as e:
            import traceback
            print(f"[{name}] FAILED: {e}", flush=True)
            traceback.print_exc()
        E.clear_fid_cache()
    print("=== NEWDS HPO DONE ===", flush=True)


if __name__ == "__main__":
    main()

"""Runs up to NEWDS_MAXPAR dataset evaluation lanes concurrently, each performing per-dataset HPO followed by full evaluation."""
import os
import sys
import subprocess
import concurrent.futures

ALL = ["spmotif_0.5", "spmotif_0.7", "spmotif_0.9",
       "ba_house_grid", "ba_house_or_grid", "graph_twitter"]
DATASETS = sys.argv[1:] or ALL
OUT = os.path.join("outputs", "hpo_newds")
os.makedirs(OUT, exist_ok=True)
_HERE = os.path.dirname(os.path.abspath(__file__))
MAXPAR = int(os.environ.get("NEWDS_MAXPAR", "3"))
FORCE = os.environ.get("NEWDS_FORCE", "") not in ("", "0")

HPO_ARGS = ["--lhs", os.environ.get("NEWDS_LHS", "250"),
            "--bo", os.environ.get("NEWDS_BO", "150"),
            "--nval", "25", "--ntest", "40", "--epochs", "150",
            "--confirm-top", "6", "--gsiq-max", "3"]


def _run(cmd, env, log_path):
    with open(log_path, "w", encoding="utf-8") as fh:
        return subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)


def lane(ds):
    base = dict(os.environ)
    base["HPO_OUT_DIR"] = OUT
    base["PYTHONIOENCODING"] = "utf-8"
    rc_hpo = rc_eval = 0

    if FORCE or not os.path.exists(os.path.join(OUT, f"best_hp_{ds}.json")):
        env = dict(base); env["HPO_SNAP_DIR"] = os.path.join("_hpo_snapshots", f"nd_{ds}")
        cmd = [sys.executable, os.path.join(_HERE, "_newds_hpo_run.py"), "--datasets", ds] + HPO_ARGS
        if FORCE:
            cmd.append("--force")
        rc_hpo = _run(cmd, env, os.path.join(OUT, f"log_hpo_{ds}.log"))
        print(f"[{ds}] HPO rc={rc_hpo}", flush=True)
    else:
        print(f"[{ds}] HPO skipped (best_hp exists)", flush=True)

    if FORCE or not os.path.exists(os.path.join(OUT, f"eval_records_{ds}.csv")):
        env = dict(base); env["HPO_SNAP_DIR"] = os.path.join("_hpo_snapshots", f"nde_{ds}")
        rc_eval = _run([sys.executable, os.path.join(_HERE, "_newds_eval.py"), ds], env,
                       os.path.join(OUT, f"log_eval_{ds}.log"))
        print(f"[{ds}] EVAL rc={rc_eval}", flush=True)
    else:
        print(f"[{ds}] EVAL skipped (eval_records exists)", flush=True)
    return ds, rc_hpo, rc_eval


print(f"=== NEWDS RUN ALL: {len(DATASETS)} datasets, max_par={MAXPAR} ===", flush=True)
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=MAXPAR) as ex:
    futs = {ex.submit(lane, ds): ds for ds in DATASETS}
    for fut in concurrent.futures.as_completed(futs):
        try:
            results.append(fut.result())
        except Exception as e:
            print(f"[{futs[fut]}] LANE EXC {type(e).__name__}: {e}", flush=True)

print("\n=== SUMMARY ===", flush=True)
for ds, a, b in sorted(results):
    print(f"  {ds:20s} hpo_rc={a} eval_rc={b}", flush=True)
print("=== ALL NEWDS LANES DONE ===", flush=True)

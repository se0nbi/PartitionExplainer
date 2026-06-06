"""Build the hpo_tuning.ipynb notebook that visualizes the learned HPO results."""
import json, os

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text})


md("""# Learned Per-Dataset Hyperparameter Calibration

This notebook visualizes the learned per-dataset hyperparameter calibration. The search itself lives in
`_hpo_engine.py` + `_hpo_run.py`; this notebook reads `outputs/hpo/*` and renders the results.

**What the system does.** Instead of hand-tuning, it *trains a surrogate model*
`R(dataset_meta_features ⊕ hyperparameters) → explanation quality` from every config it
evaluates, uses that model to drive a Bayesian search per dataset, and exposes
`predict_hp(database)` that derives hyperparameters from a database's own properties
(generalization checked leave-one-dataset-out). The fixed `μ = √n` group-size rule is
replaced by a learned law `μ = c·n^β` whose coefficients are tuned per database.

**Objective:** balanced margin vs the best baseline across the metric×sparsity cells, with
cells we currently *lose* up-weighted 2:1 and a HARD Pareto guard that forbids regressing any
cell we already win. Tuning on the `val` split; reporting on the held-out `test` split.""")

md("""## 1. Setup""")
code("""import os, sys, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob as _g
for _p in _g.glob("src/*/"): sys.path.insert(0, _p)
import _hpo_engine as E

HP = E.HP_DIR
BOARD = E.BOARD_DIR
print("Reading artifacts from:", HP, BOARD)

def jload(name):
    hits = _g.glob(os.path.join(HP, "**", name), recursive=True)
    return json.load(open(hits[0], encoding="utf-8")) if hits else None

DATASETS = ["ba_2motifs", "bbbp", "bace", "proteins", "mutag", "enzymes",
            "benzene", "graph_sst2", "graph_sst5"]
states = {d: jload(f"state_{d}.json") for d in DATASETS}
states = {d: s for d, s in states.items() if s}
print("states available:", list(states))""")

md("""## 2. (Re)run the search

The search runs as a background process (resumable; a dataset whose `best_hp_<ds>.json`
exists is skipped). To launch or resume from a terminal:

```
python src/core/_hpo_run.py --datasets ba_2motifs bbbp bace proteins enzymes \\
    --lhs 300 --bo 150 --nval 25 --ntest 40 --epochs 150 --confirm-top 6
```

`initial_hp.json` holds the default configuration. To fall back to it, ignore
`best_hp_*.json` and use `initial_hp.json`.""")
code("""print("initial (revert) HP:", jload("initial_hp.json"))""")

md("""## 3. Round A — hyperparameter sensitivity (which HPs to attack)

One-at-a-time screen: how much each hyperparameter moves the objective. This is the
reviewer-grade sensitivity analysis justifying *which* hyperparameters matter per dataset.""")
code("""fig, axes = plt.subplots(1, len(states), figsize=(5*len(states), 4), squeeze=False)
for ax, (d, s) in zip(axes[0], states.items()):
    sens = dict(s["sensitivity"])
    ks = sorted(sens, key=sens.get)
    ax.barh(ks, [sens[k] for k in ks], color="#4C72B0")
    ax.set_title(f"{d}\\nobjective sensitivity"); ax.set_xlabel("max-min objective")
plt.tight_layout(); plt.show()""")

md("""## 4. Rounds B→C — search convergence (best objective so far)

Latin-hypercube exploration (B) then RF-surrogate Bayesian optimization (C). The dashed line
is the Round-0 default objective; the curve is the running best.""")
code("""fig, axes = plt.subplots(1, len(states), figsize=(5*len(states), 4), squeeze=False)
for ax, (d, s) in zip(axes[0], states.items()):
    y = np.array(s["samples_obj"], float)
    best = np.maximum.accumulate(y)
    ax.plot(best, lw=2, label="best so far")
    ax.axhline(s["default_obj"], ls="--", c="grey", label="default")
    ax.set_title(d); ax.set_xlabel("config #"); ax.set_ylabel("objective"); ax.legend()
plt.tight_layout(); plt.show()""")

md("""## 5. The trained surrogate — what drives explanation quality

Feature importances of `R(meta ⊕ θ → quality)`: which dataset properties and which
hyperparameters most determine performance. `predict_hp(meta)` derives an HP vector from a
database's properties alone.""")
code("""imp = jload("surrogate_importance.json")
if imp:
    fi = imp["feature_importance"][:14][::-1]
    plt.figure(figsize=(7,5))
    plt.barh([k for k,_ in fi], [v for _,v in fi], color="#55A868")
    plt.title("Surrogate feature importance"); plt.xlabel("importance"); plt.tight_layout(); plt.show()

pk = _g.glob(os.path.join(HP, "**", "surrogate.pkl"), recursive=True)
if pk:
    surr = pickle.load(open(pk[0], "rb"))["model"]
    for d, s in states.items():
        ph = E.predict_hp(surr, s["meta"], seed=0)
        print(f"predict_hp({d}) = "
              f"gamma_H={ph.gamma_H:.2f} gamma_S={ph.gamma_S:.2f} "
              f"mu={ph.mu_c:.2f}*n^{ph.mu_beta:.2f} "
              f"node_aug={ph.node_aug_coef:.2f} group_aug={ph.group_aug_coef:.2f}")""")

md("""## 6. The learned group-size law  μ = c · n^β   (reviewer's "ideal group size")

Replaces the arbitrary `μ = √n`. Shows the tuned law per dataset vs the √n baseline.""")
code("""sl = jload("size_law.json")
if sl:
    rows = []
    for d, v in sl.items():
        rows.append({"dataset": d, "mu_c": round(v["mu_c"],3), "mu_beta": round(v["mu_beta"],3),
                     "avg_n": round(v["avg_n"],1),
                     "mu@avg_n (learned)": round(v["mu_ideal_at_avg_n"],2),
                     "sqrt(n)": round(v["sqrt_n"],2),
                     "k0 (learned)": v["k0_at_avg_n"], "k0 (sqrt)": v["k0_sqrt"]})
    display(pd.DataFrame(rows))
    nn = np.linspace(5, 60, 100)
    plt.figure(figsize=(7,5))
    plt.plot(nn, np.sqrt(nn), "k--", label="sqrt(n) (old default)")
    for d, v in sl.items():
        plt.plot(nn, v["mu_c"]*nn**v["mu_beta"], label=f"{d}: {v['mu_c']:.2f}·n^{v['mu_beta']:.2f}")
    plt.xlabel("graph size n"); plt.ylabel("ideal group size μ"); plt.legend()
    plt.title("Learned group-size law per database"); plt.tight_layout(); plt.show()""")

md("""## 7. Default → tuned results (val and held-out test)

Per dataset, win-cell counts and mean normalized margin, default vs tuned. The Pareto guard
guarantees tuned never regresses a cell the default won (on val); the test column is the
held-out generalization.""")
code("""rep = os.path.join(BOARD, "report.csv")
if os.path.exists(rep):
    df = pd.read_csv(rep); display(df)
    fig, axes = plt.subplots(1, 2, figsize=(13,4))
    x = np.arange(len(df)); w = 0.35
    for ax, split in zip(axes, ["val", "test"]):
        ax.bar(x-w/2, df[f"{split}_wins_default"], w, label="default", color="#999")
        ax.bar(x+w/2, df[f"{split}_wins_tuned"], w, label="tuned", color="#C44E52")
        ax.set_xticks(x); ax.set_xticklabels(df["dataset"], rotation=30, ha="right")
        ax.set_title(f"{split} win-cells"); ax.legend()
    plt.tight_layout(); plt.show()
    print(f"TOTAL test wins: {df.test_wins_default.sum()} -> {df.test_wins_tuned.sum()} "
          f"({df.test_wins_tuned.sum()-df.test_wins_default.sum():+d})")""")

md("""## 8. Generalization — leave-one-dataset-out (LODO)

Train the surrogate on the *other* datasets, predict the held-out one's HP from its
meta-features alone. `heldout_R2` = how well the surrogate ranks configs on an unseen database;
`theta_dist` = distance between the predicted HP and the directly-searched best (lower=better).""")
code("""lodo = jload("lodo.json")
if lodo:
    display(pd.DataFrame([{"dataset": d, "heldout_R2": round(v["heldout_r2"],3),
                           "theta_dist_to_searched": round(v["theta_dist_to_searched"],3)}
                          for d, v in lodo.items()]))""")

md("""## 9. Calibrated hyperparameters per dataset

The cell below prints the calibrated configuration for each dataset: the entropy/granularity
weights `gamma_H`/`gamma_S`, the learned group-size law `mu = mu_c * n^mu_beta` (with the number
of initial groups `k0 = ceil(n / mu)`), and the node- and group-level interaction-augmentation
coefficients.""")
code("""for d in DATASETS:
    b = jload(f"best_hp_{d}.json")
    if not b: continue
    hp = b["hp"]; acc = b.get("accepted")
    print(f"# {d}  (accepted={acc})")
    print(f"    gamma_H={hp['gamma_H']:.3f}  gamma_S={hp['gamma_S']:.3f}")
    print(f"    mu_ideal = {hp['mu_c']:.3f} * n**{hp['mu_beta']:.3f}   "
          f"# k0 = ceil(n/mu_ideal)")
    print(f"    node_aug_coef = {hp['node_aug_coef']:.3f}   "
          f"group_aug_coef = {hp['group_aug_coef']:.3f}\\n")""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

# nbformat wants source as list of lines (keepends)
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)

out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "hpo_tuning.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(CELLS), "cells")

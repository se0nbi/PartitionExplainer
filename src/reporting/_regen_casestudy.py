"""Regenerate the molecular case-study figures with light pastel highlight fills."""
import sys, types, os, json, shutil, random
import numpy as np
import matplotlib
matplotlib.use("Agg")

NB = "functionalgroup.ipynb"
PATH_REWRITES = [
    ("/content/outputs", "outputs"),
]
SOURCE_PATCHES = [
    ("import torch, torch_geometric, torch_sparse, torch_scatter, torch_cluster, pyg_lib",
     "import torch, torch_geometric\ntry:\n    import torch_sparse, torch_scatter, torch_cluster, pyg_lib\nexcept ImportError:\n    pass"),
    # Lighten highlight fills toward white so the atom labels stay legible.
    ("highlightAtomColors=atom_colors,",
     "highlightAtomColors={_k: tuple(_ci+(1-_ci)*0.62 for _ci in _v) for _k,_v in atom_colors.items()},"),
]
DEMO = [("acetaminophen", "CC(=O)Nc1ccc(cc1)O"),
        ("2,4-dinitrophenol", "Oc1ccc(cc1[N+]([O-])=O)[N+]([O-])=O"),
        ("cyclophosphamide", "ClCCN(CCCl)P1(=O)NCCCO1")]
FIGMAP = {"acetaminophen": "figures/fig_molecule.pdf",
          "2,4-dinitrophenol": "figures/fig_dnp.pdf",
          "cyclophosphamide": "figures/fig_cyclo.pdf"}


def cell_src(idx, nb):
    s = "".join(nb["cells"][idx]["source"])
    for o, n in PATH_REWRITES:
        s = s.replace(o, n)
    for o, n in SOURCE_PATCHES:
        s = s.replace(o, n)
    return s


def seed_all(s=42):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def main():
    import torch
    nb = json.load(open(NB, encoding="utf-8"))
    stub = types.ModuleType("_fg_runner"); sys.modules["_fg_runner"] = stub
    ns = {"__name__": "_fg_runner"}; stub.__dict__.update(ns)
    for idx in range(0, 8):                       # definition cells only
        try:
            exec(compile(cell_src(idx, nb), f"<cell {idx}>", "exec"), ns, ns)
            stub.__dict__.update(ns)
        except Exception as e:
            print(f"  (cell {idx} skipped: {str(e)[:80]})", flush=True)
    DumplingGNN = ns["DumplingGNN"]; explain = ns["explain_molecule"]; savefig = ns["save_case_study_figure"]
    dev = torch.device("cpu")
    m = DumplingGNN(input_dim=8, hidden_channels=64, dropout=0.1).to(dev)
    m.load_state_dict(torch.load("outputs/models/dumplinggnn_tox21_8dim.pth", map_location=dev, weights_only=False))
    m.eval()
    print("model loaded", flush=True)
    os.makedirs("outputs/case_studies", exist_ok=True)
    for name, smi in DEMO:
        seed_all(42)                              # deterministic partition/values
        res = explain(m, smi, dev, n_samples_st2=100, n_samples_sv=100, sa_iterations=300, verbose=False)
        out = f"outputs/case_studies/case_study_{name.replace(' ', '_').replace(',', '')}.pdf"
        savefig(res, name, out)
        shutil.copy(out, FIGMAP[name])
        dz = res["vfunc_stats"]["total_change"]
        gv = [round(v, 3) for v in res["group_agg"]["group_values"]]
        print(f"  {name}: dz={dz:+.3f} group_values={gv} -> {FIGMAP[name]}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

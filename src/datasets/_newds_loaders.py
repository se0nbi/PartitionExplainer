"""Dataset loaders for the SPMotif, BA House/Grid, and Graph-Twitter datasets."""
import os
import sys

import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data

import _hpo_engine as E

# PyG/DIG processed caches (SPMotif / SentiGraph) need torch.load(weights_only=False) on torch>=2.6.
E.patch_torch_load()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NEWDS_DATA = os.path.join(ROOT, "outputs", "dig_data_newds")
os.makedirs(NEWDS_DATA, exist_ok=True)
MAGE_DATASET_DIR = os.path.join(ROOT, "_vendor", "MAGE", "dataset")

SPMOTIF_BIASES = {"spmotif_0.5": 0.5, "spmotif_0.7": 0.7, "spmotif_0.9": 0.9}
BA_TASKS = {"ba_house_grid": "housegrid", "ba_house_or_grid": "or"}

_HOUSE = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4)]                       # 5 nodes
_GRID = [(r * 3 + c, r * 3 + c + 1) for r in range(3) for c in range(2)] + \
        [(r * 3 + c, r * 3 + c + 3) for r in range(2) for c in range(3)]        # 9 nodes


def _to_list(dataset):
    return [dataset[i] for i in range(len(dataset))]


def _load_spmotif(name):
    b = SPMOTIF_BIASES[name]
    if MAGE_DATASET_DIR not in sys.path:
        sys.path.insert(0, MAGE_DATASET_DIR)
    import spmotif as _sp  # top-level import avoids the package __init__ dig chain
    root = os.path.join(NEWDS_DATA, name)
    ds = _sp.SPMotif(root, name=name, b=b)
    graphs = []
    for d in _to_list(ds):
        d = d.clone()
        d.x = d.x.float()
        nl = getattr(d, "node_label", None)
        if nl is not None:
            d.gt_node_mask = (nl.view(-1) != 0)
        graphs.append(d)
    input_dim = int(graphs[0].x.size(1))
    ncls = int(max(int(g.y.view(-1)[0]) for g in graphs)) + 1
    avg_n = float(np.mean([g.num_nodes for g in graphs]))
    n_gt = sum(int(bool(getattr(g, "gt_node_mask", torch.zeros(1)).any())) for g in graphs)
    info = {"name": name, "task": "graph_classification", "input_dim": input_dim,
            "num_classes": ncls, "has_ground_truth": True,
            "ground_truth_type": "node_mask", "avg_nodes": avg_n}
    print(f"  Loaded {name}: {len(graphs)} graphs, input_dim={input_dim}, "
          f"num_classes={ncls}, avg_nodes={avg_n:.1f}, graphs_with_GT={n_gt}", flush=True)
    return graphs, info


def _gen_ba_variant(task, n_graphs=1000, seed=42):
    rng = np.random.RandomState(seed)
    if task == "housegrid":
        combos, probs = ["house", "grid"], [0.5, 0.5]
        def label(c): return 0 if c == "house" else 1
    elif task == "or":
        combos, probs = ["none", "house", "grid", "both"], [0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3]
        def label(c): return 0 if c == "none" else 1
    else:  # and
        combos, probs = ["none", "house", "grid", "both"], [0.5 / 3, 0.5 / 3, 0.5 / 3, 0.5]
        def label(c): return 1 if c == "both" else 0
    graphs = []
    for _ in range(n_graphs):
        combo = combos[int(rng.choice(len(combos), p=probs))]
        nb = int(rng.randint(12, 19))
        G = nx.barabasi_albert_graph(nb, 1, seed=int(rng.randint(1_000_000)))
        motif_nodes = []

        def attach(edges, k):
            base = G.number_of_nodes()
            for a, b in edges:
                G.add_edge(base + a, base + b)
            G.add_edge(int(rng.randint(nb)), base)
            motif_nodes.extend(range(base, base + k))

        if combo in ("house", "both"):
            attach(_HOUSE, 5)
        if combo in ("grid", "both"):
            attach(_GRID, 9)
        n = G.number_of_nodes()
        deg = dict(G.degree()); clus = nx.clustering(G)
        x = torch.tensor([[deg[i], clus[i], 1.0 / (deg[i] + 1)] for i in range(n)],
                         dtype=torch.float)
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        es = list(G.edges())
        ei = torch.tensor(es + [(b, a) for a, b in es], dtype=torch.long).t().contiguous()
        d = Data(x=x, edge_index=ei, y=torch.tensor([label(combo)], dtype=torch.long))
        m = torch.zeros(n, dtype=torch.bool)
        if motif_nodes:
            m[motif_nodes] = True
        d.gt_node_mask = m
        graphs.append(d)
    return graphs


def _load_ba_variant(name):
    task = BA_TASKS[name]
    cache = os.path.join(NEWDS_DATA, name + ".pt")
    if os.path.exists(cache):
        graphs = torch.load(cache, weights_only=False)
    else:
        graphs = _gen_ba_variant(task, 1000, seed=42)
        torch.save(graphs, cache)
    avg_n = float(np.mean([g.num_nodes for g in graphs]))
    n_gt = sum(int(bool(g.gt_node_mask.any())) for g in graphs)
    info = {"name": name, "task": "graph_classification", "input_dim": 3,
            "num_classes": 2, "has_ground_truth": True,
            "ground_truth_type": "node_mask", "avg_nodes": avg_n}
    print(f"  Loaded {name}: {len(graphs)} graphs, input_dim=3, num_classes=2, "
          f"avg_nodes={avg_n:.1f}, graphs_with_GT={n_gt}", flush=True)
    return graphs, info


def _load_graph_twitter(name="graph_twitter"):
    from dig.xgraph.dataset import SentiGraphDataset
    root = os.path.join(NEWDS_DATA, "Graph-Twitter")
    ds = SentiGraphDataset(root, name="Graph-Twitter")
    graphs = []
    for d in _to_list(ds):
        d = d.clone()
        d.x = d.x.float()
        graphs.append(d)
    input_dim = int(graphs[0].x.size(1))
    ncls = int(ds.num_classes)
    avg_n = float(np.mean([g.num_nodes for g in graphs]))
    info = {"name": name, "task": "graph_classification", "input_dim": input_dim,
            "num_classes": ncls, "has_ground_truth": False,
            "ground_truth_type": "none", "avg_nodes": avg_n}
    print(f"  Loaded {name}: {len(graphs)} graphs, input_dim={input_dim}, "
          f"num_classes={ncls}, avg_nodes={avg_n:.1f}", flush=True)
    return graphs, info


NEW_DATASETS = list(SPMOTIF_BIASES) + list(BA_TASKS) + ["graph_twitter"]

_orig_install = E.install_extra_loaders


def _patched_install(ns):
    _orig_install(ns)
    base_loader = ns["load_dig_dataset"]

    def loader(name):
        key = name.lower()
        if key in SPMOTIF_BIASES:
            return _load_spmotif(key)
        if key in BA_TASKS:
            return _load_ba_variant(key)
        if key in ("graph_twitter", "graphtwitter", "graph-twitter"):
            return _load_graph_twitter("graph_twitter")
        return base_loader(name)

    ns["load_dig_dataset"] = loader


E.install_extra_loaders = _patched_install
print("[_newds_loaders] patched E.install_extra_loaders; new datasets:",
      ", ".join(NEW_DATASETS), flush=True)

"""Graph explainer baselines (MAGE and SAME) wrapping a GCN with adapters.

  MAGE  (Bui et al., ICML 2024): Myerson-Taylor structure-aware explainer. Calls
        model(data=Data) and returns (motifs, info); info["indices"] is the n*n
        Myerson-Taylor interaction matrix. Per-node score = diag + 0.5*sum|off-diag|.
  SAME  (Ye et al., NeurIPS 2023): structure-aware Shapley multipiece explanation via
        MCTS + find_explanations. Calls model(batch)->probs; returns a node
        coalition (binary mask).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

_MAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "_vendor", "MAGE"))
if _MAGE_DIR not in sys.path:
    sys.path.insert(0, _MAGE_DIR)


class _MageAdapter(torch.nn.Module):
    """MAGE invokes model(data=Data) and expects class logits."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, data=None, x=None, edge_index=None, batch=None, **kw):
        if data is not None:
            x = data.x
            edge_index = data.edge_index
            batch = getattr(data, "batch", None)
        out = self.m(x, edge_index, batch)
        return out if out.dim() > 1 else out.unsqueeze(0)


class _SameAdapter(torch.nn.Module):
    """SAME invokes model(batch) and expects class probabilities."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, batch):
        out = self.m(batch.x, batch.edge_index, getattr(batch, "batch", None))
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return F.softmax(out, dim=-1)


def run_mage(model, data, device, num_classes=2, num_motifs=3, beta=0.5,
             order=2, num_samples=50):
    """MAGE Myerson-Taylor -> per-node importance (np.ndarray) or None."""
    try:
        from mage.mage import Mage
        from mage.maskers import PyGDataMasker
        adapter = _MageAdapter(model).to(device).eval()
        d = data.clone().to(device)
        explainer = Mage(adapter, masker=PyGDataMasker(),
                         payoff_type="norm_prob", device=device)
        motifs, info = explainer.explain(
            d, num_motifs=num_motifs, beta=beta, ord=order, method="myt",
            num_samples=num_samples, target_class="max", connectivity="viky")
        M = np.asarray(info["indices"], dtype=float)
        n = int(data.x.size(0))
        if M.shape[0] != n:
            return None
        diag = np.diag(M)
        node = diag + 0.5 * (np.abs(M).sum(axis=1) - np.abs(diag))
        return node if np.isfinite(node).all() and node.any() else None
    except Exception:
        return None


def run_same(model, data, device, num_classes=2, rollout=20, sample_num=50,
             local_radius=4, c_puct=10.0, max_ex_size=5):
    """SAME structure-aware Shapley multipiece -> binary node-coalition mask or None."""
    try:
        from baselines.utils.same import (GnnNets_GC2value_func, MCTS, reward_func,
                                           find_explanations)
        adapter = _SameAdapter(model).to(device).eval()
        d = data.clone().to(device)
        n = int(d.x.size(0))
        if n < 3:
            return None
        with torch.no_grad():
            logit = model(d.x, d.edge_index, getattr(d, "batch", None))
            pred = int((logit if logit.dim() > 1 else logit.unsqueeze(0)).argmax(-1))
        d.y = torch.tensor(pred)
        max_ex = int(min(max_ex_size, max(2, n - 1)))
        expand = int(min(14, n))
        param = SimpleNamespace(
            reward_method="mc_l_shapley", local_raduis=local_radius,
            sample_num=sample_num, c_puct=c_puct, single_explanation_size=3,
            candidate_size=7, explanation_exploration_method="permutation",
            rollout=rollout, min_atoms=max_ex, expand_atoms=expand,
            high2low=True, max_ex_size=max_ex)
        cfg = SimpleNamespace(
            explainers=SimpleNamespace(param=param),
            models=SimpleNamespace(param=SimpleNamespace(graph_classification=True)),
            datasets=SimpleNamespace(subgraph_building_method="split"))
        value_func = GnnNets_GC2value_func(adapter, target_class=d.y)
        payoff = reward_func(param, value_func, subgraph_building_method="split")
        mcts = MCTS(d.x, d.edge_index, score_func=payoff, n_rollout=rollout,
                    min_atoms=max_ex, c_puct=c_puct, expand_atoms=expand, high2low=True)
        results = mcts.mcts(verbose=False)
        final = find_explanations(results, max_nodes=max_ex, gnnNets=adapter, data=d,
                                  config=cfg, subgraph_building_method="split")
        coalition = list(final.coalition)
        s = np.zeros(n)
        for c in coalition:
            if 0 <= int(c) < n:
                s[int(c)] = 1.0
        return s if s.sum() > 0 else None
    except Exception:
        return None


def run_graphext(model, data, device, num_classes=2, n_samples=80, seed=0):
    """GraphEXT (Wu, Hao & Fan, IJCAI-2025): Shapley value under structural
    externalities, implemented from the paper's Algorithms 1 & 2.

    Value fn V(S,P): partition P is the cycle-decomposition of a random permutation;
    S's nodes are grouped into connected components using only graph edges WHOSE
    ENDPOINTS SHARE A P-COALITION; V = sum over those components R of f(G_R)[pred],
    where G_R is the induced subgraph and f the GNN's predicted-class probability.
    Estimated by sampling a random join order pi + random partition P per sample and
    accumulating each node's incremental marginal contribution as it joins S
    (Algorithm 2). Components updated incrementally (union-find) so cost is O(T*n)
    forward passes, not O(T*n^2). Returns per-node phi (np.ndarray) or None."""
    try:
        from collections import defaultdict
        d = data.clone().to(device)
        model.eval()
        n = int(d.x.size(0))
        if n < 2:
            return None
        x = d.x
        ei = d.edge_index.cpu().numpy()
        with torch.no_grad():
            out = model(d.x, d.edge_index, getattr(d, "batch", None))
            out = out if out.dim() > 1 else out.unsqueeze(0)
            pred = int(out.argmax(-1))
        adj = defaultdict(set)
        for a, b in zip(ei[0].tolist(), ei[1].tolist()):
            if a != b:
                adj[a].add(b); adj[b].add(a)

        fcache = {}
        def f_comp(nodes_fs):
            v = fcache.get(nodes_fs)
            if v is not None:
                return v
            nodes = sorted(nodes_fs)
            remap = {nd: i for i, nd in enumerate(nodes)}
            es = [(remap[a], remap[b]) for a in nodes for b in adj[a] if b in nodes_fs]
            e = (torch.tensor(es, dtype=torch.long, device=device).t().contiguous()
                 if es else torch.zeros((2, 0), dtype=torch.long, device=device))
            idx = torch.tensor(nodes, dtype=torch.long, device=device)
            bb = torch.zeros(len(nodes), dtype=torch.long, device=device)
            with torch.no_grad():
                o = model(x[idx], e, bb)
                o = o if o.dim() > 1 else o.unsqueeze(0)
                val = float(torch.softmax(o, dim=-1)[0, pred])
            fcache[nodes_fs] = val
            return val

        rng = np.random.RandomState((seed * 1000003 + n) % (2 ** 31 - 1))
        phi = np.zeros(n)
        for _t in range(n_samples):
            pi = rng.permutation(n)
            A = rng.permutation(n)
            part = -np.ones(n, dtype=int); c = 0
            for i in range(n):
                if part[i] < 0:
                    j = i
                    while part[j] < 0:
                        part[j] = c; j = int(A[j])
                    c += 1
            comp_of, comp_nodes, comp_val, nextid = {}, {}, {}, 0
            for step in range(n):
                u = int(pi[step])
                merge_ids = {comp_of[w] for w in adj[u]
                             if w in comp_of and part[w] == part[u]}
                newnodes = {u}
                old_sum = 0.0
                for cid in merge_ids:
                    newnodes |= comp_nodes[cid]
                    old_sum += comp_val[cid]
                newval = f_comp(frozenset(newnodes))
                phi[u] += (newval - old_sum)
                for cid in merge_ids:
                    del comp_nodes[cid]; del comp_val[cid]
                comp_nodes[nextid] = newnodes; comp_val[nextid] = newval
                for nd in newnodes:
                    comp_of[nd] = nextid
                nextid += 1
        phi /= n_samples
        return phi if np.isfinite(phi).all() and phi.any() else None
    except Exception:
        return None

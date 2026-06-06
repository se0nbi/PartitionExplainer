# Results

Faithfulness is scored per `(dataset, metric)` cell over four probability-space metrics
(harmonic fidelity, necessity Fid+, sufficiency Fid−, characterization) at five sparsity levels,
using fair common-graph averaging across all methods. PARSE is compared against ten explainers:
GNNExplainer, PGExplainer, GraphSHAP-IQ, GStarX, SME, SubgraphX, GraphSVX, MAGE, SAME, and GraphEXT.

Regenerate everything below with:

```bash
python src/evaluation/_paper_artifacts.py board   # leaderboard
python src/evaluation/_paper_artifacts.py stats   # dataset statistics
python src/evaluation/_paper_artifacts.py fig     # comparison figure
```

## Leaderboard

**PARSE wins 51 of 60 (dataset, metric) cells (85.0%)** and is the top method on every dataset. The
nearest baselines (GStarX and GraphEXT) take 4 cells each, SAME takes 1, and the seven saliency /
exact-Shapley baselines take none.

| Metric | PARSE | Next best |
|---|---|---|
| Sufficiency (Fid−) | **15 / 15** | — |
| Characterization | **15 / 15** | — |
| Necessity (Fid+) | **12 / 15** | GraphEXT 2, SAME 1 |
| Harmonic fidelity | 9 / 15 | GStarX 4, GraphEXT 2 |

The group-level partition alone — the single-output explanation — is itself first on 44 / 60 cells (73%).

| Split | PARSE cells | Datasets led |
|---|---|---|
| Real-world (9 datasets) | 31 / 36 (86%) | 9 / 9 |
| Synthetic (6 datasets) | 20 / 24 (83%) | 6 / 6 |

## Datasets

Correctness is evaluated on three controlled interaction tasks with known ground-truth rules
(AND / OR / DIST); faithfulness is evaluated on the fifteen-dataset benchmark suite.

| Dataset | Type | Graphs | Avg. nodes | Classes |
|---|---|---:|---:|---:|
| AND | Controlled | 5,066 | 15.9 | 2 |
| OR | Controlled | 5,021 | 15.9 | 2 |
| DIST | Controlled | 3,795 | 19.2 | 2 |
| BA-2Motifs | Synthetic | 1,000 | 25.0 | 2 |
| BA-House-Grid | Synthetic | 1,000 | 21.9 | 2 |
| BA-House-OR-Grid | Synthetic | 1,000 | 19.7 | 2 |
| SPMotif-0.5 | Synthetic | 18,000 | 45.5 | 3 |
| SPMotif-0.7 | Synthetic | 18,000 | 46.4 | 3 |
| SPMotif-0.9 | Synthetic | 18,000 | 47.4 | 3 |
| Benzene | Real-world | 12,000 | 20.6 | 2 |
| BACE | Real-world | 1,513 | 34.1 | 2 |
| BBBP | Real-world | 2,038 | 24.1 | 2 |
| MUTAG | Real-world | 188 | 17.9 | 2 |
| PROTEINS | Real-world | 1,113 | 39.1 | 2 |
| ENZYMES | Real-world | 600 | 32.6 | 6 |
| Graph-SST2 | Real-world | 70,042 | 10.2 | 2 |
| Graph-SST5 | Real-world | 11,855 | 19.8 | 5 |
| Graph-Twitter | Real-world | 6,940 | 21.1 | 3 |

## Molecular case studies

PARSE is applied to three well-characterized toxic molecules from Tox21 SR-MMP (acetaminophen,
2,4-dinitrophenol, cyclophosphamide), where the recovered group interactions align with established
chemical mechanisms. Figures are regenerated with `python src/reporting/_regen_casestudy.py`.

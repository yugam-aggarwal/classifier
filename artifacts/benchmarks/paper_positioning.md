# Paper Positioning Snapshot

## Headline Verdicts
- `quality_first_default`: The current default clears the structural and predictive competitiveness criteria (3/5 structural wins, 2/2 competitive real-dataset results).
- `runtime_limitation`: The current default does not satisfy the runtime rule against greedy_hc_bn (0/5 runtime wins).
- `fast_profile_tradeoff`: known_graph_fast is a real speedup but not a new default: it improves SHD on 2/5 known-graph datasets, regresses on 3/5, and still fails the runtime gate.

## Explicit-BN Structural Comparison
| Dataset | Method SHD | Best Explicit | Best Explicit SHD | SHD Delta |
| --- | --- | --- | --- | --- |
| alarm | 26.2500 | greedy_hc_bn | 26.0000 | -0.2500 |
| child | 11.7500 | greedy_hc_bn | 12.0000 | 0.2500 |
| insurance | 19.7500 | greedy_hc_bn | 26.0000 | 6.2500 |
| synthetic_medium | 6.5000 | greedy_hc_bn | 4.2500 | -2.2500 |
| synthetic_small | 1.2500 | greedy_hc_bn | 1.5000 | 0.2500 |

## Strong Predictive Baseline Comparison
| Dataset | Method ROC-AUC | Best Strong | Best Strong ROC-AUC | ROC-AUC Gap | Competitive |
| --- | --- | --- | --- | --- | --- |
| sklearn_breast_cancer | 0.9914 | tabpfn | 0.9979 | 0.0064 | yes |
| sklearn_wine | 0.9992 | tabpfn | 0.9997 | 0.0006 | yes |

## Runtime vs `greedy_hc_bn`
| Dataset | Method Seconds | Greedy Seconds | Ratio vs Greedy | Runtime Win |
| --- | --- | --- | --- | --- |
| alarm | 31.1940 | 1.1545 | 27.0187 | no |
| child | 14.3571 | 0.0885 | 162.2013 | no |
| insurance | 21.9969 | 0.1265 | 173.9157 | no |
| synthetic_medium | 1.8486 | 0.0264 | 70.0208 | no |
| synthetic_small | 1.4360 | 0.0208 | 69.0931 | no |

## `known_graph_fast` Tradeoff vs Default
| Dataset | Default Seconds | Fast Seconds | Greedy Seconds | Default SHD | Fast SHD | SHD Delta |
| --- | --- | --- | --- | --- | --- | --- |
| alarm | 32.3765 | 5.8240 | 1.0636 | 25.0000 | 23.7500 | -1.2500 |
| child | 14.2787 | 3.0621 | 0.0846 | 12.5000 | 14.5000 | 2.0000 |
| insurance | 22.0723 | 4.5510 | 0.1211 | 19.0000 | 25.0000 | 6.0000 |
| synthetic_medium | 1.8654 | 0.3958 | 0.0258 | 6.5000 | 5.2500 | -1.2500 |
| synthetic_small | 1.7906 | 0.3644 | 0.0213 | 1.7500 | 2.2500 | 0.5000 |

## Recommended Claim Text
Neural-screened BN should be framed as a quality-first explicit BN classifier: the current default wins the staged structural criterion on 3/5 known-graph datasets and remains ROC-AUC competitive on 2/2 real datasets, while preserving explicit, inspectable DAGs.

## Recommended Limitation Text
The main limitation is runtime, not structural or predictive quality: the current default wins the staged runtime rule on only 0/5 known-graph datasets against greedy_hc_bn. The known_graph_fast ablation narrows the runtime gap substantially but is not promotable because it still loses the runtime gate and regresses SHD on 3 of 5 known-graph datasets.

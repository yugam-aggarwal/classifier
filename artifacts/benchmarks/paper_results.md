# Paper Results Draft

## Headline Metrics
- Structural wins over the best explicit BN baseline: 3/5
- Near ties on known-graph SHD (<= 0.5): 3/5
- Predictively competitive real datasets: 2/2
- Runtime wins vs `greedy_hc_bn`: 0/5

## Structural Table
| Dataset | Method SHD | Precision | Recall | Best Explicit | Best Explicit SHD | SHD Margin |
| --- | --- | --- | --- | --- | --- | --- |
| alarm | 26.2500 | 0.4093 | 0.3370 | greedy_hc_bn | 26.0000 | -0.2500 |
| child | 11.7500 | 0.5176 | 0.4000 | greedy_hc_bn | 12.0000 | 0.2500 |
| insurance | 19.7500 | 0.6933 | 0.4231 | greedy_hc_bn | 26.0000 | 6.2500 |
| synthetic_medium | 6.5000 | 0.1875 | 0.0667 | greedy_hc_bn | 4.2500 | -2.2500 |
| synthetic_small | 1.2500 | 0.7500 | 0.2375 | greedy_hc_bn | 1.5000 | 0.2500 |

## Predictive Table
| Dataset | Method ROC-AUC | Method Log Loss | Best Strong | Best Strong ROC-AUC | ROC-AUC Gap |
| --- | --- | --- | --- | --- | --- |
| sklearn_breast_cancer | 0.9914 | 0.1126 | tabpfn | 0.9979 | 0.0064 |
| sklearn_wine | 0.9992 | 0.0708 | tabpfn | 0.9997 | 0.0006 |

## Runtime Table
| Dataset | Default Seconds | known_graph_fast Seconds | Greedy Seconds | Default/Greedy | Fast Reduction % |
| --- | --- | --- | --- | --- | --- |
| alarm | 31.1940 | 5.3917 | 1.1545 | 27.0187 | 82.7155 |
| child | 14.3571 | 2.9341 | 0.0885 | 162.2013 | 79.5637 |
| insurance | 21.9969 | 4.3403 | 0.1265 | 173.9157 | 80.2688 |
| synthetic_medium | 1.8486 | 0.3560 | 0.0264 | 70.0208 | 80.7447 |
| synthetic_small | 1.4360 | 0.2976 | 0.0208 | 69.0931 | 79.2769 |

## Results Narrative Draft
- Across the five known-graph datasets, the default neural-screened BN beats the best explicit BN baseline on SHD on 3/5 datasets and is within 0.5 SHD on 3/5. The strongest structural result is on insurance, where the method reaches SHD 19.75 versus 26.0 for the best explicit baseline, while alarm is effectively tied at 26.25 versus 26.0.
- On the two real datasets, the method remains predictively competitive with strong tabular baselines while keeping an explicit DAG: it is within the staged ROC-AUC competitiveness gap on 2/2 datasets. Specifically, for Breast cancer, it reaches ROC-AUC 0.9914 with log loss 0.1126, and for Wine, it reaches ROC-AUC 0.9992 with log loss 0.0708.
- The limiting factor is runtime, not quality. The default method is slower than greedy_hc_bn on all 5/5 known-graph datasets, and the known_graph_fast ablation reduces runtime materially but still does not beat greedy_hc_bn on any dataset or earn promotion into the default path.

## Limitations Draft
- The current method should not be presented as a speed-oriented replacement for greedy hill-climbing BN search. Its strength is the quality/interpretability tradeoff, not raw runtime.
- Incremental runtime tuning appears exhausted in the current design. The separate fast-mode probe found no candidate that beat greedy_hc_bn on at least 3/5 known-graph datasets while staying within the SHD tolerance gate.

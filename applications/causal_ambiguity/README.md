# Order-Indexed Causal Ambiguity - HCAT application module

This module extends Higher-Order Conditional Attainability (HCAT) to causal discovery through bounded-evidence compatibility classes, structural/interventional/task-relative ambiguity, decision sufficiency, and consequence-directed intervention design.

## One-click reproduction

```bash
pip install -r applications/causal_ambiguity/requirements.txt
python applications/causal_ambiguity/reproduce.py
```

The reproduction now runs both the original exact finite audit and a controlled finite-sample synthetic benchmark, followed by the automated tests.

## Exact finite audit

The exact audit retains the deliberately over-informative ordered-DAG surrogate as a falsification check, verifies the explicit SCM realization of cardinality--consequence incomparability, verifies the exact maximin regret certificate, and reproduces the constructed intervention-design separation.

## Synthetic benchmark validation (2026-09-05)

The benchmark compares four matched intervention policies: OICA task-ambiguity minimization, compatible-class-size reduction, a graph-oriented residual-diameter proxy, and random intervention. It spans dimensions 5, 10, and 20; linear-Gaussian, nonlinear-additive, and binary-logistic mechanisms; sample sizes 250, 500, 1000, and 5000; 12 seeds; and at most 10 intervention steps. This gives 432 generated benchmark instances per policy and 1,728 matched policy runs.

With decision-sufficiency tolerance epsilon=0.30, the reproducible run reports:

| Policy | success rate | mean stopping time | final task ambiguity | final regret | final graph-diameter proxy |
|---|---:|---:|---:|---:|---:|
| OICA | 0.6574 | 9.1227 | 0.2415 | 0.0129 | 0.1983 |
| Graph | 0.4792 | 10.0532 | 0.3140 | 0.0145 | 0.1688 |
| ClassSize | 0.2639 | 10.5486 | 0.4551 | 0.0339 | 0.2142 |
| Random | 0.5671 | 9.6782 | 0.2720 | 0.0117 | 0.1912 |

The paired OICA-minus-Graph difference in stopping time is -0.9306 interventions (95% bootstrap CI -1.0880 to -0.7731), and the paired difference in final task ambiguity is -0.0724 (95% CI -0.0895 to -0.0556). OICA and the graph-oriented criterion select different first interventions in 83.10% of matched instances. OICA does not dominate every endpoint: the graph proxy attains a smaller final structural-diameter proxy and OICA's final-regret confidence interval overlaps zero. These distinctions are retained as part of the falsifiable evaluation rather than suppressed.

### Scope boundary

This run is a controlled synthetic finite-sample benchmark. It is **not** an external CSuite validation and no CSuite superiority claim is made. The repository contains the CSuite-ready benchmark framing, but external CSuite execution should be reported only after those data are actually run.

## Outputs

- `results/synthetic_benchmark_trajectories.csv`
- `results/synthetic_benchmark_endpoints.csv`
- `results/synthetic_benchmark_summary.csv`
- `results/paired_oica_vs_graph.csv`
- `reports/BENCHMARK_SUMMARY.json`
- `reports/BENCHMARK_PROTOCOL.md`
- `reports/final_reproduction_benchmark.txt`
- generated benchmark figures in `figures/`

## HCAT parent framework

Akhtar, M. A. K. (2026). *Higher-Order Conditional Attainability: Opacity, Observation Order, and Decision Guarantees Beyond Local Views* (Version V1). Zenodo. https://doi.org/10.5281/zenodo.22336652

## License

Apache License 2.0. See the repository root LICENSE and this module's NOTICE.

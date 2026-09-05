# HCAT v1.0 - Higher-Order Conditional Attainability

A reproducible research package for bounded-order observation classes, exact finite opacity, attainability rank, and robust-decision guarantees.

## GitHub workflow
Upload this folder to a GitHub repository. Open **Actions -> HCAT Reproduce -> Run workflow**. The workflow installs dependencies, runs all tests, regenerates every CSV and figure, and uploads the results as a GitHub Actions artifact.

## Local reproduction
```bash
pip install -r requirements.txt
pytest -q
python reproduce.py
python app.py
```

## What is exact
For binary systems with `d=4`, the package enumerates **all 65,535 nonempty global relations**. Therefore compatible-class size, reconstruction risk, task opacity, full-order rank, and the finite decision guarantee are exact for the reported d=4 experiments. The parity scaling theorem is analytic for arbitrary finite d.

## Experiments
1. Five exact finite families: parity, threshold, sparse XOR, one-hot, hypergraph constraints.
2. 250 random global relations with exact bounded-order compatibility classes and paired decision comparison.
3. Wisconsin Diagnostic Breast Cancer structural support audit (569 samples, UCI/scikit-learn copy), with three median-binarized high-correlation features plus diagnosis.
4. 200 bootstrap resamples for rank stability.
5. Parity hierarchy scaling.

## Important boundary
The package does not claim that parity, lossless joins, marginal inconsistency, local-to-global obstruction, robust optimization, or sup metrics are historically new. The proposed contribution is the HCAT organization of bounded-order observational equivalence classes into quantitative operational radius/opacity, sufficient observation order, and decision guarantees.

## Data
The real-data example uses the Wisconsin Diagnostic Breast Cancer dataset distributed with scikit-learn and originally hosted by the UCI Machine Learning Repository (DOI: 10.24432/C5DW2B). No patient identifiers are used.

## License
Apache License 2.0. Copyright 2026 Mohammad Amir Khusru Akhtar.

## Reproduced headline findings
- Exact d=4 enumeration over all 65,535 nonempty relations.
- 250 random targets per k for k=1,2,3; all 750 decision-bound audits pass.
- HCAT improves certified worst-case utility in the random benchmark at every tested k, while average realized regret is deliberately reported without a universal-dominance claim.
- WDBC structural support audit: 569 rows, exact rank 4, stable in 200 bootstrap resamples.

See `reports/RESULTS_SUMMARY.md` and the generated CSV files for exact values.

## Repository contents
`src/` contains the mathematical implementation; `data/` contains reproducible transformed data and metadata; `results/` and `figures/` are regenerated outputs; `tests/` contains automated checks; `reports/prior_prototypes/` preserves earlier experiment packages and reports used during development; `.github/workflows/reproduce.yml` provides the manual one-click workflow.

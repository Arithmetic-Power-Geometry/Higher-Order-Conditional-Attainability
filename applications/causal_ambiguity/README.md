# Order-Indexed Causal Ambiguity - HCAT application module

This module extends Higher-Order Conditional Attainability (HCAT) to a causal-discovery research program centered on bounded-order compatibility classes, structural/interventional/task-relevant ambiguity, decision sufficiency, and consequence-directed intervention design.

## Scope
The current finite benchmark is deliberately conservative. It implements an exact graph-level surrogate observation hierarchy over ordered DAGs and audits structural diameter, intervention-signature diameter, and a task-specific reachability diameter. It is a reproducible theorem/algorithm sandbox, not a claim that this surrogate is a universal statistical causal-observation operator. The module also contains a concrete finite SCM realization of the cardinality--consequence incomparability theorem and an executable audit of the exact maximin regret bound used in the manuscript.

## One-click reproduction
Run the GitHub Actions workflow `Causal Ambiguity Reproduction`, or locally:

```bash
pip install -r applications/causal_ambiguity/requirements.txt
python applications/causal_ambiguity/reproduce.py
```

Outputs are written to `applications/causal_ambiguity/results/` and reports to `applications/causal_ambiguity/reports/`.

## Scientific status
The package verifies the cardinality-versus-consequential-diameter separation, its explicit finite SCM realization, an exact robust-regret certificate, and a constructed intervention-design counterexample. The current exact graph-level surrogate is intentionally retained even though it is over-informative and produces singleton compatibility classes at the lowest implemented order; this negative result is part of the falsification audit.

## HCAT parent framework
The causal construction specializes Higher-Order Conditional Attainability. The parent framework is archived as:

Akhtar, M. A. K. (2026). *Higher-Order Conditional Attainability: Opacity, Observation Order, and Decision Guarantees Beyond Local Views* (Version V1). Zenodo. https://doi.org/10.5281/zenodo.22336652

## Manuscript alignment (revision 2026-09-05)
The companion manuscript now (i) cites the HCAT Version V1 Zenodo record, (ii) proves cardinality--consequence incomparability using an explicit SCM family, (iii) states the robust-decision theorem with the exact bound `regret <= L * radius <= L * diameter`, and (iv) normalizes the Geffner et al. (2022) CSuite citation to arXiv:2202.02195. The files `results/causal_scm_incomparability_realization.csv` and `results/exact_regret_certificate.csv` audit the two strengthened theorem claims.

## License
Apache License 2.0. See the repository root LICENSE and this module's NOTICE.

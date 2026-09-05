# Order-Indexed Causal Ambiguity - HCAT application module

This module extends Higher-Order Conditional Attainability (HCAT) to a causal-discovery research program centered on bounded-order compatibility classes, structural/interventional/task-relevant ambiguity, decision sufficiency, and consequence-directed intervention design.

## Scope
The current finite benchmark is deliberately conservative. It implements an exact graph-level surrogate observation hierarchy over ordered DAGs and audits structural diameter, intervention-signature diameter, and a task-specific reachability diameter. It is a reproducible theorem/algorithm sandbox, not a claim that this surrogate is a universal statistical causal-observation operator.

## One-click reproduction
Run the GitHub Actions workflow `Causal Ambiguity Reproduction`, or locally:

```bash
pip install -r applications/causal_ambiguity/requirements.txt
python applications/causal_ambiguity/reproduce.py
```

Outputs are written to `applications/causal_ambiguity/results/` and reports to `applications/causal_ambiguity/reports/`.

## Scientific status
The package verifies the cardinality-versus-consequential-diameter separation and a constructed intervention-design counterexample. The current exact graph-level surrogate is intentionally retained even though it is over-informative and produces singleton compatibility classes at the lowest implemented order; this negative result is part of the falsification audit.

## License
Apache License 2.0. See the repository root LICENSE and this module's NOTICE.

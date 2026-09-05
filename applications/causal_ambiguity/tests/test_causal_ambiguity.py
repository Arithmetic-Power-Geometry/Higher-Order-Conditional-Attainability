import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from causal_ambiguity import *

def test_metric_bounds():
    gs=all_ordered_dags(3)
    for a in gs:
      for b in gs:
        assert 0<=structural_distance(a,b)<=1
        assert 0<=intervention_distance(a,b,3)<=1
        assert 0<=task_distance(a,b,3)<=1

def test_cardinality_incomparability_constructed():
    x=synthetic_incomparability_examples()
    assert x['A_size']>x['B_size'] and x['A_omega']<x['B_omega']
    assert x['D_size']>x['C_size'] and x['D_omega']>x['C_omega']

def test_intervention_counterexample():
    x=intervention_counterexample()
    assert x['IA_size']<x['IB_size'] and x['IA_omega']>x['IB_omega']

def test_exact_audit_runs():
    gs=all_ordered_dags(4)
    for k in range(1,5):
        assert len(compatibility_classes(gs,4,k))>=1

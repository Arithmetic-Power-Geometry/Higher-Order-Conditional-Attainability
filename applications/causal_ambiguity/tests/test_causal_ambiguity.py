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

def test_causal_scm_realization():
    x=causal_scm_incomparability_realization()
    assert x['A']['cardinality']>x['B']['cardinality']
    assert x['A']['task_diameter']<x['B']['task_diameter']
    assert x['D']['cardinality']>x['C']['cardinality']
    assert x['D']['task_diameter']>x['C']['task_diameter']

def test_exact_regret_certificate():
    utilities={'safe':{'m0':0.7,'m1':0.7},'risky':{'m0':1.0,'m1':0.0}}
    d=lambda a,b: 0.0 if a==b else 1.0
    x=exact_maximin_regret_bound(utilities,'m0',['m0','m1'],d,L=1.0)
    assert x['verified']
    assert x['regret'] <= x['radius_bound'] + 1e-12
    assert x['radius_bound'] <= x['diameter_bound'] + 1e-12

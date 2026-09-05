from src.hcat import *
def test_parity_local_global():
    for k in (1,2,3):
        a=parity(4,0);b=parity(4,1)
        assert observation_signature(a,4,k)==observation_signature(b,4,k)
    assert observation_signature(parity(4,0),4,4)!=observation_signature(parity(4,1),4,4)
def test_exact_class_contains_target():
    t=threshold(4,2)
    for k in range(1,5): assert t in set(map(int,compatible_masks_exact(t,4,k)))
def test_nested_classes():
    t=hypergraph_demo(4);sizes=[len(compatible_masks_exact(t,4,k)) for k in range(1,5)]
    assert all(a>=b for a,b in zip(sizes,sizes[1:])) and sizes[-1]==1
def test_radius_diameter():
    t=threshold(4,2)
    for k in range(1,5):
        C=compatible_masks_exact(t,4,k);rho=task_radius_exact(t,C,4);om=task_diameter_milp(t,4,k)
        assert rho<=om+1e-12 and om<=2*rho+1e-12
def test_regret_bound():
    t=hypergraph_demo(4);model=DecisionModel(4)
    for k in range(1,5):
        C=compatible_masks_exact(t,4,k);r=robust_regret_bound(t,C,model)
        assert r['rho_holds'] and r['omega_holds']
def test_full_order_singleton():
    for t in family_targets_for_test(): assert len(compatible_masks_exact(t,4,4))==1
def family_targets_for_test(): return [parity(4,0),threshold(4,2),sparse_xor(4,2),onehot(4),hypergraph_demo(4)]

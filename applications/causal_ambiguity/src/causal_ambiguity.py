from __future__ import annotations
from itertools import combinations
from typing import Dict, List, Sequence, Tuple
import numpy as np

Graph = Tuple[int, ...]

def edge_pairs(d: int) -> List[Tuple[int,int]]:
    return [(i,j) for i in range(d) for j in range(i+1,d)]

def graph_from_bits(d: int, bits: int) -> Graph:
    return tuple((bits >> r) & 1 for r,_ in enumerate(edge_pairs(d)))

def all_ordered_dags(d: int) -> List[Graph]:
    return [graph_from_bits(d,b) for b in range(2**len(edge_pairs(d)))]

def graph_edges(g: Graph, d: int) -> List[Tuple[int,int]]:
    return [e for bit,e in zip(g, edge_pairs(d)) if bit]

def reachability(g: Graph, d: int) -> np.ndarray:
    A=np.zeros((d,d),dtype=np.int8)
    for i,j in graph_edges(g,d): A[i,j]=1
    R=A.copy()
    for k in range(d): R=np.maximum(R,R[:,k:k+1]*R[k:k+1,:])
    return R

def structural_distance(g1: Graph, g2: Graph) -> float:
    return 0.0 if not g1 else sum(a!=b for a,b in zip(g1,g2))/len(g1)

def intervention_signature(g: Graph, d: int) -> Tuple[int,...]:
    R=reachability(g,d)
    return tuple(int(R[i,j]) for i in range(d) for j in range(d) if i!=j)

def intervention_distance(g1: Graph,g2: Graph,d:int)->float:
    s1=intervention_signature(g1,d); s2=intervention_signature(g2,d)
    return sum(a!=b for a,b in zip(s1,s2))/len(s1)

def task_value(g: Graph,d:int,source:int=0,target:int|None=None)->float:
    if target is None: target=d-1
    return float(reachability(g,d)[source,target])

def task_distance(g1: Graph,g2: Graph,d:int,source:int=0,target:int|None=None)->float:
    return abs(task_value(g1,d,source,target)-task_value(g2,d,source,target))

def observation_signature(g: Graph,d:int,k:int)->Tuple:
    # Exact finite surrogate used by this benchmark. Order 1 records all adjacent edge bits;
    # higher orders add length-r directed path existence. This creates nested causal evidence.
    R=reachability(g,d)
    if k<=1:
        return tuple(g)
    sig=[tuple(g)]
    edges=set(graph_edges(g,d))
    for r in range(2,k+1):
        vals=[]
        for S in combinations(range(d),r+1):
            path=all((S[t],S[t+1]) in edges for t in range(r))
            vals.append((S,int(path)))
        sig.append(tuple(vals))
    return tuple(sig)

def compatibility_classes(graphs:Sequence[Graph],d:int,k:int)->Dict[Tuple,List[Graph]]:
    out={}
    for g in graphs: out.setdefault(observation_signature(g,d,k),[]).append(g)
    return out

def diameter(cls:Sequence[Graph], metric)->float:
    if len(cls)<2:return 0.0
    return max(metric(a,b) for i,a in enumerate(cls) for b in cls[i+1:])

def class_metrics(cls:Sequence[Graph],d:int)->Dict[str,float]:
    return {'class_size':len(cls),'omega_struct':diameter(cls,structural_distance),'omega_interv':diameter(cls,lambda a,b:intervention_distance(a,b,d)),'omega_task':diameter(cls,lambda a,b:task_distance(a,b,d))}

def synthetic_incomparability_examples():
    # Constructed finite compatibility classes proving cardinality and consequential diameter are incomparable.
    # Class A: four models tightly clustered in task values; Class B: two models maximally separated.
    A=[0.00,0.10,0.15,0.20]; B=[0.00,1.00]
    # Reverse direction: more models can also have larger diameter.
    C=[0.00,0.20]; D=[0.00,0.25,0.50,0.75,1.00]
    return {'A_size':len(A),'A_omega':max(A)-min(A),'B_size':len(B),'B_omega':max(B)-min(B),
            'C_size':len(C),'C_omega':max(C)-min(C),'D_size':len(D),'D_omega':max(D)-min(D)}

def intervention_counterexample():
    # Deliberate decision-design counterexample. I_A removes more models but leaves wide task range;
    # I_B removes fewer models but sharply reduces consequential ambiguity.
    return {'initial_size':1000,'initial_omega':1.0,
            'IA_size':50,'IA_omega':0.85,'IB_size':500,'IB_omega':0.08}

def causal_scm_incomparability_realization():
    """Concrete finite SCM realization used by the manuscript incomparability theorem.

    Observational regime: context Z is observed, treatment X is fixed to 0, and
    Y = X * 1[U <= p], U~Unif(0,1). Thus models sharing Z are observationally
    indistinguishable, while P(Y=1 | do(X=1)) = p. The task metric is |p-p'|.
    """
    classes = {
        'A': [0.00, 0.10, 0.15, 0.20],
        'B': [0.00, 1.00],
        'C': [0.00, 0.20],
        'D': [0.00, 0.25, 0.50, 0.75, 1.00],
    }
    out = {}
    for z, ps in classes.items():
        out[z] = {
            'context': z,
            'observational_signature': f'Z={z};X=0;Y=0',
            'p_values': ps,
            'cardinality': len(ps),
            'task_diameter': max(ps)-min(ps),
        }
    return out

def exact_maximin_regret_bound(utilities, true_model, compatible_models, distance, L=1.0):
    """Audit the exact theorem: regret <= L * radius <= L * diameter."""
    if isinstance(utilities, dict):
        actions = list(utilities.keys())
        U = lambda a,m: utilities[a][m]
    else:
        actions = list(utilities.actions)
        U = utilities
    a_star = max(actions, key=lambda a: U(a,true_model))
    a_hat = max(actions, key=lambda a: min(U(a,m) for m in compatible_models))
    regret = U(a_star,true_model)-U(a_hat,true_model)
    radius = max(distance(true_model,m) for m in compatible_models)
    diameter = max(distance(m,n) for m in compatible_models for n in compatible_models)
    return {'a_star':a_star,'a_hat':a_hat,'regret':regret,'radius':radius,'diameter':diameter,
            'radius_bound':L*radius,'diameter_bound':L*diameter,
            'verified': regret <= L*radius + 1e-12 and radius <= diameter + 1e-12}

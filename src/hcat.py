from __future__ import annotations
from dataclasses import dataclass
from itertools import product, combinations
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Sequence, Tuple
import numpy as np

State=Tuple[int,...]

@lru_cache(maxsize=None)
def states(d:int)->Tuple[State,...]:
    return tuple(product((0,1),repeat=d))

@lru_cache(maxsize=None)
def subsets_upto(d:int,k:int)->Tuple[Tuple[int,...],...]:
    return tuple(S for r in range(1,k+1) for S in combinations(range(d),r))

def relation_mask(d:int,predicate:Callable[[State],bool])->int:
    m=0
    for i,x in enumerate(states(d)):
        if predicate(x): m|=(1<<i)
    return m

def mask_states(mask:int,d:int)->Tuple[State,...]:
    return tuple(x for i,x in enumerate(states(d)) if (mask>>i)&1)

@lru_cache(maxsize=None)
def preimage_masks(d:int,S:Tuple[int,...])->Dict[Tuple[int,...],int]:
    out:Dict[Tuple[int,...],int]={}
    for i,x in enumerate(states(d)):
        y=tuple(x[j] for j in S)
        out[y]=out.get(y,0)|(1<<i)
    return out

def projection_values(mask:int,d:int,S:Tuple[int,...]):
    return frozenset(y for y,pm in preimage_masks(d,S).items() if mask & pm)

def observation_signature(mask:int,d:int,k:int):
    return tuple((S,tuple(sorted(projection_values(mask,d,S)))) for S in subsets_upto(d,k))

def compatible_masks_exact(target:int,d:int,k:int,include_empty:bool=False)->np.ndarray:
    if d>4:
        raise ValueError('Exact global-relation enumeration is intentionally limited to d<=4.')
    nstates=2**d
    start=0 if include_empty else 1
    arr=np.arange(start,1<<nstates,dtype=np.uint32)
    ok=np.ones(arr.shape[0],dtype=bool)
    for S in subsets_upto(d,k):
        target_vals=projection_values(target,d,S)
        for y,pm in preimage_masks(d,S).items():
            has=(arr & np.uint32(pm))!=0
            ok &= has if y in target_vals else ~has
    return arr[ok]

def projection_disagreement(a:int,b:int,d:int,k:int)->float:
    subs=subsets_upto(d,k)
    return sum(projection_values(a,d,S)!=projection_values(b,d,S) for S in subs)/len(subs)

def join_reconstruction(target:int,d:int,k:int)->int:
    m=0
    for i,x in enumerate(states(d)):
        good=True
        for S in subsets_upto(d,k):
            if tuple(x[j] for j in S) not in projection_values(target,d,S):
                good=False; break
        if good:m|=1<<i
    return m

def normalized_symdiff(a:int,b:int,d:int)->float:
    return int(a^b).bit_count()/(2**d)

def task_radius_exact(target:int,C:Sequence[int],d:int)->float:
    if len(C)==0:return 0.0
    vals=np.asarray(C,dtype=np.uint32)
    pc=_popcount_table(2**d)
    return float(pc[np.bitwise_xor(vals,np.uint32(target))].max())/(2**d)

@lru_cache(maxsize=None)
def _popcount_table(nbits:int):
    n=1<<nbits
    return np.fromiter((i.bit_count() for i in range(n)),dtype=np.uint8,count=n)

def task_diameter_exact(C:Sequence[int],d:int)->float:
    # Diameter via XOR shifts on the Boolean indicator of C. Exact and fast for d<=4.
    nbits=2**d
    n=1<<nbits
    present=np.zeros(n,dtype=bool)
    vals=np.asarray(C,dtype=np.uint32)
    present[vals]=True
    pc=_popcount_table(nbits)
    order=np.argsort(pc)[::-1]
    idx=np.arange(n,dtype=np.uint32)
    for z in order:
        z=int(z)
        if np.any(present & present[idx ^ np.uint32(z)]):
            return int(pc[z])/nbits
    return 0.0

def threshold_rank(class_sizes:Dict[int,int],tau_class:int=1):
    for k in sorted(class_sizes):
        if class_sizes[k]<=tau_class:return k
    return None

@dataclass(frozen=True)
class DecisionModel:
    d:int
    reward:float=10.0
    costs:Tuple[float,...]|None=None
    def __post_init__(self):
        if self.costs is None:
            vals=[]
            for x in states(self.d):
                s=sum(x)
                vals.append(0.6+0.55*s+0.10*s*s+0.05*sum((i+1)*v for i,v in enumerate(x)))
            object.__setattr__(self,'costs',tuple(vals))
    def utility(self,mask:int,state_index:int)->float:
        return (self.reward if (mask>>state_index)&1 else 0.0)-self.costs[state_index]
    def optimal_action(self,mask:int)->int:
        return max(range(2**self.d),key=lambda i:self.utility(mask,i))
    def regret(self,mask:int,action:int)->float:
        a0=self.optimal_action(mask)
        return max(0.0,self.utility(mask,a0)-self.utility(mask,action))
    def robust_action(self,C:Sequence[int])->int:
        # Exact maximin over the compatible relation class.
        vals=np.asarray(C,dtype=np.uint32)
        best=None;bestv=-1e99
        for i in range(2**self.d):
            guaranteed=bool(np.all((vals & np.uint32(1<<i))!=0))
            worst=(self.reward if guaranteed else 0.0)-self.costs[i]
            if worst>bestv:bestv=worst;best=i
        return int(best)

    def worst_case_utility(self,C:Sequence[int],action:int)->float:
        vals=np.asarray(C,dtype=np.uint32)
        guaranteed=bool(np.all((vals & np.uint32(1<<action))!=0))
        return (self.reward if guaranteed else 0.0)-self.costs[action]
    def support_fraction(self,C:Sequence[int],action:int)->float:
        vals=np.asarray(C,dtype=np.uint32)
        return float(np.mean((vals & np.uint32(1<<action))!=0)) if len(vals) else 0.0
    def join_action(self,target:int,k:int)->int:
        j=join_reconstruction(target,self.d,k)
        ids=[i for i in range(2**self.d) if (j>>i)&1]
        if not ids:return min(range(2**self.d),key=lambda i:self.costs[i])
        return min(ids,key=lambda i:self.costs[i])

def utility_distance(a:int,b:int,model:DecisionModel)->float:
    # costs are common across envelopes, so membership changes produce reward-scale difference.
    return max(abs(model.utility(a,i)-model.utility(b,i)) for i in range(2**model.d))/model.reward

def utility_radius(target:int,C:Sequence[int],model:DecisionModel)->float:
    # With shared action costs and binary success reward, any distinct relation differs by exactly one reward unit in d_U.
    if len(C)<=1: return 0.0
    return 1.0 if any(int(f)!=int(target) for f in C) else 0.0

def utility_diameter(C:Sequence[int],model:DecisionModel)->float:
    # With common state costs and binary success reward, any distinct relations have normalized d_U=1.
    return 0.0 if len(C)<=1 else 1.0

def robust_regret_bound(target:int,C:Sequence[int],model:DecisionModel):
    a=model.robust_action(C)
    reg=model.regret(target,a)
    rho=utility_radius(target,C,model)
    omega=utility_diameter(C,model)
    return {'action':a,'regret':reg,'rho_u':rho,'omega_u':omega,
            'rho_bound':model.reward*rho,'omega_bound':model.reward*omega,
            'rho_holds':reg<=model.reward*rho+1e-12,
            'omega_holds':reg<=model.reward*omega+1e-12}

def task_diameter_milp(target:int,d:int,k:int)->float:
    """Exact diameter of the bounded-order compatibility class under normalized symmetric difference.
    Solves a 0-1 MILP over two compatible global relations. Intended for d<=4 experiments.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    n=2**d
    # variables f[0:n], g[n:2n], y[2n:3n], maximize sum y -> minimize -sum y
    N=3*n
    c=np.zeros(N); c[2*n:]=-1.0
    integrality=np.ones(N,dtype=int)
    lb=np.zeros(N); ub=np.ones(N)
    rows=[]; lows=[]; highs=[]
    # Compatibility constraints for f and g.
    for S in subsets_upto(d,k):
        vals=projection_values(target,d,S)
        for yv,pm in preimage_masks(d,S).items():
            ids=[i for i in range(n) if (pm>>i)&1]
            if yv in vals:
                for off in (0,n):
                    row={off+i:1.0 for i in ids}; rows.append(row); lows.append(1.0); highs.append(np.inf)
            else:
                for off in (0,n):
                    for i in ids:
                        rows.append({off+i:1.0}); lows.append(0.0); highs.append(0.0)
    # XOR linearization y_i = |f_i-g_i|
    for i in range(n):
        # y >= f-g  -> -f + g + y >=0
        rows.append({i:-1.0,n+i:1.0,2*n+i:1.0}); lows.append(0.0); highs.append(np.inf)
        # y >= g-f
        rows.append({i:1.0,n+i:-1.0,2*n+i:1.0}); lows.append(0.0); highs.append(np.inf)
        # y <= f+g -> -f-g+y <=0
        rows.append({i:-1.0,n+i:-1.0,2*n+i:1.0}); lows.append(-np.inf); highs.append(0.0)
        # y <= 2-f-g -> f+g+y <=2
        rows.append({i:1.0,n+i:1.0,2*n+i:1.0}); lows.append(-np.inf); highs.append(2.0)
    A=lil_matrix((len(rows),N),dtype=float)
    for r,row in enumerate(rows):
        for j,v in row.items(): A[r,j]=v
    res=milp(c,integrality=integrality,bounds=Bounds(lb,ub),constraints=LinearConstraint(A.tocsr(),np.array(lows),np.array(highs)),
             options={'time_limit':20.0,'presolve':True})
    if not res.success:
        raise RuntimeError(f'MILP diameter failed: {res.message}')
    return float(round(-res.fun))/(2**d)

def parity(d:int,p:int=0)->int:
    return relation_mask(d,lambda x:sum(x)%2==p)
def threshold(d:int,q:int)->int:
    return relation_mask(d,lambda x:sum(x)>=q)
def sparse_xor(d:int,order:int)->int:
    order=min(order,d-1)
    return relation_mask(d,lambda x:x[-1]==(sum(x[:order])%2))
def onehot(d:int)->int:
    return relation_mask(d,lambda x:sum(x)==1)
def hypergraph_demo(d:int=4)->int:
    if d!=4: raise ValueError('demo currently uses d=4')
    return relation_mask(4,lambda x:(x[0]^x[1]^x[2])==0 and (x[1]^x[3])==1)

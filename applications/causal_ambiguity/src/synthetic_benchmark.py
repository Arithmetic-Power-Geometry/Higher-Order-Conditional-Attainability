from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]
RES=OUT/'results'; FIG=OUT/'figures'; REP=OUT/'reports'
for p in (RES,FIG,REP): p.mkdir(parents=True,exist_ok=True)

POLICIES=['OICA','ClassSize','Graph','Random']
MECHS=['linear_gaussian','nonlinear_additive','binary_logistic']
DIMS=[5,10,20]
NS=[250,500,1000,5000]
SEEDS=list(range(12))
EPS=0.30
MAX_STEPS=10

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

def generate_case(d, mech, n, seed):
    rng=np.random.default_rng(seed + 1000*d + 37*n + 131*MECHS.index(mech))
    m=36 if d<=10 else 28
    true_u=rng.uniform(.15,.95,size=4)
    true_best=int(np.argmax(true_u))
    candidates=[]
    for j in range(m):
        complexity=max(0.02, 0.52*(250/n)**0.35*(5/d)**0.08)
        noise=rng.normal(0,complexity,size=4)
        if mech=='nonlinear_additive': noise += 0.12*np.sin((j+1)*np.arange(1,5))
        elif mech=='binary_logistic': noise = 0.85*noise + rng.normal(0,0.05,size=4)
        u=np.clip(true_u+noise,0,1)
        graph_error=float(np.clip(rng.beta(2.0,5.0)+0.25*np.mean(np.abs(noise)),0,1))
        candidates.append({'u':u,'g':graph_error})
    candidates.append({'u':true_u.copy(),'g':0.0})
    task_axis=np.vstack([np.eye(4),rng.normal(size=(2,4))]); task_axis[4:]/=np.linalg.norm(task_axis[4:],axis=1,keepdims=True)
    graph_axis=rng.uniform(.4,1.0,size=6)
    cost=rng.uniform(.8,1.6,size=6)
    return rng,true_u,true_best,candidates,task_axis,graph_axis,cost

def model_stats(C):
    U=np.array([c['u'] for c in C]); G=np.array([c['g'] for c in C])
    omega=float(np.max(U.max(0)-U.min(0))) if len(C)>1 else 0.0
    gdiam=float(G.max()-G.min()) if len(C)>1 else 0.0
    return omega,gdiam,len(C)

def update_class(C, true_u, task_axis, graph_axis, intervention, rng, nstep):
    if len(C)<=1: return C
    axis=task_axis[intervention]
    true_task=float(np.dot(true_u,axis)); true_graph=0.0
    sigma=max(0.025,0.18/math.sqrt(nstep+1))
    observed_task=true_task+rng.normal(0,sigma); observed_graph=true_graph+rng.normal(0,sigma)
    scores=[]
    for c in C:
        s_task=abs(float(np.dot(c['u'],axis))-observed_task)
        s_graph=abs(c['g']*graph_axis[intervention]-observed_graph)
        scores.append(0.68*s_task+0.32*s_graph)
    scores=np.array(scores)
    q=min(0.88,0.62+0.045*nstep)
    thresh=max(np.quantile(scores,q), 1.35*sigma)
    new=[c for c,s in zip(C,scores) if s<=thresh]
    if not new: new=[C[int(np.argmin(scores))]]
    return new

def predicted_post(C,true_u,task_axis,graph_axis,i,criterion):
    if len(C)<=1:return 0.0
    axis=task_axis[i]
    U=np.array([c['u'] for c in C]); proj=U@axis
    med=np.median(proj)
    halves=[[c for c,p in zip(C,proj) if p<=med],[c for c,p in zip(C,proj) if p>med]]
    vals=[]
    for H in halves:
        if not H: continue
        om,gd,sz=model_stats(H)
        if criterion=='omega': vals.append(om)
        elif criterion=='size': vals.append(sz)
        else: vals.append(gd/max(0.1,graph_axis[i]))
    return max(vals) if vals else 0.0

def run_policy(policy, d, mech, n, seed):
    rng,true_u,true_best,C,task_axis,graph_axis,cost=generate_case(d,mech,n,seed)
    rng=np.random.default_rng(seed*100003 + d*997 + n + POLICIES.index(policy)*7919 + MECHS.index(mech)*131)
    cumcost=0.0; reached=None; rows=[]
    for t in range(MAX_STEPS+1):
        omega,gdiam,sz=model_stats(C)
        U=np.array([c['u'] for c in C])
        robust=int(np.argmax(U.min(axis=0)))
        regret=float(true_u[true_best]-true_u[robust])
        rows.append(dict(policy=policy,d=d,mechanism=mech,n=n,seed=seed,step=t,omega_task=omega,graph_diameter=gdiam,class_size=sz,cumulative_cost=cumcost,regret=regret,reached_eps=int(omega<=EPS)))
        if omega<=EPS and reached is None: reached=t
        if t==MAX_STEPS: break
        if policy=='OICA':
            i=int(np.argmin([predicted_post(C,true_u,task_axis,graph_axis,j,'omega') for j in range(6)]))
        elif policy=='ClassSize':
            i=int(np.argmin([predicted_post(C,true_u,task_axis,graph_axis,j,'size') for j in range(6)]))
        elif policy=='Graph':
            i=int(np.argmin([predicted_post(C,true_u,task_axis,graph_axis,j,'graph') for j in range(6)]))
        else:
            i=int(rng.integers(0,6))
        C=update_class(C,true_u,task_axis,graph_axis,i,rng,t+1)
        cumcost+=float(cost[i])
    return rows, (reached if reached is not None else MAX_STEPS+1)

def main():
    allrows=[]; stoprows=[]
    for d in DIMS:
      for mech in MECHS:
        for n in NS:
          for seed in SEEDS:
            for p in POLICIES:
              rr,stop=run_policy(p,d,mech,n,seed); allrows.extend(rr)
              last=rr[-1]
              stoprows.append(dict(policy=p,d=d,mechanism=mech,n=n,seed=seed,tau_eps=stop,final_omega=last['omega_task'],final_regret=last['regret'],final_class_size=last['class_size'],final_graph_diameter=last['graph_diameter'],final_cost=last['cumulative_cost'],success=int(stop<=MAX_STEPS)))
    traj=pd.DataFrame(allrows); stops=pd.DataFrame(stoprows)
    traj.to_csv(RES/'synthetic_benchmark_trajectories.csv',index=False)
    stops.to_csv(RES/'synthetic_benchmark_endpoints.csv',index=False)
    summary=stops.groupby('policy').agg(instances=('seed','size'),success_rate=('success','mean'),mean_tau=('tau_eps','mean'),median_tau=('tau_eps','median'),mean_final_omega=('final_omega','mean'),mean_regret=('final_regret','mean'),mean_cost=('final_cost','mean'),mean_graph_diameter=('final_graph_diameter','mean'),mean_class_size=('final_class_size','mean')).reset_index()
    summary.to_csv(RES/'synthetic_benchmark_summary.csv',index=False)
    wide=stops.pivot_table(index=['d','mechanism','n','seed'],columns='policy',values=['tau_eps','final_omega','final_regret','final_cost'])
    rng=np.random.default_rng(20260905); pairs=[]
    for metric in ['tau_eps','final_omega','final_regret','final_cost']:
        diff=(wide[(metric,'OICA')]-wide[(metric,'Graph')]).to_numpy()
        boots=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(4000)])
        pairs.append({'metric':metric,'mean_OICA_minus_Graph':diff.mean(),'ci95_low':np.quantile(boots,.025),'ci95_high':np.quantile(boots,.975),'oica_better_fraction':float(np.mean(diff<0))})
    pd.DataFrame(pairs).to_csv(RES/'paired_oica_vs_graph.csv',index=False)
    disagreements=[]
    for d in DIMS:
      for mech in MECHS:
       for n in NS:
        for seed in SEEDS:
          _,tu,_,C,ta,ga,_=generate_case(d,mech,n,seed)
          io=int(np.argmin([predicted_post(C,tu,ta,ga,j,'omega') for j in range(6)]))
          ig=int(np.argmin([predicted_post(C,tu,ta,ga,j,'graph') for j in range(6)]))
          disagreements.append(io!=ig)
    disagreement=float(np.mean(disagreements))
    meantraj=traj.groupby(['policy','step'])[['omega_task','regret','class_size','graph_diameter']].mean().reset_index()
    for y,name,ylabel in [('omega_task','fig5_task_ambiguity','Mean residual task ambiguity'),('regret','fig6_regret','Mean decision regret')]:
        plt.figure(figsize=(6.4,4.2))
        for p in POLICIES:
            q=meantraj[meantraj.policy==p]; plt.plot(q.step,q[y],marker='o',label=p)
        plt.xlabel('Intervention step'); plt.ylabel(ylabel); plt.legend(); plt.tight_layout(); plt.savefig(FIG/f'{name}.pdf'); plt.close()
    base=traj[traj.step==0][['policy','d','mechanism','n','seed','omega_task','class_size']].rename(columns={'omega_task':'omega0','class_size':'size0'})
    end=traj[traj.step==MAX_STEPS].merge(base,on=['policy','d','mechanism','n','seed'])
    plt.figure(figsize=(6.4,4.2)); plt.scatter((end.size0-end.class_size)/end.size0,(end.omega0-end.omega_task),s=11,alpha=.35); plt.xlabel('Compatible-class reduction fraction'); plt.ylabel('Task-ambiguity reduction'); plt.tight_layout(); plt.savefig(FIG/'fig7_class_vs_task_reduction.pdf'); plt.close()
    result_summary={'benchmark_type':'controlled synthetic finite-sample SCM proxy benchmark','external_csuite_executed':False,'instances_per_policy':int(len(stops)/4),'total_policy_runs':int(len(stops)),'trajectory_rows':int(len(traj)),'epsilon':EPS,'max_steps':MAX_STEPS,'first_step_oica_graph_disagreement_rate':disagreement,'policy_summary':summary.set_index('policy').round(6).to_dict(orient='index'),'paired_oica_vs_graph':pairs,'falsification_note':'This controlled synthetic benchmark is empirical within the generated finite-sample model classes, but it is not a substitute for external CSuite validation.'}
    (REP/'BENCHMARK_SUMMARY.json').write_text(json.dumps(result_summary,indent=2))
    (REP/'BENCHMARK_PROTOCOL.md').write_text('# Synthetic benchmark protocol\n\nThis benchmark compares OICA task-ambiguity selection, compatible-class-size reduction, a graph-oriented residual-diameter proxy, and random intervention under matched generated cases. It spans d={5,10,20}, three mechanism families, n={250,500,1000,5000}, 12 seeds, and up to ten intervention steps. The compatibility class is updated using finite-sample noisy intervention evidence and an adaptive tolerance. Primary outcomes are task ambiguity, decision-sufficiency stopping time, intervention cost, regret, graph-diameter proxy, and compatible-class size. Paired bootstrap confidence intervals use 4,000 resamples.\n\nImportant scope boundary: CSuite was not executed in this run; no external-benchmark superiority claim is made.\n')
    print(summary.to_string(index=False)); print(pd.DataFrame(pairs).to_string(index=False)); print('disagreement',disagreement)

if __name__ == '__main__':
    main()

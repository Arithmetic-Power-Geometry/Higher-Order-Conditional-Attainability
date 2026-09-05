from __future__ import annotations
from pathlib import Path
import json, math, random, time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.datasets import load_breast_cancer
from .hcat import *


def family_targets(d=4):
    return {'parity_even':parity(d,0),'threshold_ge2':threshold(d,2),
            'sparse_xor_order2':sparse_xor(d,2),'onehot':onehot(d),
            'hypergraph_constraints':hypergraph_demo(d)}

def exact_family_audit(d=4):
    rows=[];model=DecisionModel(d)
    for name,target in family_targets(d).items():
        class_sizes={}
        for k in range(1,d+1):
            C=compatible_masks_exact(target,d,k)
            class_sizes[k]=len(C)
            rr=robust_regret_bound(target,C,model)
            ja=model.join_action(target,k)
            rows.append({'family':name,'d':d,'k':k,'target_states':target.bit_count(),
                'compatible_global_relations':len(C),
                'join_states':join_reconstruction(target,d,k).bit_count(),
                'join_reconstruction_error':normalized_symdiff(target,join_reconstruction(target,d,k),d),
                'task_radius':task_radius_exact(target,C,d),'task_opacity':task_diameter_milp(target,d,k),
                'utility_radius':rr['rho_u'],'utility_opacity':rr['omega_u'],
                'join_regret':model.regret(target,ja),'hcat_robust_regret':rr['regret'],
                'regret_bound_rho':rr['rho_bound'],'regret_bound_omega':rr['omega_bound'],
                'rho_bound_holds':rr['rho_holds'],'omega_bound_holds':rr['omega_holds']})
        rank=threshold_rank(class_sizes)
        for r in rows[-d:]:r['exact_attainability_rank']=rank
    return pd.DataFrame(rows)

def random_relation_decision_benchmark(d=4,n=250,seed=20260905):
    rng=np.random.default_rng(seed);model=DecisionModel(d)
    rows=[]
    # nontrivial random relations; density spread prevents one degenerate regime
    targets=[]
    while len(targets)<n:
        p=float(rng.uniform(.18,.82))
        bits=rng.random(2**d)<p
        m=sum((1<<i) for i,b in enumerate(bits) if b)
        if m and m!=(1<<(2**d))-1: targets.append(int(m))
    for j,target in enumerate(targets):
        for k in (1,2,3):
            C=compatible_masks_exact(target,d,k)
            rr=robust_regret_bound(target,C,model)
            ja=model.join_action(target,k); ha=rr['action']
            rows.append({'rep':j,'k':k,'density':target.bit_count()/(2**d),
                         'class_size':len(C),'join_regret':model.regret(target,ja),
                         'hcat_regret':rr['regret'],'rho_u':rr['rho_u'],'omega_u':rr['omega_u'],
                         'join_worst_utility':model.worst_case_utility(C,ja),
                         'hcat_worst_utility':model.worst_case_utility(C,ha),
                         'join_support_fraction':model.support_fraction(C,ja),
                         'hcat_support_fraction':model.support_fraction(C,ha),
                         'bound_holds':rr['rho_holds'] and rr['omega_holds']})
    return pd.DataFrame(rows)

def paired_stats(df):
    out=[]
    rng=np.random.default_rng(7)
    for k,g in df.groupby('k'):
        a=g.join_regret.to_numpy(); b=g.hcat_regret.to_numpy(); diff=a-b
        try:
            w=wilcoxon(a,b,alternative='greater',zero_method='zsplit')
            stat=float(w.statistic);p=float(w.pvalue)
        except Exception:
            stat=float('nan');p=float('nan')
        boots=[]
        for _ in range(4000):
            idx=rng.integers(0,len(diff),len(diff));boots.append(float(np.mean(diff[idx])))
        lo,hi=np.quantile(boots,[.025,.975])
        gu=(g.hcat_worst_utility-g.join_worst_utility).to_numpy()
        try:
            wg=wilcoxon(g.hcat_worst_utility,g.join_worst_utility,alternative='greater',zero_method='zsplit')
            gp=float(wg.pvalue)
        except Exception: gp=float('nan')
        out.append({'k':k,'n':len(g),'join_regret_mean':float(a.mean()),'hcat_regret_mean':float(b.mean()),
                    'mean_regret_reduction':float(diff.mean()),'reduction_ci95_low':float(lo),
                    'reduction_ci95_high':float(hi),'wilcoxon_regret_statistic':stat,'wilcoxon_regret_one_sided_p':p,
                    'join_worst_utility_mean':float(g.join_worst_utility.mean()),
                    'hcat_worst_utility_mean':float(g.hcat_worst_utility.mean()),
                    'mean_guaranteed_utility_gain':float(gu.mean()),
                    'wilcoxon_guarantee_one_sided_p':gp,
                    'join_support_fraction_mean':float(g.join_support_fraction.mean()),
                    'hcat_support_fraction_mean':float(g.hcat_support_fraction.mean()),
                    'bound_pass_rate':float(g.bound_holds.mean())})
    return pd.DataFrame(out)

def real_wdbc_audit():
    data=load_breast_cancer();X=np.asarray(data.data,float);y=np.asarray(data.target,int)
    corrs=np.array([abs(np.corrcoef(X[:,j],y)[0,1]) for j in range(X.shape[1])])
    idx=np.argsort(np.nan_to_num(corrs))[-3:][::-1]
    med=np.median(X[:,idx],axis=0);B=(X[:,idx]>med).astype(int)
    masks=[]
    m=0
    for i in range(len(y)):
        s=tuple(list(B[i])+[int(y[i])]);si=states(4).index(s);m|=1<<si
    model=DecisionModel(4);rows=[];sizes={}
    for k in range(1,5):
        C=compatible_masks_exact(m,4,k);sizes[k]=len(C);rr=robust_regret_bound(m,C,model)
        j=join_reconstruction(m,4,k)
        rows.append({'dataset':'Wisconsin Diagnostic Breast Cancer','n_samples':len(y),
                     'selected_features':' | '.join(data.feature_names[j] for j in idx),
                     'k':k,'observed_global_states':m.bit_count(),'compatible_global_relations':len(C),
                     'join_states':j.bit_count(),'join_error':normalized_symdiff(m,j,4),
                     'task_radius':task_radius_exact(m,C,4),'task_opacity':task_diameter_milp(m,4,k),
                     'utility_opacity':rr['omega_u']})
    rank=threshold_rank(sizes)
    for r in rows:r['exact_attainability_rank']=rank
    return pd.DataFrame(rows),idx,m

def real_wdbc_bootstrap(reps=200,seed=2026):
    data=load_breast_cancer();X=np.asarray(data.data,float);y=np.asarray(data.target,int)
    corrs=np.array([abs(np.corrcoef(X[:,j],y)[0,1]) for j in range(X.shape[1])])
    idx=np.argsort(np.nan_to_num(corrs))[-3:][::-1];med=np.median(X[:,idx],axis=0);B=(X[:,idx]>med).astype(int)
    rng=np.random.default_rng(seed);rows=[]
    for r in range(reps):
        samp=rng.integers(0,len(y),len(y));m=0
        for i in samp:
            s=tuple(list(B[i])+[int(y[i])]);m|=1<<states(4).index(s)
        sizes={}
        for k in range(1,5):
            C=compatible_masks_exact(m,4,k);sizes[k]=len(C)
        rank=threshold_rank(sizes)
        rows.append({'rep':r,'observed_states':m.bit_count(),'rank':rank,
                     **{f'class_size_k{k}':sizes[k] for k in range(1,5)}})
    return pd.DataFrame(rows)

def parity_scaling(max_d=10):
    rows=[]
    for d in range(2,max_d+1):
        for k in range(1,d+1):
            rows.append({'d':d,'k':k,'proper_order':k<d,
                         'projection_disagreement_even_vs_odd':0.0 if k<d else 1.0/(2**d-1),
                         'exact_task_opacity_witness':1.0 if k<d else 0.0,
                         'attainability_rank':d})
    return pd.DataFrame(rows)

def generate_figures(results:Path,figures:Path):
    import matplotlib.pyplot as plt
    figures.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(results/'exact_family_audit.csv')
    for metric,name,ylabel in [('compatible_global_relations','fig1_class_size','Compatible global relations'),
                               ('task_opacity','fig2_opacity','Exact task opacity'),
                               ('join_reconstruction_error','fig3_join_error','Join reconstruction error')]:
        fig,ax=plt.subplots(figsize=(7.2,4.5))
        for fam,g in df.groupby('family'):
            ax.plot(g.k,g[metric],marker='o',label=fam.replace('_',' '))
        if metric=='compatible_global_relations': ax.set_yscale('log')
        ax.set_xlabel('Observation order k');ax.set_ylabel(ylabel);ax.set_xticks([1,2,3,4]);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(figures/f'{name}.pdf');fig.savefig(figures/f'{name}.png',dpi=220);plt.close(fig)
    s=pd.read_csv(results/'random_decision_stats.csv')
    fig,ax=plt.subplots(figsize=(6.8,4.2));x=np.arange(len(s));w=.34
    ax.bar(x-w/2,s.join_regret_mean,w,label='Join baseline');ax.bar(x+w/2,s.hcat_regret_mean,w,label='HCAT robust')
    ax.set_xticks(x);ax.set_xticklabels([f'k={int(k)}' for k in s.k]);ax.set_ylabel('Mean decision regret');ax.legend();fig.tight_layout()
    fig.savefig(figures/'fig4_decision_regret.pdf');fig.savefig(figures/'fig4_decision_regret.png',dpi=220);plt.close(fig)
    r=pd.read_csv(results/'real_wdbc_audit.csv')
    fig,ax=plt.subplots(figsize=(6.8,4.2));ax.plot(r.k,r.compatible_global_relations,marker='o',label='Compatible relations');ax.set_yscale('log')
    ax.set_xlabel('Observation order k');ax.set_ylabel('Compatible global relations (log scale)');ax.set_xticks([1,2,3,4]);fig.tight_layout()
    fig.savefig(figures/'fig5_real_class_collapse.pdf');fig.savefig(figures/'fig5_real_class_collapse.png',dpi=220);plt.close(fig)
    b=pd.read_csv(results/'real_wdbc_bootstrap.csv')
    fig,ax=plt.subplots(figsize=(6.8,4.2));counts=b['rank'].value_counts().sort_index();ax.bar(counts.index.astype(str),counts.values)
    ax.set_xlabel('Exact attainability rank');ax.set_ylabel('Bootstrap frequency');fig.tight_layout();fig.savefig(figures/'fig6_bootstrap_rank.pdf');fig.savefig(figures/'fig6_bootstrap_rank.png',dpi=220);plt.close(fig)

def run_all(root:Path):
    results=root/'results';figs=root/'figures';results.mkdir(exist_ok=True);figs.mkdir(exist_ok=True)
    exact=exact_family_audit();exact.to_csv(results/'exact_family_audit.csv',index=False)
    rnd=random_relation_decision_benchmark();rnd.to_csv(results/'random_decision_benchmark.csv',index=False)
    stats=paired_stats(rnd);stats.to_csv(results/'random_decision_stats.csv',index=False)
    real,idx,m=real_wdbc_audit();real.to_csv(results/'real_wdbc_audit.csv',index=False)
    boot=real_wdbc_bootstrap();boot.to_csv(results/'real_wdbc_bootstrap.csv',index=False)
    parity=parity_scaling();parity.to_csv(results/'parity_scaling.csv',index=False)
    generate_figures(results,figs)
    manifest={'selected_wdbc_feature_indices':[int(x) for x in idx],'wdbc_relation_mask':int(m),
              'exact_family_rows':len(exact),'random_benchmark_rows':len(rnd),'bootstrap_reps':len(boot)}
    (root/'reports/run_manifest.json').write_text(json.dumps(manifest,indent=2))
    return exact,rnd,stats,real,boot

if __name__=='__main__':
    run_all(Path(__file__).resolve().parents[1])

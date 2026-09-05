from pathlib import Path
import csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from causal_ambiguity import *
ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/'results'; REP=ROOT/'reports'; RES.mkdir(exist_ok=True); REP.mkdir(exist_ok=True)
rows=[]
for d in [3,4,5]:
    gs=all_ordered_dags(d)
    for k in range(1,d+1):
        classes=compatibility_classes(gs,d,k); vals=[class_metrics(c,d) for c in classes.values()]
        rows.append({'d':d,'k':k,'n_graphs':len(gs),'n_classes':len(classes),'max_class_size':max(v['class_size'] for v in vals),'max_omega_struct':max(v['omega_struct'] for v in vals),'max_omega_interv':max(v['omega_interv'] for v in vals),'max_omega_task':max(v['omega_task'] for v in vals)})
with open(RES/'exact_order_audit.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
inc=synthetic_incomparability_examples()
with open(RES/'cardinality_task_incomparability.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=inc.keys());w.writeheader();w.writerow(inc)
ic=intervention_counterexample()
with open(RES/'intervention_counterexample.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=ic.keys());w.writeheader();w.writerow(ic)
scm=causal_scm_incomparability_realization()
with open(RES/'causal_scm_incomparability_realization.csv','w',newline='') as f:
    fields=['class','observational_signature','p_values','cardinality','task_diameter']
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for z,v in scm.items():
        w.writerow({'class':z,'observational_signature':v['observational_signature'],'p_values':'|'.join(map(str,v['p_values'])),'cardinality':v['cardinality'],'task_diameter':v['task_diameter']})
utilities={'safe':{'m0':0.7,'m1':0.7},'risky':{'m0':1.0,'m1':0.0}}
dist=lambda a,b: 0.0 if a==b else 1.0
reg=exact_maximin_regret_bound(utilities,'m0',['m0','m1'],dist,L=1.0)
with open(RES/'exact_regret_certificate.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=reg.keys()); w.writeheader(); w.writerow(reg)
summary={'exact_rows':len(rows),'cardinality_incomparability_verified':inc['A_size']>inc['B_size'] and inc['A_omega']<inc['B_omega'] and inc['D_size']>inc['C_size'] and inc['D_omega']>inc['C_omega'], 'causal_scm_realization_verified':scm['A']['cardinality']>scm['B']['cardinality'] and scm['A']['task_diameter']<scm['B']['task_diameter'] and scm['D']['cardinality']>scm['C']['cardinality'] and scm['D']['task_diameter']>scm['C']['task_diameter'], 'exact_regret_certificate_verified':bool(reg['verified']), 'intervention_counterexample_verified':ic['IA_size']<ic['IB_size'] and ic['IA_omega']>ic['IB_omega']}
(REP/'RESULTS_SUMMARY.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

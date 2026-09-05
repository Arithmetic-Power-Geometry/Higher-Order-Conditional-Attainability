import gradio as gr
from pathlib import Path
import pandas as pd
from src.hcat import *
from src.experiments import family_targets

families=family_targets(4)

def analyze(family,k,reward):
    target=families[family];C=compatible_masks_exact(target,4,int(k));model=DecisionModel(4,float(reward));j=join_reconstruction(target,4,int(k));rr=robust_regret_bound(target,C,model)
    return pd.DataFrame([{'family':family,'k':int(k),'target states':target.bit_count(),'compatible global relations':len(C),
        'join states':j.bit_count(),'join error':normalized_symdiff(target,j,4),'task radius':task_radius_exact(target,C,4),
        'task opacity':task_diameter_exact(C,4),'utility opacity':rr['omega_u'],'join regret':model.regret(target,model.join_action(target,int(k))),
        'HCAT robust regret':rr['regret'],'regret upper bound':rr['rho_bound']}])

def custom(mask_hex,k,reward):
    try:mask=int(str(mask_hex),0)
    except:return pd.DataFrame([{'error':'Enter an integer mask, e.g. 0x6996'}])
    if mask<=0 or mask>=2**16:return pd.DataFrame([{'error':'For d=4 mask must be between 1 and 65535.'}])
    C=compatible_masks_exact(mask,4,int(k));model=DecisionModel(4,float(reward));j=join_reconstruction(mask,4,int(k));rr=robust_regret_bound(mask,C,model)
    return pd.DataFrame([{'mask':hex(mask),'k':int(k),'states':mask.bit_count(),'class size':len(C),'join states':j.bit_count(),
                         'join error':normalized_symdiff(mask,j,4),'task radius':task_radius_exact(mask,C,4),'task opacity':task_diameter_exact(C,4),
                         'robust regret':rr['regret'],'bound':rr['rho_bound']}])

def load_csv(name):
    p=Path(__file__).parent/'results'/name
    return pd.read_csv(p) if p.exists() else pd.DataFrame([{'status':'Run python reproduce.py first.'}])

with gr.Blocks(title='HCAT Explorer') as demo:
    gr.Markdown('# Higher-Order Conditional Attainability (HCAT) Explorer\nExact d=4 global-relation audit, decision layer, real-data results, and reproducibility viewer.')
    with gr.Tab('Exact family analyzer'):
        f=gr.Dropdown(list(families),value='parity_even',label='Target family');k=gr.Slider(1,4,1,step=1,label='Observation order k');r=gr.Number(10,label='Success reward');o=gr.Dataframe()
        gr.Button('Analyze').click(analyze,[f,k,r],o)
    with gr.Tab('Custom relation mask'):
        m=gr.Textbox('0x6996',label='16-bit relation mask');k2=gr.Slider(1,4,2,step=1);r2=gr.Number(10);o2=gr.Dataframe();gr.Button('Analyze custom relation').click(custom,[m,k2,r2],o2)
    with gr.Tab('Exact results'):
        out=gr.Dataframe(value=lambda:load_csv('exact_family_audit.csv'))
    with gr.Tab('Random decision benchmark'):
        out2=gr.Dataframe(value=lambda:load_csv('random_decision_stats.csv'))
    with gr.Tab('WDBC real-data audit'):
        out3=gr.Dataframe(value=lambda:load_csv('real_wdbc_audit.csv'))
    with gr.Tab('Bootstrap stability'):
        out4=gr.Dataframe(value=lambda:load_csv('real_wdbc_bootstrap.csv').head(200))
    with gr.Tab('Parity scaling'):
        out5=gr.Dataframe(value=lambda:load_csv('parity_scaling.csv'))
    gr.Markdown('The WDBC tab is a structural support-reconstruction audit, not a clinical decision system.')
if __name__=='__main__': demo.launch()

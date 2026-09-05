import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
subprocess.check_call([sys.executable,str(root/'src'/'run_experiments.py')])
subprocess.check_call([sys.executable,'-m','pytest','-q',str(root/'tests')])
print('Causal ambiguity reproduction completed successfully.')

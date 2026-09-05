from pathlib import Path
from src.experiments import run_all
run_all(Path(__file__).resolve().parent)
print('All HCAT results and figures regenerated.')

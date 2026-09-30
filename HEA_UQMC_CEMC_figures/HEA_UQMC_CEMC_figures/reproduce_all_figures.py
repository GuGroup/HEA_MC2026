from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
for name in ['Fig_1', 'Fig_2', 'Fig_3', 'Fig_4', 'Fig_5', 'Fig_6', 'Fig_S6', 'Fig_S7', 'Fig_S8', 'Fig_S9', 'Fig_S10', 'Fig_S11', 'Fig_S12', 'Fig_S13', 'Fig_S14', 'Fig_S15']:
    subprocess.run([sys.executable,str(root/name/'plot.py')],check=True)

#!/usr/bin/env python3
"""Evaluate all five sites and recreate the three original facet parity figures.
Run: python evaluate_all.py
Use --plots-only to redraw facet plots from already generated predictions.
"""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SITES = ['fcc111/fcc_hollow', 'fcc111/top', 'fcc100/hollow', 'fcc100/bridge', 'fcc110/bridge']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    if not args.plots_only:
        for site in SITES:
            print('EVALUATE',site,flush=True)
            subprocess.run([sys.executable,str(root/site/'testing/evaluate_model.py')],check=True)
    spec=importlib.util.spec_from_file_location('plotting',root/SITES[0]/'testing/evaluate_model.py')
    plotting=importlib.util.module_from_spec(spec);spec.loader.exec_module(plotting)
    for facet,sites,labels,name in [
        ('fcc111',['top','fcc_hollow'],['(a)','(b)'],'H_adsorption_parity_top_fcc_train_test.png'),
        ('fcc100',['hollow','bridge'],['(c)','(d)'],'H_adsorption_parity_fcc100_hollow_bridge_train_test.png'),
        ('fcc110',['bridge'],['(e)'],'H_adsorption_parity_fcc110_long_bridge_train_test.png')]:
        fig,axes=plt.subplots(1,len(sites),figsize=(9*len(sites),9),squeeze=False)
        for ax,site,label in zip(axes[0],sites,labels):
            r=root/facet/site/'testing/results'
            train=pd.read_csv(r/'train_predictions.csv')
            test=pd.read_csv(r/'test_predictions_original_plot.csv')
            plotting.draw_parity_panel(ax,train,test,label)
        if len(sites)>1:fig.tight_layout(w_pad=2.0)
        else:fig.tight_layout()
        fig.savefig(root/facet/name,dpi=300)
        plt.close(fig)

if __name__=='__main__':main()

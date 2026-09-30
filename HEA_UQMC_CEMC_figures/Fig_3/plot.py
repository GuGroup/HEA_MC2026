"""Reproduce this figure from the numeric inputs in data/."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parent
DATA=BASE/'data'
PT={'Pd':'#FFD700','Pt':'#1F77B4','Ir':'#2CA02C','Rh':'#FF9900','Ru':'#E31A1C'}
FE={'Pd':'#FFD700','Pt':'#1F77B4','Fe':'#E31A1C','Co':'#FF9900','Ni':'#2CA02C'}
METHODS=['Homogeneous','CEMC','CEMC + layer shuffled']
OH=r'$\Delta E_{\mathrm{OH}}$ (eV)';H=r'$\Delta G_{\mathrm{H}}$ (eV)'

def label(ax,s):ax.text(.02,.97,s,transform=ax.transAxes,va='top',fontweight='bold',fontsize=12)
def save(fig):
 out=BASE/'outputs';out.mkdir(exist_ok=True)
 for ext in ['png','pdf']:fig.savefig(out/(BASE.name+'.'+ext),dpi=300,bbox_inches='tight')
 plt.close(fig);print(out/(BASE.name+'.png'))
def figure3():
 fig,axes=plt.subplots(2,2,figsize=(11.4,8.1),layout='constrained')
 for col,tag in enumerate(['Pt','Fe']):
  ax=axes[0,col];tab=pd.read_csv(DATA/f'{tag}_composition.csv');colors=PT if tag=='Pt' else FE
  for e in tab.element.unique():
   d=tab[tab.element==e];ax.plot(d.temperature_K,d.smoothed_mean_surface_atomic_fraction,color=colors[e],label=e,lw=2.2)
  ax.set_ylim(0,.8 if tag=='Pt' else 1);ax.set_ylabel('Surface atomic fraction');ax.legend(ncol=2,frameon=False,loc='upper left',bbox_to_anchor=(.1,1));label(ax,'(a)' if col==0 else '(b)')
  ax=axes[1,col];tab=pd.read_csv(DATA/f'{tag}_wc.csv');pairs=list(tab.pair.unique());colors=plt.get_cmap('tab10')(np.linspace(0,1,len(pairs)))
  for pair,color in zip(pairs,colors):
   d=tab[tab.pair==pair];ax.plot(d.temperature_K,d.raw_mean_WC_alpha,color=color,lw=.8,alpha=.2);ax.plot(d.temperature_K,d.smoothed_mean_WC_alpha,color=color,label=pair,lw=1.7)
  ax.axhline(0,color='.4',ls='--',lw=.6);ax.set_ylabel(r'$\alpha_{ij}^{\mathrm{surf}}$');ax.set_ylim((-1.5,1) if tag=='Pt' else (-2,1));ax.legend(ncol=2,frameon=False,loc='lower left',fontsize=8);label(ax,'(c)' if col==0 else '(d)')
 for ax in axes.flat:ax.set_xlim(2000,298);ax.set_xticks([2000,1600,1200,800,298]);ax.set_xlabel('Temperature (K)')
 save(fig)
def density():
 cases=['Pt','fcc100_hollow'] if BASE.name=='Fig_4' else ['fcc100_bridge','fcc110_bridge','fcc111_hollow','fcc111_top']
 fig,axes=plt.subplots(len(cases),3,figsize=(12.5,3.7*len(cases)),squeeze=False,layout='constrained')
 for row,case in enumerate(cases):
  tab=pd.read_csv(DATA/f'{case}_density.csv');meta=json.loads((DATA/f'{case}_metadata.json').read_text());colors=PT if case=='Pt' else FE;els=['Ir','Pd','Pt','Rh','Ru'] if case=='Pt' else list(meta['elements']);mx=tab.density.max()*1.12
  for col,method in enumerate(METHODS):
   ax=axes[row,col]
   for e in els:
    d=tab[(tab.method==method)&(tab.element==e)];ax.plot(d.bin_center_eV,d.density,color=colors[e],lw=1.7,label=e)
   ax.axvline(meta['optimum_eV'],color='black',ls='--',lw=1.1);ax.set_title(method,fontsize=11);ax.set_xlabel(OH if case=='Pt' else H);ax.set_ylim(-mx*.025,mx);ax.set_xlim(tab.bin_center_eV.min(),tab.bin_center_eV.max())
   activity=meta['activities'][method];xy=(.05,.69) if col==0 else (.03,.94)
   if case=='fcc110_bridge' or (case=='fcc111_hollow' and col==0) or (case=='fcc111_top' and col==0):xy=(.58,.9)
   ax.text(*xy,f'Activity = {activity:.2g}',transform=ax.transAxes,va='top',fontsize=11,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1.5))
   if col==0:
    ax.legend(ncol=2,frameon=False,fontsize=8,loc='upper left');ax.set_ylabel('Probability density');ax.text(-.12,1.08,f'({chr(97+row)})',transform=ax.transAxes,fontweight='bold',fontsize=13)
   else:ax.tick_params(labelleft=False)
 save(fig)
def figure5():
 fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
 for i,(tag,ax) in enumerate(zip(['Pt','Fe'],axes)):
  folder=DATA/tag;tab=pd.read_csv(next(folder.glob('*bins.csv')));meta=json.loads((folder/'metadata.json').read_text());colors=PT if tag=='Pt' else FE;column='smoothed_probability' if tag=='Pt' else 'smoothed_probability_density_eV_inv'
  for e,d in tab.groupby('element',sort=False):ax.plot(d.bin_center_eV,d[column],color=colors[e],label=e,lw=1.7);ax.fill_between(d.bin_center_eV,d[column],color=colors[e],alpha=.08)
  optimum=1.1 if tag=='Pt' else 0;ax.axvline(optimum,color='#D62728',ls='--',lw=1);x=np.linspace(tab.bin_left_eV.min(),tab.bin_right_eV.max(),2000);ax.set_xlim(x.min(),x.max());ax.set_ylim(bottom=0);ax.set_xlabel(OH if tag=='Pt' else H);ax.set_ylabel('Probability');ax2=ax.twinx();ax2.plot(x,np.exp(-abs(x-optimum)/(8.617333262145e-5*298)),color='#222222',lw=1.5,label='Activity volcano');ax2.set_ylim(0,1.05);ax2.set_ylabel('Activity');h,l=ax.get_legend_handles_labels();h2,l2=ax2.get_legend_handles_labels();ax.legend(h+h2,l+l2,ncol=4,frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(.5,1));label(ax,f'({chr(97+i)})')
 save(fig)
def mass():
 cases=['Pt','fcc100_hollow'] if BASE.name=='Fig_6' else ['fcc100_bridge','fcc110_bridge','fcc111_hollow','fcc111_top'];fig,axes=plt.subplots(len(cases),5,figsize=(16,3.3*len(cases)),squeeze=False)
 for row,case in enumerate(cases):
  folder=DATA/case;pt=case=='Pt';tab=pd.read_csv(folder/('probability_mass_weighted_histograms_1x5_bin_data_requested_shift.csv' if pt else 'weighted_histogram_bin_data.csv'));elements=['Ir','Pd','Pt','Rh','Ru'] if pt else ['Fe','Co','Ni','Pd','Pt'];opt=1.1 if pt else 0
  for col,e in enumerate(elements):
   ax=axes[row,col];d=tab[tab.element==e];x=d.bin_center_eV;w=float(d.bin_right_eV.iloc[0]-d.bin_left_eV.iloc[0]);inside=d['cemc_inside_random_support_mass' if pt else 'cemc_mass_inside_random_support'];outside=d['cemc_outside_random_support_mass' if pt else 'cemc_mass_outside_random_support'];rnd=d['random_probability_mass' if pt else 'random_fractional_probability_mass'];ax.axvspan(opt-.1,opt+.1,color='#808080',alpha=.15);ax.bar(x,inside,width=w*.92,color='#4C78A8',alpha=.8);ax.bar(x,outside,width=w*.92,color='#E45756',alpha=.82)
   if pt:ax.stairs(rnd,np.r_[d.bin_left_eV.to_numpy(),d.bin_right_eV.iloc[-1]],color='black',lw=1)
   else:ax.step(x,rnd,where='mid',color='black',lw=1)
   ax.axvline(opt,color='black',ls='--',lw=1);ax.set_title(e,fontsize=12);ax.set_xlabel(OH if pt else H);ax.grid(axis='y',alpha=.18);ax.text(.04,.95,f'$C_{{{e}}}$ = {100*inside.sum():.1f}%',transform=ax.transAxes,va='top',fontsize=11,bbox=dict(facecolor='white',edgecolor='none',alpha=.78,pad=1));ax.tick_params(labelsize=8)
   if col==0:ax.set_ylabel('Probability mass');ax.text(-.22,1.12,f'({chr(97+row)}) '+('' if pt else case.replace('fcc','fcc(').replace('_',') ')),transform=ax.transAxes,fontsize=11,fontweight='bold')
  ymax=max(ax.get_ylim()[1] for ax in axes[row]);xmin=tab.bin_left_eV.min();xmax=tab.bin_right_eV.max()
  for ax in axes[row]:ax.set_ylim(0,ymax);ax.set_xlim(xmin,xmax)
 handles=[Line2D([0],[0],color='black',label='Homogeneous'),Patch(facecolor='#4C78A8',label='CEMC inside Homogeneous distribution'),Patch(facecolor='#E45756',label='CEMC outside Homogeneous distribution'),Patch(facecolor='#808080',alpha=.15,label='Optimal window'),Line2D([0],[0],color='black',ls='--',label='Optimal')]
 fig.legend(handles=handles,ncol=5,loc='upper center',fontsize=10,frameon=False);fig.tight_layout(rect=(.01,0,1,.94),h_pad=2.5,w_pad=.5);save(fig)
def layers():
 fig,axes=plt.subplots(4,1,figsize=(10,11.7),layout='constrained')
 for row,(tag,ax) in enumerate(zip(['Pt','Fe100','Fe110','Fe111'],axes)):
  tab=pd.read_csv(DATA/f'{tag}.csv');colors=PT if tag=='Pt' else FE;els=['Pd','Pt','Rh','Ru','Ir'] if tag=='Pt' else ['Pd','Pt','Co','Fe','Ni'];left=np.zeros(5);y=np.arange(5)
  for e in els:
   v=tab[e+' (%)'].to_numpy();ax.barh(y,v,left=left,height=.72,color=colors[e],label=e)
   for k,value in enumerate(v):
    if value>=2:ax.text(left[k]+value/2,k,f'{value:.1f}',ha='center',va='center',fontsize=8)
   left+=v
  ax.set_yticks(y,tab.iloc[:,0]);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xlabel('Subsurface composition (%)');ax.legend(ncol=5,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.22));ax.text(-.13,1.04,f'({chr(97+row)})',transform=ax.transAxes,fontsize=13,fontweight='bold')
 save(fig)
def main():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42})
 {'Fig_3':figure3,'Fig_4':density,'Fig_5':figure5,'Fig_6':mass,'Fig_S13':layers,'Fig_S14':density,'Fig_S15':mass}[BASE.name]()
if __name__=='__main__':main()

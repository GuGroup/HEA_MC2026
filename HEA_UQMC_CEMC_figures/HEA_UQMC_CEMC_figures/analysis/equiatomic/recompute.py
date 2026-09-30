"""Recompute figure inputs from the packaged equiatomic structures and models."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import atomic_numbers
from evaluate import evaluate,METHODS
import surface_core
R=Path(__file__).resolve().parents[2]
CHECK={}

def smooth(t,y):
 delta=t[:,None]-t[None,:];w=np.exp(-.5*(delta/50)**2);w[abs(delta)>150]=0;w[:,~np.isfinite(y)]=0
 return np.divide(w@np.nan_to_num(y),w.sum(axis=1),out=np.full(len(t),np.nan),where=w.sum(axis=1)>0)
def surface():
 for tag,facet in [('Pt','PtPdRhRuIr_fcc111'),('Fe','FeCoNiPdPt_fcc100')]:
  d=np.load(R/'workflows'/facet/'equi-atomic/dense_10K/cemc_structures_10K.npz');idx,i,j,_,_=surface_core.surface_graph(d);T=d['temperatures_K'];comp=pd.read_csv(R/'Fig_3/data'/f'{tag}_composition.csv');wc=pd.read_csv(R/'Fig_3/data'/f'{tag}_wc.csv');errs=[]
  for e,g in comp.groupby('element',sort=False):
   vals=(d['Z'][:,:,idx]==atomic_numbers[e]).mean(axis=2).mean(axis=1);tmap={int(t):v for t,v in zip(T,vals)};v=np.asarray([tmap[int(t)] for t in g.temperature_K]);errs += [float(np.max(abs(v-g.raw_mean_surface_atomic_fraction))),float(np.max(abs(smooth(g.temperature_K.to_numpy(),v)-g.smoothed_mean_surface_atomic_fraction)))]
  elements=list(comp.element.unique());zmap={atomic_numbers[e]:ei for ei,e in enumerate(elements)};allw=np.asarray([[surface_core.wc_for_structure(z,idx,i,j,zmap,5) for z in zs] for zs in d['Z']]);
  for pair,g in wc.groupby('pair',sort=False):
   a,b=pair.split('-');vals=allw[:,:,elements.index(a),elements.index(b)];count=np.isfinite(vals).sum(axis=1);v=np.divide(np.nansum(vals,axis=1),count,out=np.full(len(T),np.nan),where=count>0);tmap=dict(zip(T,v));v=np.asarray([tmap[int(t)] for t in g.temperature_K]);errs += [float(np.nanmax(abs(v-g.raw_mean_WC_alpha))),float(np.nanmax(abs(smooth(g.temperature_K.to_numpy(),v)-g.smoothed_mean_WC_alpha)))]
  CHECK[tag+'_surface_composition_WC_max_error']=max(errs)
 for tag,facet,site in [('Pt','PtPdRhRuIr_fcc111','top'),('Fe100','FeCoNiPdPt_fcc100','hollow'),('Fe110','FeCoNiPdPt_fcc110','bridge'),('Fe111','FeCoNiPdPt_fcc111','hollow')]:
  d=np.load(R/'workflows'/facet/'equi-atomic/cases'/site/'structures.npz');z=d['Z'][-1];zz=(d['scaled_positions']@d['cell_A'])[:,2];keys=np.round(zz,6);layers=[np.flatnonzero(keys==v) for v in sorted(set(keys))];tab=pd.read_csv(R/'Fig_S13/data'/f'{tag}.csv');err=0.
  for col in tab.columns:
   if col in ['Layer pair','Total (%)']:continue
   e=col.split()[0];v=[100*(z[:,np.r_[layers[k],layers[-1-k]]]==atomic_numbers[e]).mean() for k in range(5)];err=max(err,float(np.max(abs(v-tab[col]))))
  CHECK[tag+'_layer_pair_max_percentage_error']=err

def distributions():
 cases=[('Pt','PtPdRhRuIr_fcc111','top'),('fcc100_hollow','FeCoNiPdPt_fcc100','hollow'),('fcc100_bridge','FeCoNiPdPt_fcc100','bridge'),('fcc110_bridge','FeCoNiPdPt_fcc110','bridge'),('fcc111_hollow','FeCoNiPdPt_fcc111','hollow'),('fcc111_top','FeCoNiPdPt_fcc111','top')]
 original=pd.read_csv(R/'analysis/equiatomic/original_no_shift_statistics.csv')
 for tag,facet,site in cases:
  case=R/'workflows'/facet/'equi-atomic/cases'/site;cache=case/'site_predictions_no_shift.npz';metafile=case/'activity_no_shift.json'
  arrays,meta=evaluate(case)
  np.savez_compressed(cache,**arrays);metafile.write_text(json.dumps(meta,indent=2));print('EVALUATED',tag,meta['activities'],flush=True)
  fig='4' if tag in ['Pt','fcc100_hollow'] else 'S14';out=R/f'Fig_{fig}/data';out.mkdir(parents=True,exist_ok=True)
  elements=meta['elements'];opt=meta['optimum_eV'];allv=np.concatenate([arrays[k+'_values'].ravel() for k in ['random','cemc','shuffled']])
  if tag=='Pt':
   ref=np.asarray(json.loads((out/'Pt_bin_reference.json').read_text())['common_bin_edges_eV']);width=ref[1]-ref[0];shifts=np.asarray([{'Ir':.05068,'Pd':.53799,'Pt':.28892,'Rh':.20160,'Ru':.20917}[e] for e in elements]);extra=np.concatenate([(arrays[k+'_values']-arrays[k+'_fractions']@shifts).ravel() for k in ['random','cemc','shuffled']]);lo=min(allv.min(),extra.min(),opt)-.06;hi=max(allv.max(),extra.max(),opt)+.06;left=ref[0]+min(0,int(np.floor((lo-ref[0])/width)))*width;right=ref[-1]+max(0,int(np.ceil((hi-ref[-1])/width)))*width;edges=np.linspace(left,right,int(round((right-left)/width))+1)
  else:edges=np.linspace(min(allv.min(),opt)-.06,max(allv.max(),opt)+.06,91)
  rows=[];kernel=np.exp(-.5*(np.arange(-6,7)/1.5)**2);kernel/=kernel.sum()
  for method,k in zip(METHODS,['random','cemc','shuffled']):
   values=arrays[k+'_values'].ravel();weights=arrays[k+'_fractions'].reshape(-1,5)
   for ei,e in enumerate(elements):
    counts=np.histogram(values,bins=edges,weights=weights[:,ei])[0];density=np.convolve(counts/(len(values)*(edges[1]-edges[0])),kernel,mode='same')
    rows.extend({'method':method,'element':e,'bin_center_eV':float(x),'density':float(y)} for x,y in zip((edges[:-1]+edges[1:])/2,density))
  pd.DataFrame(rows).to_csv(out/f'{tag}_density.csv',index=False);(out/f'{tag}_metadata.json').write_text(json.dumps(meta|{'histogram_edges_eV':edges.tolist(),'smoothing':'Gaussian sigma=1.5 bins, support -6..6; weighted counts / (all site count * bin width)'},indent=2))
  if tag!='Pt':
   tab=original[(original.facet.astype(str)==facet[-3:])&(original.site==site)];errors=[]
   for method,k,origmethod in zip(METHODS,['random','cemc','shuffled'],['Random','CEMC','CEMC + layer shuffle']):
    per=tab[tab.method==origmethod].drop_duplicates('temperature_K');per=per[pd.to_numeric(per.temperature_K,errors='coerce').notna()];errors.append(abs(meta['activities'][method]-per.activity_no_log.mean()))
   CHECK[tag+'_activity_max_error']=max(errors)
  # Compare atom-level no-shift predictions against Fig.5 source tables.
  if tag in ['Pt','fcc100_hollow']:
   folder=R/'Fig_5/data'/('Pt' if tag=='Pt' else 'Fe');tab=pd.read_csv(next(folder.glob('*site_values.csv')));col='OH_BE_eV' if tag=='Pt' else 'deltaG_H_eV';CHECK[tag+'_Fig5_site_energy_max_error']=float(np.max(abs(tab[col].to_numpy()-arrays['random_values'].ravel())))
  if tag=='Pt':
   original_pt=pd.read_csv(R/'analysis/equiatomic/original_Pt_no_shift_activity.csv');CHECK['Pt_activity_max_error']=max(abs(meta['activities'][m]-original_pt.loc[original_pt.method==om,'activity_no_log'].mean()) for m,om in zip(METHODS,['Random','CEMC','CEMC + layer shuffle']))
  figmass='6' if tag in ['Pt','fcc100_hollow'] else 'S15';massdir=R/f'Fig_{figmass}/data'/tag
  if tag=='Pt':shifts=json.loads((massdir/'probability_mass_and_paired_bootstrap_requested_shift_metadata.json').read_text())['BE_shifts_eV'];file=massdir/'probability_mass_weighted_histograms_1x5_bin_data_requested_shift.csv';rcol='random_probability_mass';ccol='cemc_probability_mass'
  else:
   row=pd.read_csv(massdir/'BE_shifts.csv').iloc[0];shifts={e:row[f'be_shift_{e}_exact_eV'] for e in elements};file=massdir/'weighted_histogram_bin_data.csv';rcol='random_fractional_probability_mass';ccol='cemc_fractional_probability_mass'
  (case/'figure_mass_BE_shifts.json').write_text(json.dumps(shifts,indent=2));tab=pd.read_csv(file);errors=[]
  for ei,e in enumerate(elements):
   t=tab[tab.element==e];edges=np.r_[t.bin_left_eV.to_numpy(),t.bin_right_eV.iloc[-1]]
   for k,col in [('random',rcol),('cemc',ccol)]:
    values=(arrays[k+'_values']-arrays[k+'_fractions']@np.asarray([shifts[e] for e in elements])).ravel();weights=arrays[k+'_fractions'].reshape(-1,5)[:,ei];counts=np.histogram(values,bins=edges,weights=weights)[0];mass=counts/counts.sum();errors.append(float(np.max(abs(mass-t[col]))))
  CHECK[tag+'_Fig6_S15_probability_mass_max_error']=max(errors)

def main():
 surface();distributions();(R/'validation/equiatomic_numeric_checks.json').write_text(json.dumps(CHECK,indent=2));print(json.dumps(CHECK,indent=2))
if __name__=='__main__':main()

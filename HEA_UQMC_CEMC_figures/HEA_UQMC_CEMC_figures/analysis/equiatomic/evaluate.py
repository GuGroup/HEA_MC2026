"""Evaluate saved equiatomic structures without rerunning CEMC."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from ase import Atoms
import pt_core,fe_core
METHODS=['Homogeneous','CEMC','CEMC + layer shuffled']

def evaluate(case,structures=None,shifts=None,activity_model=None):
 info=json.loads((case/'provenance.json').read_text());cfg=json.loads((case/'config.json').read_text());pt=info['facet'].startswith('Pt');core=pt_core if pt else fe_core;elements=list(core.ELEMENTS)
 with np.load(structures or case/'structures.npz') as src:d={k:src[k] for k in src.files}
 if 'random_Z' not in d:
  with np.load(case/'structures.npz') as initial:d['random_Z']=initial['random_Z']
 atoms=Atoms(numbers=d['Z'][0,0],cell=d['cell_A'],scaled_positions=d['scaled_positions'],pbc=d['pbc']);zz=np.round(atoms.positions[:,2],6);layers=[np.flatnonzero(zz==v) for v in sorted(set(zz))]
 model=core.parse_model(activity_model or case/cfg['activity_model']);shift={e:float((shifts or {}).get(e,0)) for e in elements};sets={'random':d['random_Z'],'cemc':d['Z'].reshape(-1,d['Z'].shape[-1])};shuffled=[]
 for ti,t in enumerate(d['temperatures_K']):
  for run,z in enumerate(d['Z'][ti]):
   seed=int.from_bytes(hashlib.sha256('|'.join(map(str,info['shuffle_seed_prefix']+[run,int(t)])).encode()).digest()[:8],'little');rng=np.random.default_rng(seed);s=z.copy()
   for layer in layers:v=s[layer].copy();rng.shuffle(v);s[layer]=v
   shuffled.append(s)
 sets['shuffled']=np.asarray(shuffled);arrays={}
 for method,zs in sets.items():
  values=[];fractions=[]
  for z in zs:
   if pt:
    v,labels=core.evaluate(model,layers[-1],core.occupancy_from_numbers(z),shift);f=np.asarray([labels==e for e in elements],float).T
   else:
    result=core.evaluate_sites(model,core.atomic_numbers_to_occ(z),np.asarray([shift[e] for e in elements]));v=result[2]-model.e_opt;f=result[3]
   values.append(v);fractions.append(f)
  arrays[method+'_values']=np.asarray(values);arrays[method+'_fractions']=np.asarray(fractions)
 optimum=model['e_opt'] if pt else 0.;temp=model['activity_temperature'] if pt else model.activity_temperature
 meta={'case':info['facet']+'/'+info['site'],'elements':elements,'shifts_eV':shift,'optimum_eV':optimum,'activity_temperature_K':temp,'temperatures_K':d['temperatures_K'].tolist(),'n_runs':len(d['random_Z']),'energy_coordinate':'DeltaE_OH_eV' if pt else 'DeltaG_H_eV','activities':{m:float(np.exp(-abs(arrays[k+'_values']-optimum)/(8.617333262145e-5*temp)).mean()) for m,k in zip(METHODS,sets)}}
 return arrays,meta

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--structures',type=Path);p.add_argument('--shifts',type=Path);p.add_argument('--activity-model',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();arrays,meta=evaluate(a.case,a.structures,json.loads(a.shifts.read_text()) if a.shifts else None,a.activity_model);a.output.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output/'site_predictions.npz',**arrays);(a.output/'activity.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()

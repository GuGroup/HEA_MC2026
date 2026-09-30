"""Portable equiatomic CEMC generation and structure export."""
from pathlib import Path
import argparse,json,subprocess,sys,os
import numpy as np
from ase import Atoms
from ase.io import write
BASE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['build','slabs','export','evaluate']);p.add_argument('--site',choices=sorted(x.name for x in (BASE/'cases').iterdir()));p.add_argument('--ranks',type=int,default=1);p.add_argument('--runs',type=int,default=20);p.add_argument('--dense',action='store_true',help='Save snapshots every 10 K instead of 100 K');p.add_argument('--output',type=Path);p.add_argument('--structures',type=Path);p.add_argument('--temperature',type=int,default=298);p.add_argument('--activity-model',type=Path,help='Compatible exported model text file');p.add_argument('--shifts',type=Path,help='JSON dictionary of per-element BE shifts in eV');a=p.parse_args()
 b=BASE/'build';b.mkdir(exist_ok=True)
 if a.command=='build':
  subprocess.run(['mpicxx','-O3','-std=c++17','-DUSE_MPI',str(BASE/'src/uq_cemc_mpi.cpp'),'-o',str(b/'uq_cemc_mpi')],check=True)
  subprocess.run(['g++','-O3','-std=c++17',str(BASE/'src/reconstruct_random_slab.cpp'),'-o',str(b/'reconstruct_random_slab')],check=True);return
 if not a.site:p.error('--site is required')
 case=BASE/'cases'/a.site;out=(a.output or BASE/'generated'/a.site).resolve();out.mkdir(parents=True,exist_ok=True)
 if a.command=='evaluate':
  cmd=[sys.executable,str(BASE.parents[2]/'analysis/equiatomic/evaluate.py'),'--case',str(case),'--output',str(out)]
  if a.structures:cmd+=['--structures',str(a.structures.resolve())]
  if a.activity_model:cmd+=['--activity-model',str(a.activity_model.resolve())]
  if a.shifts:cmd+=['--shifts',str(a.shifts.resolve())]
  subprocess.run(cmd,check=True);return
 if a.command=='export':
  with np.load(a.structures or case/'structures.npz') as d:
   idx=np.flatnonzero(d['temperatures_K']==a.temperature)
   if len(idx)!=1:p.error('Requested temperature is not in the archive')
   for run,z in enumerate(d['Z'][idx[0]]):
    atoms=Atoms(numbers=z,scaled_positions=d['scaled_positions'],cell=d['cell_A'],pbc=d['pbc']);write(out/f'run_{run:02d}_T{a.temperature}.cif',atoms)
  return
 if not 1<=a.runs<=20:p.error('--runs must be 1..20')
 if not (b/'uq_cemc_mpi').exists():p.error('Run build first')
 cfg=json.loads((case/'config.json').read_text())
 for k in ['ce_export','schedule_export','activity_model','composition_csv','experimental_activity_csv']:cfg[k]=str((case/cfg[k]).resolve())
 if a.dense:cfg['schedule_export']=str(BASE/'dense_10K/schedule_10K.txt')
 results=out/'results'
 if results.exists():p.error('Output results already exist; choose a fresh --output directory')
 cfg['output_dir']=str(results);cfg['n_runs']=str(a.runs);config=out/'run.ini';config.write_text(''.join(f'{k} = {v}\n' for k,v in cfg.items()))
 env=os.environ.copy();env['OMP_NUM_THREADS']='1';subprocess.run(['mpirun','-np',str(a.ranks),str(b/'uq_cemc_mpi'),'--config',str(config)],check=True,env=env)
 temps=sorted([int(x.name[1:]) for x in (results/'structures_by_temperature').iterdir()],reverse=True);z=[];en=[];steps=[]
 for t in temps:
  rows=sorted([json.loads(s) for s in (results/'structures_by_temperature'/f'T{t:05d}'/'comp_0000.jsonl').read_text().splitlines() if s.strip()],key=lambda r:r['run']);assert len(rows)==a.runs
  z.append([r['Z'] for r in rows]);en.append([r['energy'] for r in rows]);steps.append([r['mc_step'] for r in rows])
 random=[]
 for run in range(a.runs):
  f=out/f'random_{run}.json';subprocess.run([str(b/'reconstruct_random_slab'),'--ce-export',cfg['ce_export'],'--seeds',str(results/'random_slab_seeds.csv'),'--trial','0','--composition-index','0','--run',str(run),'--output',str(f)],check=True,stdout=subprocess.DEVNULL);random.append(json.loads(f.read_text()))
 with np.load(case/'structures.npz') as d:geom={k:d[k] for k in ['cell_A','scaled_positions','pbc']}
 np.savez_compressed(out/'structures.npz',Z=np.asarray(z,dtype=np.uint8),random_Z=np.asarray(random,dtype=np.uint8),temperatures_K=temps,energy_eV=en,mc_step=steps,**geom)
 print(out/'structures.npz')
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Portable Study 1 driver preserving its original packed format and backend."""
from pathlib import Path
import argparse,json,subprocess,sys,shutil
ROOT=Path(__file__).resolve().parent

def run(cmd):
 print(' '.join(map(str,cmd)),flush=True);subprocess.run(list(map(str,cmd)),check=True,cwd=ROOT)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',choices=['build','slabs','activity','score'])
 p.add_argument('--ranks',type=int,default=1);p.add_argument('--workers',type=int,default=1)
 p.add_argument('--shard-index',type=int,choices=range(5),default=0)
 p.add_argument('--sample',action='store_true')
 p.add_argument('--slab-root',type=Path);p.add_argument('--output',type=Path)
 p.add_argument('--activity-model',type=Path);p.add_argument('--be-mean',type=float,default=.04233);p.add_argument('--be-sigma',type=float,default=.2604)
 a=p.parse_args();build=ROOT/'build';build.mkdir(exist_ok=True)
 if a.command=='build':
  run(['mpicxx','-std=c++17','-O3','-DUSE_MPI',ROOT/'src/uq_cemc_mpi.cpp','-o',build/'generate_slabs'])
  run(['g++','-std=c++17','-O3',ROOT/'src/export_trial_be_shifts.cpp','-o',build/'export_shifts']);return
 out=(a.output.resolve() if a.output else ROOT/'outputs'/a.command);out.mkdir(parents=True,exist_ok=True)
 if a.command=='slabs':
  cfg={}
  for line in (ROOT/'inputs/uq_config.ini').read_text().splitlines():
   if '=' in line and not line.lstrip().startswith('#'):
    k,v=line.split('=',1);cfg[k.strip()]=v.strip()
  for key,name in [('ce_export','ce_export.txt'),('schedule_export','schedule_snapshots.txt'),('activity_model','activity_model_oh.txt'),('composition_csv','composition_new.csv'),('experimental_activity_csv','simulation_experimental_activity.json')]:cfg[key]=str(ROOT/'inputs'/name)
  cfg.update(output_dir=str(out/f'shard_{a.shard_index:02d}'),trial_start=str(a.shard_index),trial_stride='5')
  if a.sample:cfg.update(n_trials='281',trial_start='280',trial_stride='1',n_runs='20',max_compositions='1')
  config=out/f'config_shard_{a.shard_index:02d}.ini';config.write_text('\n'.join(f'{k} = {v}' for k,v in cfg.items())+'\n')
  run(['mpirun','-np',a.ranks,build/'generate_slabs','--config',config]);return
 if a.command=='activity':
  src=a.slab_root.resolve() if a.slab_root else ROOT/'sample/results' if a.sample else ROOT/'outputs/slabs'
  for shard in sorted(src.glob('shard_*')):
   dest=out/shard.name;dest.mkdir(exist_ok=True)
   link=dest/'packed_atoms'
   if not link.exists():link.symlink_to(shard/'packed_atoms',target_is_directory=True)
  shifts=out/'be_shifts.csv'
  with shifts.open('w') as f:subprocess.run([str(build/'export_shifts'),'10000','20260706',str(a.be_mean),str(a.be_sigma)],stdout=f,check=True)
  run([sys.executable,ROOT/'scripts/calculate_packed_activities.py','--results-root',out,'--activity-model',a.activity_model.resolve() if a.activity_model else ROOT/'inputs/activity_model_oh.txt','--topology',ROOT/'inputs/activity_topology.json','--schedule',ROOT/'inputs/schedule_snapshots.txt','--be-shifts',shifts,'--workers',a.workers]);return
 run([sys.executable,ROOT/'scripts/score_log_activity_zarr.py','--results-root',a.slab_root.resolve() if a.slab_root else ROOT/'outputs/activity','--experimental',ROOT/'inputs/orr_log_matched_activity.json','--mask',ROOT/'inputs/static_grid_mask.npy','--output-dir',out,'--world-size','5','--include-center-cross'])
if __name__=='__main__':main()

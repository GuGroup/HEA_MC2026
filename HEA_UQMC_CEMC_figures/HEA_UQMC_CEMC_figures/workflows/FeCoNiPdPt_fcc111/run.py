#!/usr/bin/env python3
"""Build the bundled backend, generate slabs once, and reevaluate their activity."""
import argparse,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(command):
    print(' '.join(map(str,command)),flush=True)
    subprocess.run(list(map(str,command)),check=True,cwd=ROOT)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['build','slabs','activity','random'])
    p.add_argument('--case')
    p.add_argument('--ranks',type=int,default=1)
    p.add_argument('--slab-dir',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--sample',action='store_true',help='One trial, first composition, one run; all 18 temperatures')
    p.add_argument('--trial-start',type=int,default=0)
    p.add_argument('--trial-end',type=int)
    p.add_argument('--activity-model',type=Path)
    p.add_argument('--be-mean',type=float)
    p.add_argument('--be-sigma',type=float)
    args=p.parse_args()
    info=json.loads((ROOT/'workflow.json').read_text())
    build=ROOT/'build';build.mkdir(exist_ok=True)
    if args.command=='build':
        for src,target in [('recover_cemc_slabs_3bit_mpi.cpp','generate_slabs'),('recover_from_slab3bit_mpi.cpp','evaluate_slabs'),('recover_random_activity_mpi.cpp','evaluate_random')]:
            flags=['-DORR_MODEL'] if info['orr'] and target=='evaluate_slabs' else []
            run(['mpicxx','-O3','-std=c++17','-DUSE_MPI',*flags,ROOT/'src'/src,'-o',build/target])
        return
    if args.case not in info['cases']:p.error('--case must be one of '+', '.join(info['cases']))
    cfg=json.loads((ROOT/'configs'/(args.case+'.json')).read_text())
    for key in ['ce_export','schedule_export','activity_model','composition_csv','experimental_activity_csv']:
        cfg[key]=str((ROOT/cfg[key]).resolve())
    if args.activity_model:cfg['activity_model']=str(args.activity_model.resolve())
    if args.be_mean is not None:cfg['be_error_mean']=str(args.be_mean)
    if args.be_sigma is not None:cfg['be_error_sigma']=str(args.be_sigma)
    if args.sample:cfg.update(n_trials='1',n_runs='1',max_compositions='1')
    default_slab=ROOT/'outputs'/('slabs_'+args.case if info['orr'] else 'slabs_shared_facet')
    output=(args.output.resolve() if args.output else default_slab if args.command=='slabs' else ROOT/'outputs'/(args.case+'_'+args.command))
    if args.command!='slabs' and output.exists() and any(output.iterdir()):
        p.error('Choose an empty --output directory; evaluation appends some summaries.')
    output.mkdir(parents=True,exist_ok=True)
    cfg['output_dir']=str(output)
    runtime=ROOT/'outputs/runtime_configs';runtime.mkdir(parents=True,exist_ok=True)
    cp=runtime/(args.case+'_'+args.command+('_sample' if args.sample else '')+'.ini')
    cp.write_text('\n'.join(f'{k} = {v}' for k,v in cfg.items())+'\n')
    mpi=['mpirun','-np',str(args.ranks)]
    common=['--config',cp,'--trial-start',args.trial_start,'--trial-end',args.trial_end or int(cfg['n_trials'])]
    if args.command=='slabs':
        run([*mpi,build/'generate_slabs',*common,'--output-dir',output,'--shard-trials','40','--no-write-activity'])
    elif args.command=='activity':
        slabs=(args.slab_dir.resolve() if args.slab_dir else ROOT/'sample/slabs' if args.sample else default_slab)
        run([*mpi,build/'evaluate_slabs',*common,'--slab-dir',slabs,'--be-dir',output/'be','--run-activity-dir',output/'activity_shards'])
    else:
        run([*mpi,build/'evaluate_random',*common,'--run-activity-dir',output/'random_activity_shards'])
if __name__=='__main__':main()

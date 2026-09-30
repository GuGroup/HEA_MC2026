#!/usr/bin/env python3
"""Recompute Study 2/3 log-activity metrics and selections from activity shards."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import selection_core as core

def read_shard(path):
    if path.suffix=='.f32':
        with path.open('rb') as f:meta=json.loads(f.read(4096).rstrip(b'\0'))
        offset=meta['data_offset_bytes']
    else:meta=json.loads(Path(str(path)+'.json').read_text());offset=0
    return meta,np.memmap(path,dtype='<f4',mode='r',offset=offset,shape=tuple(meta['shape']))

def best(points):
    clean=points.copy();terms=[]
    for key,maximum in [('tau',True),('mse',False),('crps',False)]:
        v=clean[key].to_numpy();lo,hi=v.min(),v.max();unit=np.zeros_like(v) if np.isclose(lo,hi) else (v-lo)/(hi-lo);terms.append(1-unit if maximum else unit)
    clean['trial_ideal_distance']=np.sqrt(sum(x*x for x in terms))
    return clean.sort_values(['trial_ideal_distance','tau','mse','crps','trial'],ascending=[True,False,True,True,True],kind='stable').iloc[0]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--cemc',type=Path,required=True);p.add_argument('--random',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-shards',type=int)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    info=json.loads((root/'cases.json').read_text())[a.dataset]
    if info['system']=='Study1':p.error('Use the separate Study1/run.py score command for Gaussian CRPS.')
    template=pd.read_csv(root/'analysis_data'/a.dataset/'most_probable_activity.csv');obs=template.experimental_activity.to_numpy()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);cache=out/'mean_log_cache';cache.mkdir(exist_ok=True)
    rows={};paths_by_trial={};random_logs={}
    for method,directory in [('cemc',a.cemc),('random',a.random)]:
        paths=sorted(directory.glob('*.bin')) or sorted(directory.glob('*.f32'))
        if a.max_shards:paths=paths[:a.max_shards]
        if not paths:raise ValueError(f'No activity shards in {directory}')
        frames=[]
        for path in paths:
            meta,raw=read_shard(path);data=np.asarray(raw,dtype=float)
            if np.any(data<=0) or not np.isfinite(data).all():raise ValueError('Activities must be finite and positive')
            logs=np.log(data).mean(axis=2);first=int(meta['first_trial']);n=len(logs)
            if method=='cemc':
                logs=logs.transpose(0,2,1);np.save(cache/f'{first:05d}.npy',logs)
                for j in range(n):paths_by_trial[first+j]=(first,j)
                nt=logs.shape[1];pred=core.scale_minus1_0(logs).reshape(-1,len(obs));tau,mse,crps=core.metrics_batch(pred,obs)
                frames.append(pd.DataFrame({'trial':np.repeat(np.arange(first,first+n),nt),'temp_idx':np.tile(np.arange(nt),n),'temperature':np.tile(meta['temperatures_K'],n),'tau':tau,'mse':mse,'crps':crps,'n_compositions':len(obs)}))
            else:
                for j in range(n):random_logs[first+j]=logs[j]
                tau,mse,crps=core.metrics_batch(core.scale_minus1_0(logs),obs)
                frames.append(pd.DataFrame({'trial':np.arange(first,first+n),'tau':tau,'mse':mse,'crps':crps,'n_compositions':len(obs)}))
        rows[method]=pd.concat(frames,ignore_index=True)
    rows['cemc'].to_csv(out/'cemc_all_temperatures.csv.gz',index=False)
    rows['random'].to_csv(out/'random_metrics_by_trial.csv',index=False)
    core.N_TRIALS=rows['cemc'].trial.nunique()
    cbest=core.select_best_temperature(rows['cemc']);cbest.to_csv(out/'cemc_best_temperature_by_trial.csv',index=False)
    cpoints,cbins,csel,cmeta=core.kde_assign(cbest,require_positive_tau=info['system']=='Pt')
    rpoints,rbins,rsel,rmeta=core.kde_assign(rows['random'],require_positive_tau=info['system']=='Pt')
    for prefix,frame,bins in [('cemc',cpoints,cbins),('random',rpoints,rbins)]:
        frame.to_csv(out/f'{prefix}_10000_trial_points_with_kde_probability.csv',index=False);bins.to_csv(out/f'{prefix}_3d_histogram_kde_bins.csv',index=False)
    def selection(c,r):
        result={}
        for prefix,selected in [('cemc',c),('random',r)]:
            for field in ['trial','temp_idx','temperature','tau','mse','crps']:
                if field in selected:result[prefix+'_'+field]=selected[field]
        return result
    for name,c,r in [('most_probable',csel,rsel),('best',best(cbest),best(rows['random']))]:
        first,j=paths_by_trial[int(c.trial)];clog=np.load(cache/f'{first:05d}.npy',mmap_mode='r')[j,int(c.temp_idx)];rlog=random_logs[int(r.trial)]
        table=template.copy();table['cemc_activity']=core.scale_minus1_0(clog);table['random_activity']=core.scale_minus1_0(rlog);table['cemc_mean_ln_activity']=clog;table['random_mean_ln_activity']=rlog
        table.to_csv(out/(name+'_activity.csv'),index=False)
        pd.DataFrame([selection(c,r)]).to_csv(out/('selected_kde_representative_trials.csv' if name=='most_probable' else 'best_selection.csv'),index=False)
    print('Recomputed',core.N_TRIALS,'trials for',a.dataset)
if __name__=='__main__':main()

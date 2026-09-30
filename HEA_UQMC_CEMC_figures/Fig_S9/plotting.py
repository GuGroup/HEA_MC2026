"""Recreate publication panels directly from the bundled numerical data."""
from pathlib import Path
import json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont

def activity(case,data,out,best=False):
    mode='best' if best else 'most_probable'
    fig,axes=plt.subplots(1,3,figsize=(18.5,5.8))
    if case['system']=='Study1':
        a=np.load(data/(mode+'_grids.npz'))
        for ax,key,title in zip(axes,['experimental','cemc','random'],['Experiment','CEMC','Homogeneous']):
            scatter=ax.imshow(a[key],cmap='viridis_r',vmin=-1,vmax=0)
            ax.set_title(title,fontsize=28,pad=8);ax.axis('off')
    else:
        table=pd.read_csv(data/(mode+'_activity.csv'))
        x,y=('map_x','map_y') if case['system']=='Pt' else ('tsne_1','tsne_2')
        h='random_activity' if 'random_activity' in table else 'homogeneous_activity'
        for ax,key,title in zip(axes,['experimental_activity','cemc_activity',h],['Experiment','CEMC','Homogeneous']):
            scatter=ax.scatter(table[x],table[y],c=table[key],cmap='viridis_r',vmin=-1,vmax=0,s=55 if case['system']=='Pt' else 38,edgecolors='none' if case['system']=='Pt' else 'black',linewidths=0 if case['system']=='Pt' else .25)
            ax.set_title(title,fontsize=28);ax.set_aspect('equal');ax.axis('off')
    fig.subplots_adjust(left=.02,right=.87,bottom=.05,top=.91,wspace=.07)
    cax=fig.add_axes([.92,.16,.018,.68]);bar=fig.colorbar(scatter,cax=cax,ticks=np.linspace(-1,0,6))
    bar.set_label('Scaled HER activity' if case['system']=='Fe' else 'Scaled ORR activity',fontsize=18);bar.ax.tick_params(labelsize=18)
    path=out/'activity.png';fig.savefig(path,dpi=250,facecolor='white');plt.close(fig)
    return path

def recall(case,data,out):
    table=pd.read_csv(data/('recall_activity.csv' if case['system']=='Study1' else 'most_probable_activity.csv'))
    n=len(table);k=np.arange(1,int(np.floor(.60*n))+1);fractions=k/n
    order=[np.argsort(table[c].to_numpy(),kind='stable') for c in ['experimental_activity','cemc_activity','random_activity']]
    curves=[]
    for rank in order[1:]:
        counts=np.ceil(fractions*n).astype(int) if case['system']=='Study1' else k
        curves.append(np.array([len(set(order[0][:i]).intersection(rank[:i]))/i for i in counts]))
    ref=pd.read_csv(data/'reference_recall.csv')
    for computed,col in zip(curves,['cemc_recall','random_recall']):np.testing.assert_allclose(computed,ref[col],rtol=0,atol=1e-12)
    frame=pd.DataFrame({'top_fraction_percent':fractions*100,'n_compositions':k,'cemc_recall':curves[0],'random_recall':curves[1],'baseline':fractions})
    frame.to_csv(out/'recall.csv',index=False)
    with plt.rc_context({'font.size':18,'axes.labelsize':22,'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize':22}):
        fig,ax=plt.subplots(figsize=(8.5,6.8))
        for curve,color,label in zip(curves,['#0072B2','#D55E00'],['CEMC','Homogeneous']):
            x=fractions*100;y=curve
            if case['system']=='Pt':x=np.r_[0,x,60];y=np.r_[0,y,y[-1]]
            ax.plot(x,y,color=color,lw=2.7,label=label)
        ax.plot([0,60],[0,.6],'k--',lw=2.2,label='Baseline')
        ax.set(xlim=(0,60),ylim=(0,1),xlabel='Top fraction tested (%)',ylabel='Recall of experimental active region')
        ax.set_xticks(np.arange(0,61,10));ax.set_yticks(np.arange(0,1.01,.2));ax.grid(alpha=.22,lw=.8);ax.legend(frameon=False,loc='upper left')
        fig.subplots_adjust(left=.18,right=.97,bottom=.16,top=.97)
        path=out/'recall.png';fig.savefig(path,dpi=250);plt.close(fig)
    return path

def histogram(case,data,out):
    if case['system']=='Pt':
        import hist_pt as h
        h.plot_hist(data,out,case['ml'])
    elif case['system']=='Fe':
        import hist_fe as h
        h.plot_hist(data,out,case['facet'],case['site'])
    else:
        import hist_study1 as h
        h.SOURCE_DIR=data;h.OUTDIR=out;h.plot_result()
    return next(out.glob('*.png'))

def combine(paths,labels,out,cols=1):
    thumb_width=2000 if cols==1 else 950
    tiles=[]
    for path,label in zip(paths,labels):
        im=Image.open(path).convert('RGB');im=im.resize((thumb_width,round(im.height*thumb_width/im.width)),Image.Resampling.LANCZOS)
        tile=Image.new('RGB',(thumb_width,im.height+65),'white');tile.paste(im,(0,65))
        fontpath=matplotlib.font_manager.findfont('DejaVu Sans')
        ImageDraw.Draw(tile).text((20,8),label,font=ImageFont.truetype(fontpath,36),fill='black');tiles.append(tile)
    height=max(x.height for x in tiles);canvas=Image.new('RGB',(thumb_width*cols,height*math.ceil(len(tiles)/cols)),'white')
    for i,im in enumerate(tiles):canvas.paste(im,((i%cols)*thumb_width,(i//cols)*height))
    canvas.save(out)

def main(folder):
    folder=Path(folder);config=json.loads((folder/'figure.json').read_text());out=folder/'outputs';out.mkdir(exist_ok=True)
    paths=[];labels=[]
    for panel in config['panels']:
        case=panel['case'];key=panel['dataset'];dest=out/key;dest.mkdir(exist_ok=True);data=folder/'data'/key
        func={'histogram':histogram,'activity':activity,'best_activity':lambda c,d,o:activity(c,d,o,True),'recall':recall}[config['type']]
        paths.append(func(case,data,dest));labels.append(panel['label'])
    if config['type']=='best_activity':
        for i in range(0,len(paths),6):combine(paths[i:i+6],labels[i:i+6],out/f"{config['figure']}_page_{i//6+1}.png")
    else:combine(paths,labels,out/(config['figure']+'.png'),cols=3 if config['type']=='recall' else 1)
    (out/'validation.json').write_text(json.dumps({'figure':config['figure'],'panels_rendered':len(paths),'recall_matches_reference':config['type']=='recall'},indent=2))
    print(config['figure'],'rendered',len(paths),'panels',flush=True)

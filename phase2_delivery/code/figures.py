from topology import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
BLUE,GOLD,ROSE,GRAY='#234e70','#bb7917','#c05440','#9199a0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
FIG=P2/'figures';FIG.mkdir(exist_ok=True)
SOURCE='official_ganguli_pretrained';A=P2/'analysis'/SOURCE
def export(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=190,bbox_inches='tight');fig.savefig(FIG/f'{name}.svg',bbox_inches='tight');plt.close(fig)

def evidence():
    fig,axs=plt.subplots(2,4,figsize=(17,8),layout='constrained')
    for row,split in enumerate(['validation','confirmation']):
        r=json.loads((A/f'v3/topology/stationary_{split}_test.json').read_text());o=r['observed']
        d=np.load(A/f'v3/topology/stationary_{split}_F47_n768.npz');T=o['threshold'];scale=o['distance_scale']
        a=axs[row,0];offset=0
        for dim,color,top_n in [(1,BLUE,8),(2,GOLD,8)]:
            bars=d[f'H{dim}'];life=np.minimum(bars[:,1],T)-bars[:,0];order=np.argsort(life)[::-1][:top_n]
            for j,ind in enumerate(order):a.plot([bars[ind,0]/scale,min(bars[ind,1],T)/scale],[offset+j]*2,color=color,lw=3 if j<(2 if dim==1 else 1) else 1,alpha=1 if j<(2 if dim==1 else 1) else .45)
            offset+=top_n+2
        a.set(title=f'{split}: raw persistence bars',xlabel='filtration / median distance',yticks=[3.5,13.5],yticklabels=['H1','H2'])
        a.text(.02,.98,'H0: one surviving connected component',transform=a.transAxes,va='top',fontsize=8)
        a=axs[row,1]
        for dim,color in [(0,GRAY),(1,BLUE),(2,GOLD)]:
            bars=d[f'H{dim}'];finite=np.isfinite(bars[:,1]);a.scatter(bars[finite,0]/scale,bars[finite,1]/scale,s=8,color=color,alpha=.45,label=f'H{dim}')
        a.plot([0,1],[0,1],color=GRAY,lw=.8);a.set(xlabel='birth / median distance',ylabel='death / median distance',title='Raw Euclidean space, field 47',xlim=(-.02,1.02),ylim=(-.02,1.06));a.legend(fontsize=8)
        a=axs[row,2]
        longest=max(o['beta121_intervals'],key=lambda z:z[1]-z[0])
        eps=np.linspace(max(0,longest[0]/scale-.03),min(1,longest[1]/scale+.03),900)
        full_eps=np.linspace(0,1,1500);full_counts=[]
        for dim,color in [(0,GRAY),(1,BLUE),(2,GOLD)]:
            bars=d[f'H{dim}'];counts=np.array([np.sum((bars[:,0]<=e*scale)&(bars[:,1]>e*scale)) for e in eps])
            full_counts.append(np.array([np.sum((bars[:,0]<=e*scale)&(bars[:,1]>e*scale)) for e in full_eps]))
            a.plot(eps,counts,color=color,label=f'beta{dim}')
        np.savez(FIG/f'full_Betti_curves_{split}.npz',normalized_filtration=full_eps,Betti=np.array(full_counts),distance_scale=scale)
        for left,right in o['beta121_intervals']:a.axvspan(left/scale,right/scale,color=GOLD,alpha=.12)
        a.set(xlabel='normalized filtration (zoom)',ylabel='Betti number',title='Persistent (1, 2, 1) interval');a.legend(fontsize=8);a.set_ylim(bottom=0)
        a=axs[row,3];precision=json.loads((A/f'v3/precision_audit/{split}.json').read_text())
        a.hist(precision['scores'],bins=np.linspace(-.005,.40,42),color=GRAY,edgecolor='white');a.axvline(o['score'],color=ROSE,lw=2)
        a.set(title=f'999 fresh randomizations: {split}',xlabel='joint torus statistic',ylabel='count',xlim=(-.01,.41))
        a.text(.06,.84,f"observed={o['score']:.3f}\np_MC={precision['p_plus_one']:.3f}\n10x p_MC={precision['conservative_times10_p']:.3f}",transform=a.transAxes,ha='left',va='top')
    fig.suptitle('Raw 483-unit subpopulation: stationary step 20\nTwo independent sets of 8192 positions; 768 landmarks; 1024-landmark and field-2 checks agree',fontsize=14)
    export(fig,'raw_torus_evidence')

def failures_and_resolution():
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    points=[]
    for split in ['validation','confirmation']:
        r=json.loads((A/f'v2/topology/stationary_{split}_observed.json').read_text())
        points.append(dict(**r['observed'],split=split,condition='stationary'))
        for s in r['sensitivity']:
            if s['metric']=='euclidean' and s['field']==47:points.append(dict(**s,split=split,condition='stationary'))
    points+=json.loads((A/'v2/resolution_diagnostics/results.json').read_text())
    for split,color in [('validation',BLUE),('confirmation',GOLD)]:
        p=sorted([r for r in points if r['split']==split and r['condition']=='stationary'],key=lambda r:r['landmarks'])
        axs[0].plot([r['landmarks'] for r in p],[r['H1_second_to_third_ratio'] for r in p],'-o',color=color,label=split)
        axs[1].plot([r['landmarks'] for r in p],[r['longest_beta121_interval_normalized'] for r in p],'-o',color=color,label=split)
    axs[0].axhline(2,color=GRAY,ls='--');axs[1].axhline(.05,color=GRAY,ls='--')
    axs[0].set(xlabel='landmark count',ylabel='H1 second / third lifetime',title='Low-resolution failure is retained')
    axs[1].set(xlabel='landmark count',ylabel='normalized (1,2,1) interval',title='Diagnostic convergence (not final confirmation)')
    axs[0].legend();axs[1].legend()
    dims=json.loads((A/'geometry_support/IID_intrinsic_dimensions.json').read_text())
    x=np.arange(4);vals=[r['TwoNN_trimmed'] for r in dims]
    axs[2].bar(x,vals,color=[BLUE,ROSE,BLUE,ROSE]);axs[2].axhline(2,color=GRAY,ls='--')
    axs[2].set(xticks=x,xticklabels=['static\nvalidation','motion\nvalidation','static\nconfirmation','motion\nconfirmation'],ylabel='TwoNN estimate (IID sample)',title='Raw movement states have extra dimensions')
    export(fig,'sampling_failures_and_intrinsic_dimension')

def phenotypes():
    maps=np.load(A/'discovery_ratemaps.npy');rot=np.load(A/'rotation_clusters.npz');cells=rot['cell_ids'];labels=rot['labels'];wanted=[5,16,17]
    fig,axs=plt.subplots(3,7,figsize=(14,6.5),layout='constrained');rng=np.random.default_rng(73250)
    for row,label in enumerate(wanted):
        group=cells[labels==label];examples=rng.choice(group,6,replace=False)
        for col,i in enumerate(examples):
            a=axs[row,col];a.imshow(maps[i],origin='lower',extent=[0,2.2,0,2.2],cmap='cividis',aspect='equal');a.set(title=f'neuron {i}',xticks=[],yticks=[])
            if col==0:a.set_ylabel(f'cluster {label}\nindividual rate scales')
        feat=np.load(A/'cell_features.npz');angles=np.degrees(feat['orientation_rad'][labels==label]);axs[row,6].hist(angles,bins=np.linspace(0,180,37),color=[BLUE,GOLD,ROSE][row]);axs[row,6].set(title='discovery orientations',xlabel='degrees',xlim=(0,180))
    fig.suptitle('Discovery-only rotational clusters; seeded random ratemap examples\nNo heldout PH or image-based hand-picking used for the 483-cell mask',fontsize=13)
    export(fig,'discovery_band_clusters')

def time_figures():
    d=np.load(A/'timewise/fixed_PCA_coordinates.npz');xyz=d['xyz'];condition=d['condition'];steps=[0,1,5,10,15,20]
    lo=xyz.min((0,1));hi=xyz.max((0,1));fig=plt.figure(figsize=(18,7));fig.subplots_adjust(left=.02,right=.98,bottom=.08,top=.83,wspace=.04,hspace=.15)
    for row,c in enumerate(['stationary','random_foraging']):
        for col,t in enumerate(steps):
            ax=fig.add_subplot(2,6,row*6+col+1,projection='3d');x=xyz[t,condition==c]
            ax.scatter(*x.T,s=2,alpha=.45,color=BLUE if row==0 else ROSE)
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),title=f'{c}\ns={t}',xlabel='PC1',ylabel='PC2',zlabel='PC3')
            ax.view_init(22,45)
    fig.suptitle(f"ONE fixed discovery PCA frame — projection only\n483-unit subset; first 3 PCs explain {100*d['variance_ratio'].sum():.1f}% discovery variance; full 4096-unit tensors retained",fontsize=14)
    export(fig,'matched_step_projection')
    changes=json.loads((A/'timewise/paired_state_and_ensemble_changes.json').read_text());fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for c,color in [('stationary',BLUE),('random_foraging',ROSE)]:
        rows=[r for r in changes if r['condition']==c]
        axs[0].plot([r['step'] for r in rows],[r['paired_mean_state_displacement'] for r in rows],label=c,color=color)
        axs[1].plot([r['step'] for r in rows],[r['mean_reference_distance_over_reference_scale'] for r in rows],label=c,color=color)
        axs[2].plot([r['step'] for r in rows],[r['relative_pairwise_distance_change'] for r in rows],label=c,color=color)
    for ax in axs:ax.set_xlabel('recurrent step');ax.legend(fontsize=8);ax.grid(alpha=.15)
    axs[0].set(title='Paired state displacement',ylabel='activation units');axs[1].set(title='Distance to stationary reference cloud',ylabel='distance / reference median')
    axs[2].set(title='Ensemble distance-pattern change',ylabel='relative Frobenius difference')
    fig.suptitle('State motion and ensemble changes are distinct; these curves do NOT establish manifold deformation',fontsize=12)
    export(fig,'state_motion_vs_ensemble_change')
    traces=[];buttons=[]
    for t in steps:
        for c,color in [('stationary',BLUE),('random_foraging',ROSE)]:
            mask=condition==c;z=xyz[t,mask]
            traces.append(go.Scatter3d(x=z[:,0],y=z[:,1],z=z[:,2],mode='markers',name=c,visible=t==20,
                       ids=[str(i) for i in d['sample_id'][mask]],customdata=d['sample_id'][mask].tolist(),
                       marker=dict(size=2,color=color,opacity=.5),hovertemplate='sample %{customdata}<extra></extra>'))
        visible=[False]*(2*len(steps));j=steps.index(t);visible[j*2:j*2+2]=[True,True]
        buttons.append(dict(label=f's={t}',method='update',args=[{'visible':visible},{'title.text':f'Recurrent step {t} — fixed PCA projection only'}]))
    scene={axis:dict(title=f'PC{i+1}',range=[float(lo[i]),float(hi[i])]) for i,axis in enumerate(['xaxis','yaxis','zaxis'])}
    scene.update(aspectmode='cube',uirevision='fixed',camera=dict(eye=dict(x=1.5,y=1.5,z=1.0)))
    fig=go.Figure(traces);fig.update_layout(title=dict(text='Recurrent step 20 — fixed PCA projection only',y=.98,x=.05),height=760,
      template='plotly_white',scene=scene,updatemenus=[dict(buttons=buttons,type='buttons',direction='right',active=5,y=1.07)],margin=dict(t=150,l=10,r=10,b=10))
    body=fig.to_html(full_html=False,include_plotlyjs=True)
    intro='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>RNN phase-two matched states</title><body style="font:16px system-ui;margin:24px"><h1>固定坐标系中的逐步神经状态</h1><p>每一步显示同一组 2048 个样本、相同的 483 个单元、同一 PCA 均值和基。完整 4096 单元原始激活另存为 H[21,2048,4096]。图形仅为投影。</p><p>静止与随机运动使用配对初始位置。点云是固定步骤的样本集合；下方线条是单条轨迹。验证通过的是高分辨率静止终态子群，不代表全部神经元或随机运动状态都是二维环面。</p>'''
    traj=[]
    for c,color in [('stationary',BLUE),('random_foraging',ROSE)]:
        ix=np.flatnonzero(condition==c)[512];z=xyz[:,ix]
        traj.append(go.Scatter3d(x=z[:,0],y=z[:,1],z=z[:,2],mode='lines+markers',name=f'{c}: sample {d["sample_id"][ix]}',line=dict(color=color,width=4),marker=dict(size=3)))
    f2=go.Figure(traj);f2.update_layout(scene=scene,title='Single-sample state motion, same coordinate frame',template='plotly_white',height=550)
    (FIG/'matched_step_projection.html').write_text(intro+body+'<h2>单条轨迹</h2>'+f2.to_html(full_html=False,include_plotlyjs=False)+'</body></html>',encoding='utf8')

def main():evidence();failures_and_resolution();phenotypes();time_figures();print('Research figures rendered',flush=True)
if __name__=='__main__':main()

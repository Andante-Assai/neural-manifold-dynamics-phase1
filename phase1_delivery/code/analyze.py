"""Raw-state QC and ONE pooled PCA frame; no per-step UMAP, no topology presumption."""
from phase1 import *
import h5py
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

COLORS=['#234e70','#bb7917','#c05440','#59753d']

def read_subset(manifest, label, ids, keys):
    mapping={int(x):i for i,x in enumerate(ids)};out={}
    for row in manifest['files']:
        if '/'+label+'/' not in row['path']:continue
        with h5py.File(ROOT/row['path']) as f:
            loc=[(j,mapping[int(s)]) for j,s in enumerate(f['sample_id'][...]) if int(s) in mapping]
            if not loc:continue
            src,dst=map(np.array,zip(*loc))
            for k in keys:
                shape=f[k].shape
                if k not in out:out[k]=np.empty((shape[0],len(ids))+shape[2:],dtype=f[k].dtype)
                out[k][:,dst]=f[k][:,src]
    return out

def intrinsic(x):
    # Full-neuron Euclidean distances; remove exact duplicate vectors before TwoNN.
    x=np.unique(x,axis=0).astype(np.float64)
    if len(x)<16:return dict(n_unique=len(x),twoNN=None,participation_ratio=None)
    d=cdist(x,x);np.fill_diagonal(d,np.inf)
    near=np.partition(d,1,axis=1)[:,:2];near.sort(axis=1)
    ratios=near[:,1]/np.maximum(near[:,0],1e-20)
    logs=np.log(ratios[near[:,0]>1e-9]);logs=np.sort(logs)
    # Trim extreme 10% at each end; only a scale- and sampling-dependent screen.
    trimmed=logs[int(.1*len(logs)):int(.9*len(logs))]
    est=1/trimmed.mean() if len(trimmed) and trimmed.mean()>1e-10 else None
    z=x-x.mean(0);gram=z@z.T/(len(x)-1)
    ev=np.linalg.eigvalsh(gram);ev=np.maximum(ev,0)
    pr=(ev.sum()**2/(ev@ev)) if ev@ev>0 else 0
    return dict(n_unique=len(x),twoNN=float(est) if est is not None else None,participation_ratio=float(pr))

def finite_pairs(dgm):
    return [[float(a),float(b)] for a,b in dgm if np.isfinite(b)]

def main_analysis(run):
    out=ROOT/'datasets'/run;manifest=json.loads((out/'manifest.json').read_text())
    dest=ROOT/'analysis'/run;dest.mkdir(parents=True,exist_ok=True)
    n=manifest['per_condition'];sample=np.linspace(0,n-1,min(128,n),dtype=int)
    ids=np.concatenate([sample+c*n for c in range(4)])
    core=read_subset(manifest,'core_T20',ids,['H','position','error_m'])
    ext_ids=np.concatenate([row['sample_ids'] for row in manifest['files'] if '/exploratory_T160/' in row['path']])
    ext=read_subset(manifest,'exploratory_T160',ext_ids,['H','position','error_m'])
    cond=np.repeat(np.arange(4),len(sample));ext_cond=np.repeat(np.arange(4),manifest['ext_per_condition'])
    pca=PCA(n_components=32,svd_solver='randomized',random_state=712)
    fit=core['H'].reshape(-1,core['H'].shape[-1]);pca.fit(fit)
    z=pca.transform(fit)[:,:3].reshape(21,len(ids),3)
    zext=pca.transform(ext['H'].reshape(-1,ext['H'].shape[-1]))[:,:3].reshape(161,len(ext_ids),3)
    np.savez(dest/'pooled_pca.npz',mean=pca.mean_,components=pca.components_,
             explained_variance_ratio=pca.explained_variance_ratio_,fit_sample_ids=ids,fit_steps=np.arange(21),
             core_xyz=z,extension_xyz=zext,core_sample_ids=ids,extension_sample_ids=ext_ids)
    # All raw files contribute to task performance and coverage checks.
    stats=[];coverage=[]
    for label,T in [('core_T20',20),('exploratory_T160',160)]:
        sums=np.zeros((4,T+1));counts=np.zeros(4,int);bases=np.zeros_like(sums)
        grids=np.zeros((4,T+1,20,20),int);outside=np.zeros((4,T+1),int)
        for row in manifest['files']:
            if '/'+label+'/' not in row['path']:continue
            with h5py.File(ROOT/row['path']) as f:
                pos=f['position'][...];err=f['error_m'][...];cc=f['condition'][...]
                # Author top-3 decoded initial position is the task no-motion baseline.
                initial=f['target_decoded_position'][0]
                base=np.linalg.norm(initial[None]-pos,axis=-1)
                for c in range(4):
                    sel=cc==c;sums[c]+=err[:,sel].sum(1);bases[c]+=base[:,sel].sum(1);counts[c]+=sel.sum()
                    for t in range(T+1):
                        hist,_,_=np.histogram2d(pos[t,sel,0],pos[t,sel,1],bins=20,range=[[0,BOX],[0,BOX]])
                        grids[c,t]+=hist.astype(int)
                    outside[c]+=((pos[:,sel]<0)|(pos[:,sel]>BOX)).any(-1).sum(1)
        for c in range(4):
            stats.append(dict(dataset=label,condition=CONDITIONS[c],n=int(counts[c]),
               error_by_step_m=(sums[c]/counts[c]).tolist(),no_motion_error_by_step_m=(bases[c]/counts[c]).tolist(),
               mean_error_steps_1_to_20_m=float(sums[c,1:21].mean()/counts[c]),
               error_final_m=float(sums[c,-1]/counts[c]),
               outside_box_by_step=outside[c].tolist(),
               note='t0 decoder readout was not trained; random walker can slightly overshoot hard walls as in author code'))
            for t in ([0,1,5,10,20] if T==20 else [40,80,160]):
                coverage.append(dict(dataset=label,condition=CONDITIONS[c],step=t,
                     occupied_20x20_fraction=float(np.mean(grids[c,t]>0)),outside_count=int(outside[c,t])))
    dump(dest/'task_performance.json',stats);dump(dest/'coverage.json',coverage)
    quality=[]
    for label,states,cids in [('core_T20',core['H'],cond),('exploratory_T160',ext['H'],ext_cond)]:
        for t in ([0,1,5,10,20] if label=='core_T20' else [40,80,160]):
            for c in range(4):
                x=states[t,cids==c];st=x.std(0);norm=np.linalg.norm(x,axis=-1)
                r=intrinsic(x)
                r.update(dataset=label,condition=CONDITIONS[c],step=t,
                   n_sampled=len(x),zero_fraction=float((x==0).mean()),
                   nonvarying_neuron_fraction=float((st<1e-8).mean()),
                   mean_state_norm=float(norm.mean()),std_state_norm=float(norm.std()),
                   total_neuron_variance=float(np.sum(st**2)),finite=bool(np.isfinite(x).all()))
                quality.append(r)
    dump(dest/'activation_quality_and_dimension.json',quality)
    motion=[]
    for t in range(1,21):
        delta=core['H'][t]-core['H'][t-1]
        for c in range(4):
            s=cond==c
            motion.append(dict(step=t,condition=CONDITIONS[c],
              paired_mean_delta_norm=float(np.linalg.norm(delta[s],axis=-1).mean()),
              ensemble_centroid_shift_norm=float(np.linalg.norm(delta[s].mean(0))),
              note='paired state displacement and centroid shift in all raw neurons; neither establishes manifold deformation'))
    dump(dest/'paired_step_changes.json',motion)
    # Preliminary PH in ORIGINAL full-neuron Euclidean space, fixed t, no unit selection.
    from ripser import ripser
    ph=[];diagrams={}
    for t in [5,20]:
        x=core['H'][t,cond==0].astype(np.float64)
        x=x/np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=-1)))
        dgms=ripser(cdist(x,x),distance_matrix=True,maxdim=2)['dgms']
        ph.append(dict(condition='stationary',step=t,n=len(x),neurons=x.shape[1],metric='raw full-neuron Euclidean / RMS radius',
                       H0=finite_pairs(dgms[0]),H1=finite_pairs(dgms[1]),H2=finite_pairs(dgms[2])))
        for k,d in enumerate(dgms):diagrams[f'step{t}_H{k}']=d
    # Same-size positive torus and simply connected plane controls; pipeline validation only.
    rr=np.random.default_rng(842);nn=len(sample)
    a,b=rr.uniform(0,2*np.pi,(2,nn))
    torus=np.stack([(2+np.cos(b))*np.cos(a),(2+np.cos(b))*np.sin(a),np.sin(b)],-1)
    plane=rr.uniform(-1,1,(nn,2))
    controls=[]
    for name,x in [('synthetic_torus',torus),('synthetic_plane',plane)]:
        x=x/np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=-1)))
        dg=ripser(x,maxdim=2)['dgms']
        controls.append(dict(name=name,n=nn,H1=finite_pairs(dg[1]),H2=finite_pairs(dg[2])))
    np.savez(dest/'persistence_diagrams.npz',**diagrams)
    dump(dest/'preliminary_topology.json',dict(raw_fixed_step_screens=ph,synthetic_controls=controls,
         conclusion='No torus acceptance test has been met. Short local training, full-population mixtures, limited subsamples and finite-box boundary can change topology. Do not infer a torus from a 3D projection or a few persistent bars.',
         limitation='128 stratified samples per condition, one local-training seed, no statistical null ensemble or neuron-module identification; exploratory screen only'))
    # A scientific static artifact plus a self-contained rotating 3D view, same PCA basis.
    selected=[0,1,5,20];lo=z.min((0,1));hi=z.max((0,1));pad=np.maximum((hi-lo)*.05,.01)
    fig=plt.figure(figsize=(13,10))
    fig.subplots_adjust(left=.04,right=.96,bottom=.09,top=.84,hspace=.25,wspace=.12)
    for j,t in enumerate(selected):
        ax=fig.add_subplot(2,2,j+1,projection='3d')
        for c in range(4):
            x=z[t,cond==c];ax.scatter(*x.T,s=4,alpha=.65,color=COLORS[c],label=CONDITIONS[c])
        ax.set(xlabel='pooled PC1',ylabel='pooled PC2',zlabel='pooled PC3',title=f'Recurrent step {t}',
               xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]),zlim=(lo[2]-pad[2],hi[2]+pad[2]))
        ax.view_init(elev=23,azim=40)
    fig.suptitle(f'ONE shared PCA coordinate system — projection only\n{run}; first 3 PCs explain {100*pca.explained_variance_ratio_[:3].sum():.1f}% pooled variance',fontsize=15)
    fig.legend(CONDITIONS,loc='lower center',ncol=4)
    fig.savefig(dest/'shared_pca_3d.png',dpi=180);plt.close(fig)
    traces=[];buttons=[]
    display_steps=[0,1,5,10,20,40,80,160]
    for t in display_steps:
        # Exactly the SAME paired extension sample IDs at every displayed step.
        xyz,cids,sids,pp=(zext,ext_cond,ext_ids,ext['position'])
        for c in range(4):
            mask=cids==c;xx=xyz[t,mask]
            traces.append(go.Scatter3d(x=xx[:,0],y=xx[:,1],z=xx[:,2],mode='markers',name=CONDITIONS[c],
                visible=t==20,marker=dict(size=2,color=COLORS[c],opacity=.65),
                ids=[str(s) for s in sids[mask]],
                customdata=np.column_stack([sids[mask],pp[t,mask]]).tolist(),
                hovertemplate='sample %{customdata[0]}<br>x=%{customdata[1]:.3f}m, y=%{customdata[2]:.3f}m<extra></extra>'))
        visible=[False]*(4*len(display_steps));visible[4*display_steps.index(t):4*display_steps.index(t)+4]=[True]*4
        buttons.append(dict(label=f't={t}'+(' / exploratory' if t>20 else ''),method='update',
                           args=[{'visible':visible},{'title.text':f'Step {t} — shared PCA projection only'+(' — BEYOND training horizon' if t>20 else '')}]))
    fig=go.Figure(traces)
    xyz=np.concatenate([z.reshape(-1,3),zext.reshape(-1,3)]);lo=xyz.min(0);hi=xyz.max(0)
    scene={axis:dict(title=f'pooled PC{i+1}',range=[float(lo[i]),float(hi[i])]) for i,axis in enumerate(['xaxis','yaxis','zaxis'])}
    scene.update(aspectmode='cube',camera=dict(eye=dict(x=1.5,y=1.5,z=1.1)),uirevision='shared-frame')
    fig.update_layout(title=dict(text='Step 20 — shared PCA projection only',y=.98,yanchor='top',x=.05),
          template='plotly_white',height=760,
          scene=scene,updatemenus=[dict(buttons=buttons,x=0,y=1.07,direction='right',type='buttons',active=4)],
          margin=dict(t=160,l=10,r=10,b=10))
    body=fig.to_html(full_html=False,include_plotlyjs=True)
    prefix=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>RNN shared projection</title><body style="font:16px system-ui;margin:24px;color:#222"><h1>完整神经状态的统一坐标系投影</h1><p>模型：{run}。同一组 PCA 均值与基向量用于全部步骤；前三轴只解释 {100*pca.explained_variance_ratio_[:3].sum():.1f}% 的合并方差。点云显示的是固定步骤的不同样本，不代表单条轨迹。所有按钮显示同一组 {len(ext_ids)} 个 sample_id，完整原始核心数据另有 {4*n} 条轨迹。</p><p>0–20：训练时域内；40 / 80 / 160：探索性时域外。各步使用固定轴范围、纵横比，可旋转、缩放并查看 sample_id。形状和视觉运动仅为投影，不证明环面或真实流形形变。</p>'''
    # Separate single-trajectory view; no relabeling of ensemble as trajectory.
    traj=[]
    for c in range(4):
        k=c*manifest['ext_per_condition']+manifest['ext_per_condition']//2;x=zext[:,k]
        traj.append(go.Scatter3d(x=x[:,0],y=x[:,1],z=x[:,2],mode='lines+markers',
          name=f'{CONDITIONS[c]} / sample {ext_ids[k]}',marker=dict(size=2),line=dict(color=COLORS[c],width=3)))
    tf=go.Figure(traj);tf.update_layout(template='plotly_white',height=550,scene=scene,
                 title='单条轨迹的神经状态运动：0–160 步，20 步以外为探索')
    (dest/'shared_pca_3d.html').write_text(prefix+body+'<h2>轨迹视图</h2>'+tf.to_html(full_html=False,include_plotlyjs=False)+'</body></html>',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for c,r in enumerate(stats[:4]):
        ax.plot(range(21),r['error_by_step_m'],label=CONDITIONS[c],color=COLORS[c])
    ax.set(xlabel='recurrent step',ylabel='mean decoding error (metres)',title='Controlled task performance — t0 readout was not trained')
    ax.legend();ax.grid(alpha=.2);fig.savefig(dest/'task_error.png',dpi=170);plt.close(fig)
    dump(dest/'projection_metadata.json',dict(method='one centered pooled PCA, no neuron standardization',
         fit_steps=list(range(21)),fit_samples=len(ids),total_fit_states=len(fit),
         variance_ratio_3D=float(pca.explained_variance_ratio_[:3].sum()),
         fit_only_training_horizon=True,extended_steps_transformed_in_same_basis=True,
         displayed_points_per_condition=len(sample),neuron_subset=False,
         interactive_paired_samples=len(ext_ids),interactive_same_sample_ids_at_all_steps=True,
         topology_metric='raw full-neuron Euclidean, no PCA for PH'))
    print('Analysis complete',dest,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();main_analysis(a.run)

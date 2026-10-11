"""Discovery-only cell identification. Never reads topology results or heldout maps to select units."""
from common import *
import argparse,itertools
from scipy import ndimage,signal
from sklearn.cluster import DBSCAN,KMeans
sys.path.insert(0,str(P2/'vendor/schoyen'))
from scores import GridScorer

def selected(f,source,split,primary=False):
    if source=='existing_full4096_fast':
        code={'discovery':0,'validation':1,'confirmation':2}[split]
        m=f['split_code'][...]==code
        if primary:m&=f['condition'][...]==3
        return m
    return np.ones(len(f['sample_id']),bool) if f.attrs['split']==split else np.zeros(len(f['sample_id']),bool)

def maps_and_states(source,split):
    manifest=json.loads((P2/'data'/source/'manifest.json').read_text())
    sums=np.zeros((1024,4096),np.float64);counts=np.zeros(1024,np.int64)
    states=[];ids=[]
    for r in manifest['files']:
        if source!='existing_full4096_fast' and r['split']!=split:continue
        with h5py.File(P2/r['path']) as f:
            mask=selected(f,source,split);primary=selected(f,source,split,primary=split!='discovery')
            if not mask.any():continue
            hsteps=f['H_steps'][...];hh=f['H'][...]
            pos=f['position'][...];end=hh[np.flatnonzero(hsteps==20)[0]]
            states.append(end[primary]);ids.append(f['sample_id'][...][primary])
            for t in range(11,21):
                h=hh[np.flatnonzero(hsteps==t)[0],mask]
                p=pos[t,mask];inside=((p>=0)&(p<=2.2)).all(-1)
                bins=np.minimum((p[inside]/2.2*32).astype(int),31)
                ix=bins[:,1]*32+bins[:,0]
                np.add.at(sums,ix,h[inside]);np.add.at(counts,ix,1)
    maps=sums/np.maximum(counts[:,None],1)
    missing=counts.reshape(32,32)==0
    if missing.any():
        nearest=ndimage.distance_transform_edt(missing,return_distances=False,return_indices=True)
        maps=maps.reshape(32,32,4096)[nearest[0],nearest[1]].reshape(1024,4096)
    return maps.T.reshape(4096,32,32).astype(np.float32),counts.reshape(32,32),np.concatenate(states),np.concatenate(ids)

def rotation_features(maps):
    n,size,_=maps.shape
    z=maps.astype(np.float64);z=(z-z.mean((1,2),keepdims=True))/np.maximum(z.std((1,2),keepdims=True),1e-15)
    sac=signal.fftconvolve(z,z[:,::-1,::-1],mode='same',axes=(1,2))
    yy,xx=np.ogrid[-size/2:size/2,-size/2:size/2];mask=xx*xx+yy*yy<=((size-1)/2)**2
    rows,cols=np.nonzero(mask);a=sac[:,rows,cols];a=a-a.mean(1,keepdims=True)
    denominator=np.sqrt((a*a).sum(1));scores=np.empty((n,360),np.float32)
    flat=sac.reshape(n,-1);center=(size-1)/2
    for j,angle in enumerate(np.linspace(-np.pi,np.pi,360)):
        cy=np.cos(angle)*(rows-center)+np.sin(angle)*(cols-center)+center
        cx=-np.sin(angle)*(rows-center)+np.cos(angle)*(cols-center)+center
        rr=np.floor(cy+.5).astype(int);cc=np.floor(cx+.5).astype(int)
        valid=(cy>=0)&(cy<=size-1)&(cx>=0)&(cx<=size-1)
        rr=np.clip(rr,0,size-1);cc=np.clip(cc,0,size-1)
        b=flat[:,rr*size+cc].copy();b[:,~valid]=0;b-=b.mean(1,keepdims=True)
        scores[:,j]=(a*b).sum(1)/np.maximum(denominator*np.sqrt((b*b).sum(1)),1e-15)
    return scores

def fourier_features(maps):
    # Hanning taper, zero padding fixed in advance; phases/orientations come only from discovery maps.
    n=len(maps);window=np.outer(np.hanning(32),np.hanning(32))
    z=(maps-maps.mean((1,2),keepdims=True))*window
    power=np.abs(np.fft.fftshift(np.fft.fft2(z,s=(128,128),axes=(1,2)),axes=(1,2)))**2
    freq=np.fft.fftshift(np.fft.fftfreq(128,d=1/32));yy,xx=np.meshgrid(freq,freq,indexing='ij')
    radius=np.hypot(xx,yy);angles=np.arctan2(yy,xx)%np.pi
    valid=(radius>=1.5)&(radius<=12)
    power[:,~valid]=0
    idx=power.reshape(n,-1).argmax(1);peak_r=radius.ravel()[idx];theta=angles.ravel()[idx]
    delta=np.abs((angles[None]-theta[:,None,None]+np.pi/2)%np.pi-np.pi/2)
    # Both antipodal lobes count; a three-wave grid gives ~1/3, a single-direction band higher.
    concentration=(power*(delta<=np.pi/12)).sum((1,2))/np.maximum(power.sum((1,2)),1e-20)
    return peak_r,theta,concentration

def identify(source):
    out=P2/'analysis'/source;out.mkdir(parents=True,exist_ok=True)
    maps,counts,states,ids=maps_and_states(source,'discovery')
    np.save(out/'discovery_ratemaps.npy',maps);np.save(out/'discovery_occupancy.npy',counts)
    np.save(out/'discovery_states_step20.npy',states);np.save(out/'discovery_sample_ids.npy',ids)
    means=maps.mean((1,2));variance=maps.var((1,2));active=variance>0
    cutoff=np.quantile(means[active],.9975);accepted=active&(means<=cutoff)
    cells=np.flatnonzero(accepted);accepted_maps=maps[cells]
    scorer=GridScorer(32,((0,2.2),(0,2.2)),zip([.2]*10,np.linspace(.4,1,10)))
    scores=np.full(4096,np.nan);t0=time.perf_counter()
    for j,i in enumerate(cells):
        with np.errstate(all='ignore'):scores[i]=scorer.get_scores(maps[i].astype(np.float64))[0]
        if j%500==0:print(source,'grid scoring',j,'/',len(cells),flush=True)
    np.save(out/'discovery_grid_scores.npy',scores)
    rot=rotation_features(accepted_maps);np.save(out/'rotational_features.npy',rot)
    import umap
    reducer=umap.UMAP(metric='euclidean',n_neighbors=20,min_dist=.05,init='spectral',random_state=72003,n_jobs=1)
    low=reducer.fit_transform(rot);labels=DBSCAN(eps=.5,min_samples=30).fit_predict(low)
    np.savez(out/'rotation_clusters.npz',cell_ids=cells,features=rot,embedding=low,labels=labels)
    r,theta,conc=fourier_features(accepted_maps)
    band=(r>=1.5)&(conc>=.5)
    median=float(np.median(r[band])) if band.any() else None
    cohort=band&(np.abs(r-median)<=.2*median) if median is not None else np.zeros(len(cells),bool)
    candidate=cells[cohort];reason='insufficient band units';angular={}
    if len(candidate)>=48:
        feat=np.stack([np.cos(2*theta[cohort]),np.sin(2*theta[cohort])],1)
        km=KMeans(3,random_state=72004,n_init=20).fit(feat)
        center_angles=(np.arctan2(km.cluster_centers_[:,1],km.cluster_centers_[:,0])/2)%np.pi
        sorted_angles=np.sort(center_angles);gaps=np.diff(np.r_[sorted_angles,sorted_angles[0]+np.pi])
        n_by=np.bincount(km.labels_,minlength=3)
        angular=dict(group_counts=n_by.tolist(),angles_deg=np.degrees(center_angles).tolist(),
                     circular_gaps_deg=np.degrees(gaps).tolist())
        if np.min(n_by)>=16 and np.all(np.abs(gaps-np.pi/3)<=np.pi/12):reason='frozen geometry-qualified band candidate'
        else:candidate=np.array([],int);reason='band orientation population fails predetermined three-axis coverage'
    else:candidate=np.array([],int)
    np.savez(out/'cell_features.npz',cell_ids=cells,frequency_cycles_per_box=r,orientation_rad=theta,
             band_concentration=conc,band_mask=band,frequency_cohort_mask=cohort)
    sets={'all_neurons':np.arange(4096),'grid_like':np.flatnonzero(accepted&np.isfinite(scores)&(scores>.3)),
          'primary_candidate':candidate}
    for lab in np.unique(labels):
        if lab>=0 and np.sum(labels==lab)>=64:sets[f'rotation_cluster_{lab}']=cells[labels==lab]
    np.savez(out/'frozen_neuron_sets.npz',**sets)
    selection=dict(source=source,discovery_only=True,used_topology=False,active_count=int(active.sum()),
        bias_excluded_count=int((active&~accepted).sum()),accepted_count=len(cells),grid_like_count=len(sets['grid_like']),
        primary_candidate_count=len(candidate),candidate_reason=reason,band_count=int(band.sum()),
        band_frequency_median=median,orientation_check=angular,
        rotation_clusters={str(l):int(np.sum(labels==l)) for l in np.unique(labels)},
        computational_seconds=time.perf_counter()-t0,mask_file_sha256=sha256(out/'frozen_neuron_sets.npz'),
        protocol_sha256=sha256(P2/'protocol.json'),missing_discovery_bins=int((counts==0).sum()),
        no_manual_cluster_merging=True)
    save_json(out/'selection.json',selection);print(json.dumps(selection),flush=True)
    # Only after writing the frozen membership do we read heldout data, to measure (not change) map scores.
    for split in ['validation','confirmation']:
        m,c,s,sids=maps_and_states(source,split)
        np.save(out/f'{split}_states_step20.npy',s);np.save(out/f'{split}_sample_ids.npy',sids)
        np.save(out/f'{split}_ratemaps.npy',m);np.save(out/f'{split}_occupancy.npy',c)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args();identify(a.source)

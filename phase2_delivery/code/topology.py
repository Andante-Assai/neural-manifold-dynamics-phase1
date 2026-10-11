"""Heldout raw-state PH and prespecified torus statistic; UMAP is never used for testing."""
from common import *
import argparse
from ripser import ripser
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist,squareform
from scipy.stats import rankdata

def distances(x,metric='euclidean'):
    if metric=='cosine':
        x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-20)
    if torch.cuda.is_available():
        with torch.no_grad():
            t=torch.as_tensor(np.ascontiguousarray(x),device='cuda',dtype=torch.float32)
            if metric=='euclidean':
                sq=(t*t).sum(1);d=(sq[:,None]+sq[None,:]-2*t@t.T).clamp_min(0).sqrt()
            else:d=(1-t@t.T).clamp(0,2)
            a=d.cpu().numpy();np.fill_diagonal(a,0);return a
    return squareform(pdist(x.astype(np.float64),metric))

def signature(dgms,threshold,scale):
    life=[];top=[]
    for dim in [1,2]:
        d=dgms[dim];l=(np.minimum(d[:,1],threshold)-d[:,0])/scale
        order=np.argsort(l)[::-1];life.append(l[order]);top.append(order)
    h1,h2=life
    l12=float(h1[1]) if len(h1)>=2 else 0.;l21=float(h2[0]) if len(h2) else 0.
    ratio1=l12/(float(h1[2]) if len(h1)>2 and h1[2]>0 else 1e-12)
    ratio2=l21/(float(h2[1]) if len(h2)>1 and h2[1]>0 else 1e-12)
    events={0.:np.zeros(3,int),float(threshold):np.zeros(3,int)}
    for dim,d in enumerate(dgms):
        for a,b in d:
            if 0<=a<=threshold:events.setdefault(float(a),np.zeros(3,int))[dim]+=1
            if np.isfinite(b) and 0<=b<=threshold:events.setdefault(float(b),np.zeros(3,int))[dim]-=1
    critical=sorted(events)
    intervals=[]
    beta=np.zeros(3,int)
    for a,b in zip(critical[:-1],critical[1:]):
        beta+=events[a]
        if np.array_equal(beta,[1,2,1]):
            if intervals and abs(intervals[-1][1]-a)<1e-6*scale:intervals[-1][1]=float(b)
            else:intervals.append([float(a),float(b)])
    longest=max(((b-a)/scale for a,b in intervals),default=0.)
    ok=(l12>=.15 and l21>=.15 and ratio1>=2 and ratio2>=2 and longest>=.05)
    censored=[int(np.isinf(d[:,1]).sum()) for d in dgms]
    return dict(signature_pass=bool(ok),score=min(l12,l21) if ok else 0.,
       second_H1_normalized_lifetime=l12,first_H2_normalized_lifetime=l21,
       H1_second_to_third_ratio=float(ratio1),H2_first_to_second_ratio=float(ratio2),
       beta121_intervals=intervals,longest_beta121_interval_normalized=float(longest),
       censored_bars_by_dimension=censored,bar_counts=[len(d) for d in dgms])

def ph(x,n=384,coeff=47,metric='euclidean',start=0):
    d=distances(x,metric);inds=fps(d,n,start)
    sub=d[np.ix_(inds,inds)];scale=float(np.median(sub[np.triu_indices(len(inds),1)]))
    threshold=scale
    t=time.perf_counter();ds=ripser(sub,distance_matrix=True,maxdim=2,coeff=coeff,thresh=threshold)['dgms']
    result=signature(ds,threshold,scale)
    result.update(n_input=len(x),landmarks=len(inds),field=coeff,metric=metric,
                  threshold=threshold,distance_scale=scale,seconds=time.perf_counter()-t)
    return result,ds,inds

def null_ph(x,n=384,coeff=47,full_H2=False):
    """Exact logical short-circuit: failing necessary H1 criteria guarantees joint statistic zero."""
    if full_H2:
        r,_,_=ph(x,n=n,coeff=coeff);r['H2_computed']=True;return r
    d=distances(x);ii=fps(d,n,0);sub=d[np.ix_(ii,ii)]
    scale=float(np.median(sub[np.triu_indices(len(ii),1)]))
    off=sub[np.triu_indices(len(ii),1)]
    span_bound=(scale-float(np.min(off)))/scale
    if span_bound<.15-1e-7:
        return dict(score=0.,signature_pass=False,second_H1_normalized_lifetime=None,
             first_H2_normalized_lifetime=None,H2_computed=False,H1_computed=False,
             H1_lifetime_upper_bound=span_bound,
             reason='H1 births cannot precede first edge; truncated lifetimes are all <0.15 by a rigorous edge-span bound')
    ds=ripser(sub,distance_matrix=True,maxdim=1,coeff=coeff,thresh=scale)['dgms']
    life=np.sort((np.minimum(ds[1][:,1],scale)-ds[1][:,0])/scale)[::-1]
    second=float(life[1]) if len(life)>=2 else 0.
    ratio=second/(float(life[2]) if len(life)>2 and life[2]>0 else 1e-12)
    if second<.15 or ratio<2:
        return dict(score=0.,signature_pass=False,second_H1_normalized_lifetime=second,
             first_H2_normalized_lifetime=None,H2_computed=False,
             H1_computed=True,
             reason='necessary H1 criteria fail; complete joint statistic is provably zero')
    r,_,_=ph(x,n=n,coeff=coeff);r['H2_computed']=True;return r

def dimension(x):
    d=distances(x);np.fill_diagonal(d,np.inf)
    near=np.partition(d,1,axis=1)[:,:2];near.sort(1)
    valid=near[:,0]>1e-8;log=np.log(near[valid,1]/near[valid,0]);log=np.sort(log)
    trimmed=log[int(.1*len(log)):int(.9*len(log))]
    two=float(1/np.mean(trimmed)) if len(trimmed) and np.mean(trimmed)>1e-9 else None
    z=x-x.mean(0);t=torch.as_tensor(z,device='cuda',dtype=torch.float32)
    gram=(t@t.T/(len(x)-1)).cpu().numpy();ev=np.maximum(np.linalg.eigvalsh(gram),0)
    pr=float(ev.sum()**2/(ev@ev)) if ev@ev>0 else 0.
    return dict(twoNN_trimmed=two,participation_ratio=pr,points=len(x),neurons=x.shape[1])

def transform_fit(train,neurons):
    x=train[:,neurons];mean=x.mean(0);rms=float(np.sqrt(np.mean(np.sum((x-mean)**2,1))))
    return mean,max(rms,1e-20)

def write_dgms(path,ds,inds):
    np.savez(path,H0=ds[0],H1=ds[1],H2=ds[2],landmark_indices=inds)

def screen(source):
    out=P2/'analysis'/source;dest=out/'topology';dest.mkdir(exist_ok=True)
    train=np.load(out/'discovery_states_step20.npy',mmap_mode='r')
    val=np.load(out/'validation_states_step20.npy',mmap_mode='r')
    sets=np.load(out/'frozen_neuron_sets.npz');reports=[]
    for name in sets.files:
        cells=sets[name]
        if len(cells)<8:reports.append(dict(population=name,n_cells=len(cells),status='not_testable'));continue
        mean,rms=transform_fit(train,cells);x=(val[:,cells]-mean)/rms
        for metric in ['euclidean','cosine']:
            r,ds,inds=ph(x,metric=metric)
            r.update(population=name,n_cells=len(cells),status='tested',split='validation',confirmatory=name=='primary_candidate' and metric=='euclidean')
            if metric=='euclidean':r['intrinsic_dimension']=dimension(x[inds])
            write_dgms(dest/f'validation_{name}_{metric}_F47.npz',ds,inds)
            reports.append(r);print(source,name,metric,json.dumps(r),flush=True)
    save_json(dest/'validation_screen.json',reports)

def primary(source):
    out=P2/'analysis'/source;dest=out/'topology';dest.mkdir(exist_ok=True)
    cells=np.load(out/'frozen_neuron_sets.npz')['primary_candidate']
    if len(cells)<48:
        save_json(dest/'primary_result.json',dict(source=source,status='no_frozen_candidate',accept=False,
           explanation='No discovery-only geometry-qualified candidate; no torus claim. This does not prove that all possible subpopulations lack a torus.'))
        return
    train=np.load(out/'discovery_states_step20.npy',mmap_mode='r')
    mean,rms=transform_fit(train,cells);selection=sha256(out/'frozen_neuron_sets.npz')
    pca=PCA(n_components=6,random_state=72008,svd_solver='randomized').fit(train[:,cells])
    np.savez(dest/'fixed_transform.npz',cell_ids=cells,mean=mean,rms=rms,
             PCA6_mean=pca.mean_,PCA6_components=pca.components_,PCA6_variance=pca.explained_variance_ratio_)
    reports=[]
    for split,n in [('validation',384),('confirmation',512)]:
        raw=np.load(out/f'{split}_states_step20.npy');x=(raw[:,cells]-mean)/rms
        observed,ds,inds=ph(x,n=n,coeff=47)
        write_dgms(dest/f'{split}_primary_raw_F47.npz',ds,inds)
        sensitivity=[]
        for field,metric,xx,nn in [(2,'euclidean',x,n),(47,'euclidean',x,512 if n==384 else 384),
                                   (47,'cosine',x,n),(47,'cosine',pca.transform(raw[:,cells]),n)]:
            r,dd,ii=ph(xx,n=nn,coeff=field,metric=metric)
            r['representation']='PCA6' if xx.shape[1]==6 else 'raw'
            sensitivity.append(r);write_dgms(dest/f'{split}_{r["representation"]}_{metric}_F{field}_n{nn}.npz',dd,ii)
        rng=np.random.default_rng(72201+(0 if split=='validation' else 1))
        B=199 if observed['score']>0 else 19
        scores=[];null_details=[]
        for b in range(B):
            # Recompute landmark selection under each randomization; selection of neurons remains discovery-fixed.
            shuffled=np.empty_like(x)
            for j in range(x.shape[1]):shuffled[:,j]=x[rng.permutation(len(x)),j]
            r,_,_=ph(shuffled,n=n,coeff=47)
            scores.append(r['score']);null_details.append({k:r[k] for k in ['score','signature_pass','second_H1_normalized_lifetime','first_H2_normalized_lifetime']})
            if b%20==0:
                save_json(dest/f'{split}_null_progress.json',dict(completed=b+1,planned=B,scores=scores,details=null_details))
                print(source,split,'nulls',b+1,'/',B,flush=True)
        p=1. if observed['score']==0 else (1+np.sum(np.array(scores)>=observed['score']))/(B+1)
        report=dict(split=split,source=source,observed=observed,sensitivity=sensitivity,
           B_executed=B,B_prespecified=199,p_unadjusted=float(p),null_scores=scores,null_details=null_details,
           intrinsic_dimension=dimension(x[inds]),selection_sha256=selection,
           field2_and_landmark_signature_agree=all(r['signature_pass'] for r in sensitivity[:2]),
           p_method='nonnegative score==0 implies exact p=1; otherwise empirical plus-one test')
        reports.append(report);save_json(dest/f'{split}_primary_test.json',report)
        print(source,split,'primary test',p,observed['signature_pass'],flush=True)
    save_json(dest/'primary_result.json',dict(source=source,reports=reports,selection_sha256=selection,
        accept_unadjusted=all(r['observed']['signature_pass'] and r['field2_and_landmark_signature_agree'] and r['p_unadjusted']<=.05 for r in reports),
        acceptance_requires_Holm_across_sources=True))

def controls():
    out=P2/'controls';out.mkdir(exist_ok=True);rng=np.random.default_rng(72200);n=2048
    a,b=rng.uniform(0,2*np.pi,(2,n))
    phase=rng.uniform(0,2*np.pi,(3,32))
    band=np.maximum(np.cos(np.stack([a,b,b-a],1)[:,:,None]+phase[None]),0).reshape(n,-1)
    cloud={'clifford_torus':np.stack([np.cos(a),np.sin(a),np.cos(b),np.sin(b)],1),
           'three_direction_bands':band,'plane':rng.uniform(-1,1,(n,2))}
    sphere=rng.normal(size=(n,3));sphere/=np.linalg.norm(sphere,axis=1,keepdims=True);cloud['sphere']=sphere
    results=[]
    for name,x in cloud.items():
        x=x.astype(np.float32);x-=x.mean(0);x/=np.sqrt(np.mean(np.sum(x*x,1)))
        for count in [384,512]:
            r,dg,inds=ph(x,n=count,coeff=47);r['control']=name
            results.append(r);write_dgms(out/f'{name}_{count}.npz',dg,inds)
            print('control',name,count,json.dumps(r),flush=True)
    save_json(out/'results.json',results)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['screen','primary','controls']);p.add_argument('--source')
    a=p.parse_args();setup()
    controls() if a.stage=='controls' else globals()[a.stage](a.source)

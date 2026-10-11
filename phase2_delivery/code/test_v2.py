from topology import *
def test(source):
    out=P2/'analysis'/source;dest=out/'v2/topology';dest.mkdir(parents=True,exist_ok=True)
    cells=np.load(out/'paper_cluster_candidate_v2.npy')
    if len(cells)<48:save_json(dest/'result.json',dict(source=source,status='no_candidate',accept=False));return
    train=np.load(out/'discovery_states_step20.npy',mmap_mode='r');mean,rms=transform_fit(train,cells)
    pca=PCA(6,svd_solver='randomized',random_state=72008).fit(train[:,cells])
    np.savez(dest/'transform.npz',cells=cells,mean=mean,rms=rms,PCA6_mean=pca.mean_,PCA6_components=pca.components_,variance=pca.explained_variance_ratio_)
    report=[]
    for condition in ['stationary','random_foraging']:
        for split,n,offset in [('validation',384,0),('confirmation',512,1)]:
            finished=dest/f'{condition}_{split}_test.json'
            if finished.exists():report.append(json.loads(finished.read_text()));continue
            raw=np.load(out/f'v2/{split}_{condition}_states.npy',mmap_mode='r');x=(raw[:,cells]-mean)/rms
            observed,dg,inds=ph(x,n=n,coeff=47);write_dgms(dest/f'{condition}_{split}_raw_F47.npz',dg,inds)
            sensitivities=[]
            for field,metric,xx,nn in [(2,'euclidean',x,n),(47,'euclidean',x,512 if n==384 else 384),
                  (47,'cosine',x,n),(47,'cosine',pca.transform(raw[:,cells]),n)]:
                r,dd,ii=ph(xx,n=nn,coeff=field,metric=metric);r['representation']='PCA6' if xx.shape[1]==6 else 'raw'
                write_dgms(dest/f'{condition}_{split}_{r["representation"]}_{metric}_F{field}_{nn}.npz',dd,ii);sensitivities.append(r)
            preliminary=dict(source=source,condition=condition,split=split,observed=observed,sensitivity=sensitivities)
            save_json(dest/f'{condition}_{split}_observed.json',preliminary)
            print('V2 observed',condition,split,json.dumps(observed),flush=True)
            B=199 if observed['score']>0 else 19;scores=[];nulls=[]
            rng=np.random.default_rng(72700+offset+(0 if condition=='stationary' else 10))
            progress_file=dest/f'{condition}_{split}_progress.json'
            if progress_file.exists():
                previous=json.loads(progress_file.read_text());scores=previous['scores'];nulls=previous['details']
                for _ in range(len(scores)):
                    for j in range(x.shape[1]):rng.permutation(len(x))
            for b in range(len(scores),B):
                shuffled=np.empty_like(x)
                for j in range(x.shape[1]):shuffled[:,j]=x[rng.permutation(len(x)),j]
                r=null_ph(shuffled,n=n,coeff=47);scores.append(r['score'])
                nulls.append({k:r[k] for k in ['score','signature_pass','second_H1_normalized_lifetime','first_H2_normalized_lifetime','H2_computed']})
                if b%20==0:
                    save_json(dest/f'{condition}_{split}_progress.json',dict(completed=b+1,planned=B,scores=scores,details=nulls))
                    print('V2 null',condition,split,b+1,'/',B,flush=True)
            p=1. if observed['score']==0 else float((1+np.sum(np.array(scores)>=observed['score']))/(B+1))
            d=dict(**preliminary,B_executed=B,B_planned=199,p_unadjusted=p,null_scores=scores,null_details=nulls,
               intrinsic_dimension=dimension(x[inds]),mask_sha256=sha256(out/'paper_cluster_candidate_v2.npy'),
               protocol_sha256=sha256(P2/'protocol_v2.json'),field_and_landmark_sensitivity_pass=all(r['signature_pass'] for r in sensitivities[:2]))
            save_json(dest/f'{condition}_{split}_test.json',d);report.append(d)
            print('V2 finished',condition,split,'p=',p,flush=True)
    save_json(dest/'result.json',dict(source=source,reports=report,requires_Holm_six_slots=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='official_ganguli_pretrained');a=p.parse_args();setup();test(a.source)

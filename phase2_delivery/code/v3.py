from topology import *
def collect_v3():
    source='official_ganguli_pretrained';model,ck=get_model(source,setup());out=P2/'data'/source/'v3';out.mkdir(exist_ok=True)
    checkpoint=sha256(P2/'models'/f'{source}.pt');rows=[]
    from collect import save_chunk
    with torch.no_grad():
        for split,seed in [('validation',72802),('confirmation',72803)]:
            rng=np.random.default_rng(seed)
            for begin in range(0,8192,128):
                p,u=foraging(rng,128)
                for condition in ['stationary','random_foraging']:
                    pos=np.repeat(p[:,:1],21,axis=1) if condition=='stationary' else p
                    vv=np.zeros_like(u) if condition=='stationary' else u
                    pt=torch.as_tensor(pos,device=model.centers.device);ut=torch.as_tensor(vv,device=pt.device)
                    p0=model.pc(pt[:,0]);H=model(ut,p0);logits=model.decoder(H[:,20:21])
                    err=torch.linalg.vector_norm(model.decode(logits)-pt[:,20:21],dim=-1)
                    path=out/f'{split}_{condition}_{begin:06d}.h5'
                    ids=np.arange(begin,begin+128)+(0 if split=='validation' else 8192)+(0 if condition=='stationary' else 16384)
                    save_chunk(path,H[:,20:21].cpu().numpy(),np.array([20]),ids,pos,vv,p0.cpu().numpy(),logits.cpu().numpy(),err.cpu().numpy(),split,source,checkpoint)
                    with h5py.File(path,'r+') as f:f.attrs['condition']=condition;f.create_dataset('readout_steps',data=[20])
                    rows.append(dict(path=path.relative_to(P2).as_posix(),split=split,condition=condition,n=128,sha256=sha256(path),bytes=path.stat().st_size))
                if begin%2048==0:print('v3 fresh capture',split,begin+128,'/',8192,flush=True)
    save_json(out/'manifest.json',dict(files=rows,source=source,checkpoint_sha256=checkpoint,seeds={'validation':72802,'confirmation':72803},N_per_condition=8192,all_neurons=4096,H_steps=[20],input_history_steps=list(range(21))))
    dest=P2/'analysis'/source/'v3';dest.mkdir(exist_ok=True)
    for split in ['validation','confirmation']:
        for c in ['stationary','random_foraging']:
            states=[];ids=[];pos=[];err=[]
            for r in rows:
                if r['split']!=split or r['condition']!=c:continue
                with h5py.File(P2/r['path']) as f:
                    states.append(f['H'][0]);ids.append(f['sample_id'][...]);pos.append(f['position'][20]);err.append(f['error_m'][0])
            np.save(dest/f'{split}_{c}_states.npy',np.concatenate(states))
            np.savez(dest/f'{split}_{c}_metadata.npz',sample_id=np.concatenate(ids),position=np.concatenate(pos),error_m=np.concatenate(err))

def test_v3(tag=''):
    source='official_ganguli_pretrained';out=P2/'analysis'/source;dest=out/(f'v3_recomputed_{tag}/topology' if tag else 'v3/topology');dest.mkdir(parents=True,exist_ok=True)
    cells=np.load(out/'paper_cluster_candidate_v2.npy')
    assert sha256(out/'paper_cluster_candidate_v2.npy')==json.loads((P2/'protocol_v3.json').read_text())['mask_sha256']
    train=np.load(out/'discovery_states_step20.npy',mmap_mode='r');mean,rms=transform_fit(train,cells)
    np.savez(dest/'transform.npz',cells=cells,mean=mean,rms=rms)
    reports=[]
    for condition in ['stationary','random_foraging']:
        for split,offset in [('validation',0),('confirmation',1)]:
            finished=dest/f'{condition}_{split}_test.json'
            if finished.exists():reports.append(json.loads(finished.read_text()));continue
            raw=np.load(out/f'v3/{split}_{condition}_states.npy',mmap_mode='r');x=(raw[:,cells]-mean)/rms
            observed,ds,inds=ph(x,n=768,coeff=47);write_dgms(dest/f'{condition}_{split}_F47_n768.npz',ds,inds)
            sensitivity=[]
            for coeff,n in [(2,768),(47,1024),(2,1024)]:
                r,d,i=ph(x,n=n,coeff=coeff);sensitivity.append(r);write_dgms(dest/f'{condition}_{split}_F{coeff}_n{n}.npz',d,i)
            save_json(dest/f'{condition}_{split}_observed.json',dict(observed=observed,sensitivity=sensitivity))
            print('V3 observed',condition,split,json.dumps(observed),flush=True)
            rng=np.random.default_rng(72900+offset+(0 if condition=='stationary' else 10));scores=[];details=[]
            B=199 if observed['score']>0 else 19
            progress=dest/f'{condition}_{split}_progress.json'
            if progress.exists():
                old=json.loads(progress.read_text());scores=old['scores'];details=old['details']
                for _ in range(len(scores)):
                    for j in range(x.shape[1]):rng.permutation(len(x))
            for b in range(len(scores),B):
                sh=np.empty_like(x)
                for j in range(x.shape[1]):sh[:,j]=x[rng.permutation(len(x)),j]
                r=null_ph(sh,n=768);scores.append(r['score'])
                details.append({k:r[k] for k in ['score','signature_pass','second_H1_normalized_lifetime','first_H2_normalized_lifetime','H2_computed']})
                if b%20==0:
                    save_json(progress,dict(completed=b+1,planned=B,scores=scores,details=details))
                    print('V3 nulls',condition,split,b+1,'/',B,flush=True)
            p=1. if observed['score']==0 else float((1+np.sum(np.array(scores)>=observed['score']))/(B+1))
            result=dict(source=source,condition=condition,split=split,observed=observed,sensitivity=sensitivity,
                B_executed=B,B_planned=199,p_unadjusted=p,null_scores=scores,null_details=details,
                intrinsic_dimension=dimension(x[inds]),selection_sha256=sha256(out/'paper_cluster_candidate_v2.npy'),
                protocol_sha256=sha256(P2/'protocol_v3.json'),field_and_resolution_pass=all(r['signature_pass'] for r in sensitivity))
            save_json(finished,result);reports.append(result)
            print('V3 completed',condition,split,'p=',p,flush=True)
    save_json(dest/'result.json',dict(reports=reports,requires_Holm_ten_slots=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','test']);p.add_argument('--tag',default='');a=p.parse_args();setup()
    collect_v3() if a.stage=='collect' else test_v3(a.tag)

from topology import *
from sklearn.decomposition import PCA
import argparse
@torch.no_grad()
def collect_time():
    decision=json.loads((P2/'RESULTS.json').read_text());assert decision['acceptance_by_condition']['stationary']
    source='official_ganguli_pretrained';model,ck=get_model(source,setup(73001))
    cells=np.load(P2/'analysis'/source/'paper_cluster_candidate_v2.npy')
    out=P2/'data'/source/'timewise';out.mkdir(exist_ok=True);rows=[]
    rng=np.random.default_rng(73001);checkpoint=sha256(P2/'models'/f'{source}.pt')
    from collect import save_chunk
    for begin in range(0,1024,128):
        p,u=foraging(rng,128)
        for cond in ['stationary','random_foraging']:
            pos=np.repeat(p[:,:1],21,axis=1) if cond=='stationary' else p
            vv=np.zeros_like(u) if cond=='stationary' else u
            pt=torch.as_tensor(pos,device='cuda');ut=torch.as_tensor(vv,device='cuda');p0=model.pc(pt[:,0])
            H=model(ut,p0);y=model.decoder(H);error=torch.linalg.vector_norm(model.decode(y)-pt,dim=-1)
            path=out/f'{cond}_{begin:06d}.h5';ids=np.arange(begin,begin+128)+(0 if cond=='stationary' else 1024)
            save_chunk(path,H.cpu().numpy(),np.arange(21),ids,pos,vv,p0.cpu().numpy(),y.cpu().numpy(),error.cpu().numpy(),'timewise',source,checkpoint)
            with h5py.File(path,'r+') as f:
                f.attrs['condition']=cond;f.create_dataset('candidate_neuron_ids',data=cells)
                f.create_dataset('candidate_H',data=H[:,:,cells].cpu().numpy().transpose(1,0,2),compression='gzip',compression_opts=1,shuffle=True)
                f.create_dataset('velocity_m_s',data=vv.transpose(1,0,2)/.02)
                f.create_dataset('input_present',data=np.arange(21)>0);f.create_dataset('readout_steps',data=np.arange(21))
            rows.append(dict(path=path.relative_to(P2).as_posix(),n=128,condition=cond,split='timewise',sha256=sha256(path),bytes=path.stat().st_size))
    save_json(out/'manifest.json',dict(source=source,files=rows,checkpoint_sha256=checkpoint,
       full_H_shape=[21,2048,4096],candidate_H_shape=[21,2048,len(cells)],seed=73001,
       same_sample_ids_every_step=True,same_initial_positions_between_conditions=True,H_steps=list(range(21)),
       no_recurrent_state_reset=True,neuron_mask_sha256=sha256(P2/'analysis'/source/'paper_cluster_candidate_v2.npy')))
    print('Full matched timewise states recorded',flush=True)

def analyze_time():
    source='official_ganguli_pretrained';out=P2/'analysis'/source/'timewise';out.mkdir(exist_ok=True)
    cells=np.load(P2/'analysis'/source/'paper_cluster_candidate_v2.npy');m=json.loads((P2/'data'/source/'timewise/manifest.json').read_text())
    candidate=[];positions=[];ids=[];conditions=[];full_end=[]
    for cond in ['stationary','random_foraging']:
        for r in m['files']:
            if r['condition']!=cond:continue
            with h5py.File(P2/r['path']) as f:
                candidate.append(f['candidate_H'][...]);positions.append(f['position'][...]);ids.append(f['sample_id'][...])
                conditions.extend([cond]*r['n']);full_end.append(f['H'][20])
    H=np.concatenate(candidate,axis=1);pos=np.concatenate(positions,axis=1);sample_ids=np.concatenate(ids)
    conditions=np.array(conditions)
    discovery=np.load(P2/'analysis'/source/'discovery_states_step20.npy',mmap_mode='r')
    pca=PCA(3,svd_solver='randomized',random_state=73004).fit(discovery[::4][:,cells])
    xyz=pca.transform(H.reshape(-1,len(cells))).reshape(21,2048,3)
    np.savez(out/'fixed_PCA_coordinates.npz',mean=pca.mean_,components=pca.components_,variance_ratio=pca.explained_variance_ratio_,
             xyz=xyz,sample_id=sample_ids,condition=conditions,position=pos,cell_ids=cells,
             note='ONE discovery-fitted basis, no per-step fit')
    # Endpoint reference from independent accepted stationary validation.
    reference=np.load(P2/'analysis'/source/'v3/validation_stationary_states.npy',mmap_mode='r')[:,cells]
    refscale=float(np.median(pdist(reference[::8])))
    records=[];step_reports=[]
    for c in ['stationary','random_foraging']:
        sel=conditions==c;h=H[:,sel];pp=pos[:,sel]
        endpointD=distances(h[20]);scale=float(np.median(endpointD[np.triu_indices(1024,1)]))
        for s in range(21):
            d=distances(h[s]);paired=np.linalg.norm(h[s]-h[s-1],axis=1).mean() if s>0 else 0.
            # GPU squared distances to independent reference cloud; not a geodesic or proof of deformation.
            with torch.no_grad():
                a=torch.as_tensor(h[s],device='cuda');b=torch.as_tensor(reference,device='cuda')
                nn=((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T).clamp_min(0).min(1).values.sqrt().cpu().numpy()
            records.append(dict(condition=c,step=s,paired_mean_state_displacement=float(paired),
                 centroid_distance_to_endpoint=float(np.linalg.norm(h[s].mean(0)-h[20].mean(0))),
                 relative_pairwise_distance_change=float(np.linalg.norm(d-endpointD)/max(np.linalg.norm(endpointD),1e-15)),
                 median_distance_ratio_to_endpoint=float(np.median(d[np.triu_indices(1024,1)])/scale),
                 mean_distance_to_independent_stationary_reference=float(nn.mean()),
                 mean_reference_distance_over_reference_scale=float(nn.mean()/refscale),
                 mean_physical_displacement_m=float(np.linalg.norm(pp[s]-pp[0],axis=1).mean()),
                 interpretation='paired state motion and ensemble geometry diagnostic; random-motion sample relocation and off-core states confound a deformation claim'))
        for s in [0,1,5,10,15,20]:
            r,ds,ii=ph(h[s],n=768,coeff=47);r.update(condition=c,step=s,diagnostic_only=True)
            step_reports.append(r);write_dgms(out/f'{c}_step{s}_PH.npz',ds,ii)
            save_json(out/'stepwise_topology.json',step_reports)
            print('timewise PH',c,s,r['signature_pass'],flush=True)
    save_json(out/'paired_state_and_ensemble_changes.json',records)
    print('Matched timewise analysis complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','analyze']);a=p.parse_args();setup()
    collect_time() if a.stage=='collect' else analyze_time()

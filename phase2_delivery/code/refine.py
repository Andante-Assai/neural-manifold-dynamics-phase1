from common import *
from itertools import combinations,product
import argparse
def refine(source):
    out=P2/'analysis'/source
    annotations=json.loads((out/'discovery_cluster_annotations.json').read_text())
    clusters=np.load(out/'rotation_clusters.npz');eligible=[]
    for r in annotations:
        R=r['orientation_coherence'];std=np.sqrt(max(0,-.5*np.log(max(R,1e-15))))
        if r['cluster']>=0 and r['n']>=30 and r['band_fraction']>=.5 and std<=np.pi/12 and r['frequency_CV']<=.2:
            eligible.append(r)
    choices=[]
    for trio in combinations(eligible,3):
        angle=np.sort([r['angle_deg']%180 for r in trio]);gaps=np.diff(np.r_[angle,angle[0]+180])
        freq=np.array([r['frequency_median'] for r in trio]);avg=freq.mean()
        if np.max(abs(gaps-60))>15 or np.max(abs(freq-avg))>.2*avg:continue
        k=np.array([[r['frequency_median']*np.cos(np.deg2rad(r['angle_deg'])),
                     r['frequency_median']*np.sin(np.deg2rad(r['angle_deg']))] for r in trio])
        closure=min(np.linalg.norm((k*np.array(signs)[:,None]).sum(0))/avg for signs in product([-1,1],repeat=3))
        if closure>.1:continue
        labels=sorted(r['cluster'] for r in trio)
        cells=clusters['cell_ids'][np.isin(clusters['labels'],labels)]
        choices.append(dict(cluster_labels=labels,n=len(cells),closure_error=closure,angular_gaps=gaps.tolist(),cell_ids=cells))
    choices.sort(key=lambda r:(-r['n'],r['cluster_labels']))
    best=choices[0] if choices else None
    cells=best['cell_ids'] if best else np.array([],int)
    np.save(out/'paper_cluster_candidate_v2.npy',cells)
    report=dict(source=source,selection_uses_only_discovery=True,selection_uses_topology=False,
      eligible_clusters=[r['cluster'] for r in eligible],all_eligible_triplets=[{k:v for k,v in r.items() if k!='cell_ids'} for r in choices],
      chosen_triplet={k:v for k,v in best.items() if k!='cell_ids'} if best else None,
      neurons=len(cells),mask_sha256=sha256(out/'paper_cluster_candidate_v2.npy'),protocol_sha256=sha256(P2/'protocol_v2.json'))
    save_json(out/'paper_cluster_selection_v2.json',report);print(json.dumps(report),flush=True)

@torch.no_grad()
def collect_v2(source):
    model,ck=get_model(source,setup());out=P2/'data'/source/'v2';out.mkdir(exist_ok=True)
    checkpoint=sha256(P2/'models'/f'{source}.pt');rows=[]
    from collect import save_chunk
    for split,seed in [('validation',72502),('confirmation',72503)]:
        rng=np.random.default_rng(seed)
        for begin in range(0,4096,128):
            p,u=foraging(rng,128)
            for condition in ['stationary','random_foraging']:
                pos=np.repeat(p[:,:1],21,axis=1) if condition=='stationary' else p
                vv=np.zeros_like(u) if condition=='stationary' else u
                pt=torch.as_tensor(pos,device=model.centers.device);ut=torch.as_tensor(vv,device=pt.device)
                p0=model.pc(pt[:,0]);H=model(ut,p0);logits=model.decoder(H)
                err=torch.linalg.vector_norm(model.decode(logits)-pt,dim=-1)
                dest=out/f'{split}_{condition}_{begin:06d}.h5'
                ids=np.arange(begin,begin+128)+(0 if split=='validation' else 4096)+(0 if condition=='stationary' else 8192)
                save_chunk(dest,H[:,[0,20]].cpu().numpy(),np.array([0,20]),ids,pos,vv,p0.cpu().numpy(),
                   logits.cpu().numpy(),err.cpu().numpy(),split,source,checkpoint)
                with h5py.File(dest,'r+') as f:f.attrs['condition']=condition
                rows.append(dict(path=dest.relative_to(P2).as_posix(),split=split,condition=condition,n=128,
                           sha256=sha256(dest),bytes=dest.stat().st_size))
            if begin%1024==0:print('fresh v2',split,begin+128,'/',4096,flush=True)
    save_json(out/'manifest.json',dict(source=source,files=rows,checkpoint_sha256=checkpoint,N_per_condition=4096,
       seeds={'validation':72502,'confirmation':72503},recorded_H_steps=[0,20],full_input_history=True,
       paired_initial_conditions_between_stationary_and_random=True))

def cache_v2(source):
    out=P2/'analysis'/source/'v2';out.mkdir(exist_ok=True)
    m=json.loads((P2/'data'/source/'v2/manifest.json').read_text())
    for split in ['validation','confirmation']:
        for c in ['stationary','random_foraging']:
            states=[];ids=[];pos=[];errs=[]
            for r in m['files']:
                if r['split']!=split or r['condition']!=c:continue
                with h5py.File(P2/r['path']) as f:
                    states.append(f['H'][1]);ids.append(f['sample_id'][...]);pos.append(f['position'][20]);errs.append(f['error_m'][20])
            np.save(out/f'{split}_{c}_states.npy',np.concatenate(states))
            np.savez(out/f'{split}_{c}_metadata.npz',sample_id=np.concatenate(ids),position=np.concatenate(pos),error_m=np.concatenate(errs))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','collect','cache']);p.add_argument('--source',required=True);a=p.parse_args()
    {'select':refine,'collect':collect_v2,'cache':cache_v2}[a.stage](a.source)

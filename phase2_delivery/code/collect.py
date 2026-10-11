from common import *
import argparse,shutil

def save_chunk(path,H,hsteps,ids,pos,u,p0,logits,error,split,source,checkpoint):
    with h5py.File(path,'w') as f:
        f.attrs.update(schema='phase2.raw.v1',source=source,split=split,checkpoint_sha256=checkpoint,
          H_axes='recorded_step,sample,neuron',all_neurons=True,dt_s=.02,
          input_units='metres per step, velocity_m_s=model_input/0.02',h0_readout_untrained=True)
        f.create_dataset('H',data=np.swapaxes(H,0,1),chunks=(1,min(64,len(ids)),4096),compression='gzip',compression_opts=1,shuffle=True)
        f.create_dataset('H_steps',data=hsteps)
        for key,a in {'position':pos,'model_input':u,'readout_logits':logits,'error_m':error}.items():
            a=np.swapaxes(a,0,1)
            f.create_dataset(key,data=a,compression='gzip',compression_opts=1,shuffle=True)
        f.create_dataset('input_steps',data=np.arange(21))
        f.create_dataset('initial_place_activity',data=p0)
        f.create_dataset('sample_id',data=ids)

def copy_existing():
    source='existing_full4096_fast';out=P2/'data'/source;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        print('Self-contained existing-model baseline already packaged; original phase1 files not required',flush=True);return
    oldmanifest=json.loads((OLD/'datasets/full4096_fast/manifest.json').read_text())
    rows=[];order=np.random.default_rng(72001).permutation(2048)
    groups={'discovery':order[:1024],'validation':order[1024:1536],'confirmation':order[1536:]}
    split_by_group=np.empty(2048,np.int8)
    for j,sp in enumerate(groups):split_by_group[groups[sp]]=j
    checkpoint=sha256(P2/'models'/f'{source}.pt')
    for r in oldmanifest['files']:
        if '/core_T20/' not in r['path']:continue
        p=OLD/r['path'];assert sha256(p)==r['sha256']
        dest=out/p.name
        with h5py.File(p) as f:
            ids=f['sample_id'][...];H=f['H'][...].transpose(1,0,2)
            pos=f['position'][...].transpose(1,0,2);u=f['model_input'][...].transpose(1,0,2)
            logits=f['output_logits'][...].transpose(1,0,2);error=f['error_m'][...].T
            save_chunk(dest,H,np.arange(21),ids,pos,u,f['initial_place_activity'][...],logits,error,'paired_group_split',source,checkpoint)
            with h5py.File(dest,'r+') as nf:
                nf.create_dataset('split_code',data=split_by_group[ids%2048])
                nf.create_dataset('condition',data=f['condition'][...]);nf.create_dataset('paired_initial_id',data=ids%2048)
        rows.append(dict(path=dest.relative_to(P2).as_posix(),sha256=sha256(dest),bytes=dest.stat().st_size,
                         n=len(ids),split='paired_group_split'))
    save_json(out/'manifest.json',dict(source=source,checkpoint_sha256=checkpoint,files=rows,
      split_codes={'discovery':0,'validation':1,'confirmation':2},groups={k:v.tolist() for k,v in groups.items()},
      split_seed=72001,primary_condition='random_foraging',all_neurons=4096,H_steps=list(range(21))))
    print('Existing raw states repackaged without changing old files',sum(r['bytes'] for r in rows),flush=True)

@torch.no_grad()
def pretrained():
    dev=setup();source='official_ganguli_pretrained';model,ck=get_model(source,dev)
    checkpoint=sha256(P2/'models'/f'{source}.pt');out=P2/'data'/source;out.mkdir(parents=True,exist_ok=True)
    config=[('discovery',10000,72101),('validation',2048,72102),('confirmation',2048,72103)]
    rows=[];offset=0
    for split,n,seed in config:
        rng=np.random.default_rng(seed)
        for begin in range(0,n,128):
            size=min(128,n-begin);pos,u=foraging(rng,size)
            pt=torch.as_tensor(pos,device=dev);ut=torch.as_tensor(u,device=dev)
            p0=model.pc(pt[:,0]);H=model(ut,p0);logits=model.decoder(H)
            err=torch.linalg.vector_norm(model.decode(logits)-pt,dim=-1)
            hsteps=np.arange(11,21) if split=='discovery' else np.arange(21)
            dest=out/f'{split}_{begin:06d}.h5';ids=np.arange(offset+begin,offset+begin+size)
            save_chunk(dest,H[:,hsteps].cpu().numpy(),hsteps,ids,pos,u,p0.cpu().numpy(),
                logits.cpu().numpy(),err.cpu().numpy(),split,source,checkpoint)
            if split=='discovery':
                with h5py.File(dest,'r+') as f:f.create_dataset('initial_hidden_state',data=H[:,0].cpu().numpy(),compression='gzip',compression_opts=1,shuffle=True)
            rows.append(dict(path=dest.relative_to(P2).as_posix(),sha256=sha256(dest),bytes=dest.stat().st_size,n=size,split=split))
            if begin%1024==0:print(source,split,begin+size,'/',n,flush=True)
            del H,logits
        offset+=n
    assert checkpoint==sha256(P2/'models'/f'{source}.pt')
    save_json(out/'manifest.json',dict(source=source,checkpoint_sha256=checkpoint,files=rows,
        split_seeds={k:s for k,n,s in config},split_sizes={k:n for k,n,s in config},chunk_size=128,
        discovery_recorded_H_steps=list(range(11,21)),validation_and_confirmation_H_steps=list(range(21)),
        input_history_complete=True,all_neurons=4096,weights_frozen=True))
    print('Pretrained collection complete',sum(r['bytes'] for r in rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',choices=['existing','pretrained']);a=p.parse_args()
    copy_existing() if a.source=='existing' else pretrained()

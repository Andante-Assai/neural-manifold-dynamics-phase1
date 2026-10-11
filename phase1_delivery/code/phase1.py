"""Actual author RNN, deterministic training, complete step-aligned activation capture."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, csv, hashlib, json, platform, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT.parent / 'work' / 'deps'
if DEPS.exists(): sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT / 'original' / 'src'))
sys.path.insert(0, str(ROOT / 'original' / 'src' / 'Models'))
import numpy as np
import torch
from SorscherRNN import SorscherRNN
from PlaceCells import PlaceCells

DT, BOX = .02, 2.2
CONDITIONS = ['stationary', 'constant_velocity', 'closed_circle', 'random_foraging']

def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def setup(seed, device='cuda'):
    np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return torch.device(device)

class Rectangle:
    def sample_uniform(self, n, seed=None):
        return np.random.default_rng(seed).uniform(0, BOX, (n, 2))

def make_model(cfg, device):
    pcs = PlaceCells(Rectangle(), cfg['Np'], pc_width=.12, DoG=True,
                     surround_scale=2, seed=cfg['place_seed'])
    if 'name' in cfg:
        centers=ROOT/'runs'/cfg['name']/'place_cell_centers.npy'
        if centers.exists():pcs.pcs=torch.as_tensor(np.load(centers),dtype=torch.float32)
    model = SorscherRNN([pcs], Ng=cfg['Ng'], Np=cfg['Np']).to(device)
    return model

def foraging(rng, n, steps, initial=None, return_initial_heading=False):
    """Vectorization of author Agent.sorschers_avoid_walls, on a square.
    Author 'velocities' already contain dt: these are displacements, not m/s.
    """
    pos = np.empty((n, steps+1, 2), np.float32)
    pos[:,0] = rng.uniform(0, BOX, (n,2)) if initial is None else initial
    hd = rng.uniform(0, 2*np.pi, n)
    initial_heading=hd.copy()
    u = np.zeros_like(pos)
    for t in range(1, steps+1):
        speed = rng.rayleigh(.13*2*np.pi, n) * DT
        turn = rng.normal(0, 5.76*2, n) * DT
        clipped = np.clip(pos[:,t-1], 1e-8, BOX-1e-8)
        ed = (clipped <= .03).astype(float) - (clipped >= BOX-.03).astype(float)
        angle = hd + turn
        direction = np.stack([np.cos(angle),np.sin(angle)],-1)
        toward = (ed*direction).sum(-1)<0
        sp = (ed*np.stack([np.cos(angle+np.pi/2),np.sin(angle+np.pi/2)],-1)).sum(-1)
        sn = (ed*np.stack([np.cos(angle-np.pi/2),np.sin(angle-np.pi/2)],-1)).sum(-1)
        turn += toward * np.sign(sp-sn) * np.pi/2
        speed = np.where(toward, speed/4, speed)
        hd = (hd + turn) % (2*np.pi)
        u[:,t] = np.stack([np.cos(hd),np.sin(hd)],-1)*speed[:,None]
        pos[:,t] = pos[:,t-1]+u[:,t]
    return (pos,u,initial_heading) if return_initial_heading else (pos,u)

def batch(model, rng, n, steps=20):
    p,u = foraging(rng,n,steps)
    pos = torch.as_tensor(p,device=model.device)
    labels = model.place_cell_ensembles[0].softmax_response(pos)
    return torch.as_tensor(u[:,1:],device=model.device), labels[:,0], labels[:,1:],pos

@torch.no_grad()
def evaluate(model, seed=991, n=512):
    u,p0,target,pos = batch(model,np.random.default_rng(seed),n)
    logp = model([u,p0],log_softmax=True)
    pc = model.place_cell_ensembles[0]
    pred = pc.to_euclid(logp); truth = pos[:,1:]
    err = torch.linalg.vector_norm(pred-truth,dim=-1)
    base = torch.linalg.vector_norm(pc.to_euclid(p0)[:,None]-truth,dim=-1)
    zero = pc.to_euclid(model([u*0,p0],log_softmax=True))
    perm = torch.randperm(n,device=model.device, generator=torch.Generator(device=model.device).manual_seed(seed))
    shuf = pc.to_euclid(model([u[perm],p0],log_softmax=True))
    oracle = torch.linalg.vector_norm(pc.to_euclid(target)-truth,dim=-1)
    kl = torch.sum(target*(torch.log(target.clamp_min(1e-30))-logp),-1)
    return dict(mean_error_m=err.mean().item(), final_error_m=err[:,-1].mean().item(),
                no_motion_baseline_m=base.mean().item(), final_no_motion_baseline_m=base[:,-1].mean().item(),
                zero_input_error_m=torch.linalg.vector_norm(zero-truth,dim=-1).mean().item(),
                shuffled_input_error_m=torch.linalg.vector_norm(shuf-truth,dim=-1).mean().item(),
                place_decoder_floor_m=oracle.mean().item(), mean_KL=kl.mean().item(),
                n=n,seed=seed)

def train(args):
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.run_name:cfg['name']=args.run_name
    run=ROOT/'runs'/cfg['name'];run.mkdir(parents=True,exist_ok=True)
    if (run/'last.pt').exists() and not args.resume:
        raise RuntimeError('Run already exists. Use --resume or choose a fresh --run-name.')
    dump(run/'config.json',cfg)
    device=setup(cfg['seed'],args.device); model=make_model(cfg,device)
    np.save(run/'place_cell_centers.npy',model.place_cell_ensembles[0].pcs.cpu().numpy())
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    rng=np.random.default_rng(cfg['train_seed']);start=0
    if args.resume:
        ck=torch.load(run/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['model_state_dict']);opt.load_state_dict(ck['optimizer_state_dict'])
        start=ck['step'];rng.bit_generator.state=ck['numpy_rng_state']
        torch.set_rng_state(ck['torch_rng_state'].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in ck['cuda_rng_states']])
    else:
        torch.save({'model_state_dict':model.state_dict(),'cfg':cfg,'step':0},run/'initial.pt')
        dump(run/'initial_evaluation.json',evaluate(model))
    t0=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    log=run/'training.jsonl'
    for step in range(start+1,args.steps+1):
        u,p0,labels,pos=batch(model,rng,cfg['batch_size'])
        opt.zero_grad(set_to_none=True)
        logp=model([u,p0],log_softmax=True)
        loss=model.loss_fn(logp,labels,cfg['weight_decay'])
        loss.backward();opt.step()
        if step % cfg['log_every']==0 or step==args.steps:
            ev=evaluate(model); ev.update(step=step,train_loss=float(loss.detach()),
                     elapsed_segment_s=time.perf_counter()-t0,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved())
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(ev)+'\n')
            print(json.dumps(ev),flush=True)
            ck={'model_state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),
                'cfg':cfg,'step':step,'numpy_rng_state':rng.bit_generator.state,
                'torch_rng_state':torch.get_rng_state(),'cuda_rng_states':torch.cuda.get_rng_state_all()}
            torch.save(ck,run/'last.pt')
            if ev['mean_error_m']<.095 and ev['mean_error_m']<.55*ev['no_motion_baseline_m']:
                if step>=cfg.get('min_steps',0):
                    print('heldout_path_integration_gate_passed',flush=True);break
    dump(run/'final_evaluation.json',evaluate(model,seed=992,n=1024))
    dump(run/'environment.json',{'python':sys.version,'platform':platform.platform(),
         'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),
         'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
         'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
         'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
         'completed_step':step,'runtime_segment_s':time.perf_counter()-t0})

def benchmark(args):
    out=[];device=setup(4,args.device)
    for ng in [256,512,1024,2048,4096]:
        cfg=dict(Ng=ng,Np=512,place_seed=0)
        try:
            model=make_model(cfg,device);opt=torch.optim.Adam(model.parameters(),lr=1e-4)
            torch.cuda.reset_peak_memory_stats();times=[]
            for i in range(6):
                u,p0,labels,_=batch(model,np.random.default_rng(100+i),200)
                torch.cuda.synchronize(); t=time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss=model.loss_fn(model([u,p0],log_softmax=True),labels,1e-4)
                loss.backward();opt.step();torch.cuda.synchronize()
                times.append(time.perf_counter()-t)
            row=dict(Ng=ng,Np=512,batch_size=200,T=20,success=True,
                     median_train_step_s=float(np.median(times[1:])),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved())
            del model,opt,u,p0,labels,loss
        except torch.cuda.OutOfMemoryError as e:
            row=dict(Ng=ng,success=False,error=str(e))
        torch.cuda.empty_cache();out.append(row);print(json.dumps(row),flush=True)
    dump(ROOT/'audit'/'memory_benchmark.json',out)

def paths(seed,n_per,steps=160):
    """Paired, stratified starts, deterministic all-160-step prefixes."""
    rng=np.random.default_rng(seed);side=int(np.ceil(np.sqrt(n_per)))
    grid=np.stack(np.meshgrid(np.arange(side),np.arange(side)),axis=-1).reshape(-1,2)[:n_per]
    starts=(grid+rng.uniform(.15,.85,(n_per,2)))/side*2.12+.04
    pos=np.empty((4*n_per,steps+1,2),np.float32)
    params=np.zeros((4*n_per,6),np.float64)
    for c in range(4):
        sl=slice(c*n_per,(c+1)*n_per);pos[sl,0]=starts
        if c==0:pos[sl]=starts[:,None]
        elif c==1:
            theta=rng.uniform(0,2*np.pi,n_per); direction=np.stack([np.cos(theta),np.sin(theta)],-1)
            wall=np.where(direction>0,(BOX-.01-starts)/np.maximum(direction,1e-9),
                          (starts-.01)/np.maximum(-direction,1e-9)).min(-1)
            speed=np.minimum(rng.uniform(.2,.8,n_per),wall/(steps*DT)*.90)
            pos[sl]=starts[:,None]+np.arange(steps+1)[None,:,None]*DT*direction[:,None]*speed[:,None,None]
            params[sl,0]=theta;params[sl,1]=speed
        elif c==2:
            theta=rng.uniform(0,2*np.pi,n_per)
            radius=np.minimum(.06,.4*np.minimum(starts,BOX-starts).min(-1))
            center=starts-radius[:,None]*np.stack([np.cos(theta),np.sin(theta)],-1)
            angles=theta[:,None]+np.arange(steps+1)[None,:]*2*np.pi/20
            pos[sl]=center[:,None]+radius[:,None,None]*np.stack([np.cos(angles),np.sin(angles)],-1)
            params[sl,0]=theta;params[sl,2]=radius;params[sl,3:5]=center;params[sl,5]=20
        else:
            pos[sl],_,hd0=foraging(rng,n_per,steps,starts,return_initial_heading=True)
            params[sl,0]=hd0
    u=np.zeros_like(pos);u[:,1:]=np.diff(pos,axis=1)
    return pos,u,np.repeat(np.arange(4,dtype=np.int8),n_per),params

@torch.no_grad()
def hidden(model,u,p0):
    h0=model.init_position_encoder(p0)
    hs=model.g([u[:,1:],p0])
    return torch.cat([h0[:,None],hs],dim=1)

def load_run(name,device):
    run=ROOT/'runs'/name
    ck=torch.load(run/'last.pt',map_location=device,weights_only=False)
    model=make_model(ck['cfg'],device);model.load_state_dict(ck['model_state_dict'])
    torch.nn.Module.train(model,False) # author overrides train() with a training loop
    for p in model.parameters():p.requires_grad_(False)
    return model,ck,run

@torch.no_grad()
def collect(args):
    import h5py
    device=setup(args.seed,args.device);model,ck,run=load_run(args.run,device)
    out=ROOT/'datasets'/args.run;out.mkdir(parents=True,exist_ok=True)
    before=digest(run/'last.pt')
    pos,u,cond,params=paths(args.seed,args.per_condition,160)
    np.savez(out/'trajectory_plan.npz',position=pos,model_input=u,condition=cond,parameters=params,
             sample_id=np.arange(len(pos),dtype=np.int64))
    sets=[('core_T20',np.arange(len(pos)),20),
          ('exploratory_T160',np.concatenate([c*args.per_condition+np.linspace(0,args.per_condition-1,args.ext_per_condition,dtype=int) for c in range(4)]),160)]
    inventory=[]
    for name,idx,T in sets:
        directory=out/name;directory.mkdir(exist_ok=True)
        for first in range(0,len(idx),args.chunk):
            ids=idx[first:first+args.chunk];p=pos[ids,:T+1];vv=u[ids,:T+1]
            pt=torch.as_tensor(p,device=device);vt=torch.as_tensor(vv,device=device)
            pc=model.place_cell_ensembles[0]
            target=pc.softmax_response(pt);p0=target[:,0]
            H=hidden(model,vt,p0);logits=model.p(H);P=logits.softmax(-1)
            predicted=pc.to_euclid(logits);oracle=pc.to_euclid(target)
            err=torch.linalg.vector_norm(predicted-pt,dim=-1)
            file=directory/f'chunk_{first:06d}_{first+len(ids):06d}.h5'
            with h5py.File(file,'w') as f:
                f.attrs.update(schema='phase1.v1',axes='step,sample,neuron',dt_s=DT,
                  model_input_units='metres per step (author velocities include dt)',
                  h0_definition='linear encoder(p0), BEFORE ReLU recurrent update',
                  condition_names=json.dumps(CONDITIONS),training_horizon=20,
                  exploratory_beyond_step=20,checkpoint_sha256=before)
                arrays={'H':H,'output_logits':logits,'output_probability':P,'target_place_activity':target,
                        'predicted_position':predicted,'target_decoded_position':oracle,'error_m':err}
                for key,a in arrays.items():
                    a=a.cpu().numpy();a=np.swapaxes(a,0,1)
                    f.create_dataset(key,data=a,chunks=(1,min(64,len(ids)))+a.shape[2:],compression='gzip',compression_opts=1,shuffle=True)
                for key,a in {'position':p,'model_input':vv,'velocity_m_s':vv/DT}.items():
                    f.create_dataset(key,data=np.swapaxes(a,0,1),compression='gzip',compression_opts=1)
                f.create_dataset('initial_place_activity',data=p0.cpu().numpy())
                f.create_dataset('sample_id',data=ids)
                f.create_dataset('condition',data=cond[ids]);f.create_dataset('trajectory_parameters',data=params[ids])
                f.attrs['trajectory_parameter_columns']='initial_angle_rad,constant_speed_m_s,circle_radius_m,circle_center_x_m,circle_center_y_m,circle_period_steps'
                f.attrs['trajectory_parameter_semantics']='condition-specific: stationary none; constant angle/speed; circle angle/radius/center/period; random initial heading only; unused columns are zero'
                f.create_dataset('step',data=np.arange(T+1))
                f.create_dataset('input_present',data=np.arange(T+1)>0)
                f.create_dataset('within_training_horizon',data=np.arange(T+1)<=20)
                f.create_dataset('initial_position',data=p[:,0])
                f.create_dataset('time_s',data=np.arange(T+1)*DT)
            row=dict(path=file.relative_to(ROOT).as_posix(),H_shape=[T+1,len(ids),model.Ng],
                     sample_ids=ids.tolist(),bytes=file.stat().st_size,sha256=digest(file))
            inventory.append(row);print(json.dumps({k:v for k,v in row.items() if k!='sample_ids'}),flush=True)
            del H,logits,P,target
    assert before==digest(run/'last.pt')
    dump(out/'manifest.json',dict(schema='phase1.v1',run=args.run,seed=args.seed,
         per_condition=args.per_condition,ext_per_condition=args.ext_per_condition,
         checkpoint_sha256=before,trained_steps=ck['step'],conditions=CONDITIONS,
         core_shape=[21,4*args.per_condition,model.Ng],extension_shape=[161,4*args.ext_per_condition,model.Ng],
         extension_prefix_paired=True,files=inventory))
    with (out/'samples.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['sample_id','condition','paired_initial_id','x0_m','y0_m','angle_rad','speed_m_s','radius_m','circle_center_x','circle_center_y','circle_period_steps'])
        for i in range(len(pos)):w.writerow([i,CONDITIONS[cond[i]],i%args.per_condition,*pos[i,0],*params[i]])

@torch.no_grad()
def verify(args):
    import h5py
    device=setup(42,args.device);model,ck,run=load_run(args.run,device)
    out=ROOT/'datasets'/args.run;manifest=json.loads((out/'manifest.json').read_text())
    results=[]
    for row in manifest['files']:
        file=ROOT/row['path'];assert digest(file)==row['sha256']
        with h5py.File(file) as f:
            for key in ['H','output_logits','output_probability','target_place_activity','position','model_input','error_m']:
                a=f[key][...];assert np.isfinite(a).all(),(file,key)
            assert np.allclose(np.diff(f['position'][...],axis=0),f['model_input'][1:],atol=3e-7)
            assert np.allclose(f['velocity_m_s'][...]*DT,f['model_input'][...],atol=1e-7)
            assert np.allclose(f['output_probability'][...].sum(-1),1,atol=2e-6)
            assert np.allclose(f['target_place_activity'][...].sum(-1),1,atol=2e-6)
            u0=torch.as_tensor(np.ascontiguousarray(np.swapaxes(f['model_input'][...],0,1)),device=device)
            p0=torch.as_tensor(f['initial_place_activity'][...],device=device)
            replay_tensor=hidden(model,u0,p0)
            replay=replay_tensor.cpu().numpy().transpose(1,0,2)
            stored=f['H'][...];delta=float(np.max(np.abs(replay-stored)))
            assert np.allclose(replay,stored,atol=2e-5,rtol=3e-5),(file,delta)
            readout=model.p(replay_tensor).cpu().numpy().transpose(1,0,2)
            readout_delta=float(np.max(np.abs(readout-f['output_logits'][...])))
            assert np.allclose(readout,f['output_logits'][...],atol=3e-5,rtol=5e-5)
            pp=model.p(replay_tensor).softmax(-1).cpu().numpy().transpose(1,0,2)
            assert np.allclose(pp,f['output_probability'][...],atol=2e-6,rtol=3e-5)
            # Manual recurrence independently verifies h0 and every subsequent step.
            h=model.init_position_encoder(p0);manual=[h]
            for t in range(1,u0.shape[1]):
                h=torch.relu(torch.nn.functional.linear(h,model.RNN.weight_hh_l0)+
                             torch.nn.functional.linear(u0[:,t],model.RNN.weight_ih_l0));manual.append(h)
            manual=torch.stack(manual).cpu().numpy()
            md=float(np.max(np.abs(manual-stored)))
            manual_tol=3e-5 if len(manual)<=21 else 1e-3
            assert np.allclose(manual,stored,atol=manual_tol,rtol=5e-5),(file,md)
            results.append(dict(path=row['path'],replay_max_abs_error=delta,manual_max_abs_error=md,
                                readout_max_abs_error=readout_delta,hash_verified=True))
            del replay_tensor,replay,stored,readout,pp,manual
    # Match extension sample IDs to their corresponding core trajectories.
    core={}
    for row in manifest['files']:
        if '/core_T20/' in row['path']:
            with h5py.File(ROOT/row['path']) as f:
                for i,sid in enumerate(f['sample_id'][...]):core[int(sid)]=(row['path'],i)
    prefix_errors=[]
    for row in manifest['files']:
        if '/exploratory_T160/' not in row['path']:continue
        with h5py.File(ROOT/row['path']) as ext:
            ext_h=ext['H'][:21];ext_pos=ext['position'][:21];groups={}
            for j,sid in enumerate(ext['sample_id'][...]):
                p,i=core[int(sid)]
                groups.setdefault(p,[]).append((i,j))
            for p,pairs in groups.items():
                src,dst=map(np.array,zip(*pairs))
                with h5py.File(ROOT/p) as f:
                    assert np.array_equal(f['position'][:,src],ext_pos[:,dst])
                    ch=f['H'][:,src];eh=ext_h[:,dst]
                    de=float(np.max(np.abs(ch-eh)))
                    assert np.allclose(ch,eh,atol=3e-5,rtol=5e-5)
                    prefix_errors.append(de)
    dump(out/'verification.json',dict(status='passed',all_file_hashes=True,all_finite=True,
         units_and_alignment=True,probability_normalization=True,replayed_samples_per_chunk='all',
         manual_recurrence=True,paired_prefix_max_abs_error=max(prefix_errors),results=results,
         tolerance_note='float32 cuDNN batch/sequence length can alter summation order; numerical replay, not universal bit identity'))
    print('Verification passed',flush=True)

def hashes(args):
    records=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name not in ['SHA256SUMS','file_inventory.json'] and '__pycache__' not in p.parts and '.git' not in p.parts:
            records.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p)))
    dump(ROOT/'file_inventory.json',records)
    (ROOT/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in records),encoding='utf-8')
    print(len(records),'files hashed')

def index(args):
    """Portable HDF5 virtual datasets expose global H without duplicating chunk arrays."""
    import h5py
    out=ROOT/'datasets'/args.run;m=json.loads((out/'manifest.json').read_text())
    records=[]
    dynamic=['H','output_logits','output_probability','target_place_activity','predicted_position',
             'target_decoded_position','error_m','position','model_input','velocity_m_s']
    samples=['initial_place_activity','sample_id','condition','trajectory_parameters','initial_position']
    for label in ['core_T20','exploratory_T160']:
        rows=[r for r in m['files'] if '/'+label+'/' in r['path']]
        target=out/f'{label}.h5';total=sum(r['H_shape'][1] for r in rows)
        with h5py.File(ROOT/rows[0]['path']) as first,h5py.File(target,'w',libver='latest') as f:
            for k in first.attrs:f.attrs[k]=first.attrs[k]
            f.attrs['storage']='HDF5 virtual dataset; relative links to physical lossless chunks'
            for k in dynamic+samples:
                dims=list(first[k].shape);axis=1 if k in dynamic else 0;dims[axis]=total
                layout=h5py.VirtualLayout(shape=tuple(dims),dtype=first[k].dtype)
                start=0
                for r in rows:
                    n=r['H_shape'][1];shape=list(first[k].shape);shape[axis]=n
                    source=h5py.VirtualSource((ROOT/r['path']).relative_to(target.parent).as_posix(),k,shape=tuple(shape))
                    sl=[slice(None)]*len(dims);sl[axis]=slice(start,start+n)
                    layout[tuple(sl)]=source;start+=n
                f.create_virtual_dataset(k,layout,fillvalue=np.nan if first[k].dtype.kind=='f' else -1)
            for k in ['step','input_present','within_training_horizon','time_s']:f.create_dataset(k,data=first[k][...])
        with h5py.File(target) as f:
            assert len(np.unique(f['sample_id'][...]))==total
            assert np.isfinite(f['H'][0,:4]).all()
        records.append(dict(path=target.relative_to(ROOT).as_posix(),sha256=digest(target),bytes=target.stat().st_size))
    m['virtual_indices']=records;dump(out/'manifest.json',m)
    print('Virtual indices created',flush=True)

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    b=s.add_parser('benchmark');b.add_argument('--device',default='cuda')
    t=s.add_parser('train');t.add_argument('--config',required=True);t.add_argument('--steps',type=int,required=True);t.add_argument('--resume',action='store_true');t.add_argument('--device',default='cuda');t.add_argument('--run-name')
    c=s.add_parser('collect');c.add_argument('--run',required=True);c.add_argument('--seed',type=int,default=20261011);c.add_argument('--per-condition',type=int,default=2048);c.add_argument('--ext-per-condition',type=int,default=256);c.add_argument('--chunk',type=int,default=128);c.add_argument('--device',default='cuda')
    v=s.add_parser('verify');v.add_argument('--run',required=True);v.add_argument('--device',default='cuda')
    ix=s.add_parser('index');ix.add_argument('--run',required=True)
    s.add_parser('hashes')
    a=p.parse_args();globals()[a.command](a)
if __name__=='__main__':main()

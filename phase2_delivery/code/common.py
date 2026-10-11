"""Independent phase-two foundation; first-phase artifacts are read-only inputs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4')
import sys,json,pickle,hashlib,importlib,time
from pathlib import Path
import numpy as np
P2=Path(__file__).resolve().parents[1]
OLD=P2.parent
DEPS=OLD.parent/'work'/'deps'
if DEPS.exists():sys.path.insert(0,str(DEPS))
import torch
import h5py
from scipy.spatial.distance import cdist

def save_json(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def setup(seed=72000):
    np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NumpyOnly(pickle.Unpickler):
    def find_class(self,module,name):
        ok={('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),
            ('numpy','ndarray'),('numpy','dtype'),('numpy.core.multiarray','scalar'),('numpy._core.multiarray','scalar')}
        if (module,name) not in ok:raise pickle.UnpicklingError((module,name))
        return getattr(importlib.import_module(module),name)
def numpy_weights(p):
    with open(p,'rb') as f:
        v=np.lib.format.read_magic(f)
        shape,_,dtype=(np.lib.format.read_array_header_1_0 if v==(1,0) else np.lib.format.read_array_header_2_0)(f)
        assert shape==(4,) and dtype.hasobject
        w=NumpyOnly(f).load().tolist()
    assert [x.shape for x in w]==[(512,4096),(2,4096),(4096,4096),(4096,512)]
    return w

class PathRNN(torch.nn.Module):
    def __init__(self,centers,Ng=4096):
        super().__init__();self.Ng=Ng;self.Np=len(centers)
        self.register_buffer('centers',torch.as_tensor(centers,dtype=torch.float32))
        self.encoder=torch.nn.Linear(self.Np,Ng,bias=False)
        self.RNN=torch.nn.RNN(2,Ng,nonlinearity='relu',bias=False,batch_first=True)
        self.decoder=torch.nn.Linear(Ng,self.Np,bias=False)
    def pc(self,pos):
        d=((pos[...,None,:]-self.centers)**2).sum(-1)
        a=torch.softmax(-d/(2*.12**2),-1)-torch.softmax(-d/(2*2*.12**2),-1)
        a=a-a.min(-1,keepdim=True).values
        return a/a.sum(-1,keepdim=True)
    def decode(self,act):return self.centers[act.topk(3,dim=-1).indices].mean(-2)
    def forward(self,u,p0):
        h0=self.encoder(p0);h,_=self.RNN(u[:,1:],h0[None])
        return torch.cat([h0[:,None],h],1)

def get_model(source,device):
    p=P2/'models'/f'{source}.pt'
    ck=torch.load(p,map_location='cpu',weights_only=True)
    model=PathRNN(ck['centers'],Ng=ck['Ng']).to(device)
    model.load_state_dict(ck['state_dict']);model.eval()
    for w in model.parameters():w.requires_grad_(False)
    return model,ck

def foraging(rng,n,steps=20):
    # Equation-compatible with original rat-simulator square-wall avoidance.
    pos=np.empty((n,steps+1,2),np.float32);pos[:,0]=rng.uniform(0,2.2,(n,2))
    hd=rng.uniform(0,2*np.pi,n);u=np.zeros_like(pos)
    for t in range(1,steps+1):
        speed=rng.rayleigh(.13*2*np.pi,n)*.02;turn=rng.normal(0,11.52,n)*.02
        clip=np.clip(pos[:,t-1],1e-8,2.2-1e-8)
        ed=(clip<=.03).astype(float)-(clip>=2.17).astype(float)
        a=hd+turn;di=np.stack([np.cos(a),np.sin(a)],-1);toward=(ed*di).sum(-1)<0
        sp=(ed*np.stack([np.cos(a+np.pi/2),np.sin(a+np.pi/2)],-1)).sum(-1)
        sn=(ed*np.stack([np.cos(a-np.pi/2),np.sin(a-np.pi/2)],-1)).sum(-1)
        turn+=toward*np.sign(sp-sn)*np.pi/2;speed=np.where(toward,speed/4,speed)
        hd=(hd+turn)%(2*np.pi);u[:,t]=np.stack([np.cos(hd),np.sin(hd)],-1)*speed[:,None]
        pos[:,t]=pos[:,t-1]+u[:,t]
    return pos,u

@torch.no_grad()
def evaluate(model,seed,n=2048):
    p,u=foraging(np.random.default_rng(seed),n)
    pp=torch.as_tensor(p,device=model.centers.device);uu=torch.as_tensor(u,device=pp.device)
    target=model.pc(pp);h=model(uu,target[:,0]);pred=model.decode(model.decoder(h))
    error=torch.linalg.vector_norm(pred-pp,dim=-1)
    floor=torch.linalg.vector_norm(model.decode(target)-pp,dim=-1)
    zero=model.decode(model.decoder(model(uu*0,target[:,0])))
    perm=torch.randperm(n,device=pp.device,generator=torch.Generator(device=pp.device).manual_seed(seed))
    shuffled=model.decode(model.decoder(model(uu[perm],target[:,0])))
    return dict(n=n,seed=seed,mean_error_m=float(error[:,1:].mean()),final_error_m=float(error[:,-1].mean()),
       decoder_floor_m=float(floor[:,1:].mean()),zero_input_error_m=float(torch.linalg.vector_norm(zero[:,1:]-pp[:,1:],dim=-1).mean()),
       shuffled_input_error_m=float(torch.linalg.vector_norm(shuffled[:,1:]-pp[:,1:],dim=-1).mean()),
       negative_initial_fraction=float((h[:,0]<0).float().mean()),zero_fraction_step20=float((h[:,-1]==0).float().mean()))

def export_models():
    (P2/'models').mkdir(exist_ok=True)
    if all((P2/'models'/f'{s}.pt').exists() for s in ['existing_full4096_fast','official_ganguli_pretrained']):
        return # Published models are immutable; inference and replay never rewrite checkpoints.
    old=torch.load(OLD/'runs/full4096_fast/last.pt',map_location='cpu',weights_only=False)
    centers=np.load(OLD/'runs/full4096_fast/place_cell_centers.npy')
    base=PathRNN(centers)
    s=old['model_state_dict'];mapped={'encoder.weight':s['init_position_encoder.weight'],
       'RNN.weight_ih_l0':s['RNN.weight_ih_l0'],'RNN.weight_hh_l0':s['RNN.weight_hh_l0'],
       'decoder.weight':s['decoder.weight'],'centers':torch.as_tensor(centers)}
    base.load_state_dict(mapped)
    torch.save(dict(state_dict=base.state_dict(),centers=base.centers,Ng=4096,Np=512,
        source='phase1 local 4750-step accelerated variant, not strict Schoyen reproduction',
        cfg=old['cfg'],trained_steps=old['step'],original_checkpoint_sha256=sha256(OLD/'runs/full4096_fast/last.pt')),
        P2/'models/existing_full4096_fast.pt')
    w=numpy_weights(P2/'pretrained/ganguli/example_trained_weights.npy')
    centers=np.load(P2/'pretrained/ganguli/example_pc_centers.npy')+np.float32(1.1)
    model=PathRNN(centers)
    model.load_state_dict({'encoder.weight':torch.as_tensor(w[0].T.copy()),
       'RNN.weight_ih_l0':torch.as_tensor(w[1].T.copy()),'RNN.weight_hh_l0':torch.as_tensor(w[2].T.copy()),
       'decoder.weight':torch.as_tensor(w[3].T.copy()),'centers':torch.as_tensor(centers)})
    torch.save(dict(state_dict=model.state_dict(),centers=model.centers,Ng=4096,Np=512,
       source='official Sorscher/Ganguli single-environment pretrained inference weights; not Schoyen three-environment checkpoint',
       source_commit='401dd6b5e20a754267b16eeb5bd88239b9af33e9',
       original_weights_sha256=sha256(P2/'pretrained/ganguli/example_trained_weights.npy'),
       optimizer_history_available=False,original_training_seed_known=False),P2/'models/official_ganguli_pretrained.pt')

def fps(d,n,start=0):
    n=min(n,len(d));inds=np.empty(n,int);inds[0]=start%len(d)
    nearest=d[inds[0]].copy();nearest[inds[0]]=-np.inf
    for j in range(1,n):
        inds[j]=np.argmax(nearest);nearest=np.minimum(nearest,d[inds[j]]);nearest[inds[:j+1]]=-np.inf
    return inds

if __name__=='__main__':
    export_models();dev=setup()
    for source in ['existing_full4096_fast','official_ganguli_pretrained']:
        model,ck=get_model(source,dev)
        ev=evaluate(model,72400);save_json(P2/'audit'/f'{source}_performance.json',ev)
        print(source,json.dumps(ev),flush=True)
        del model

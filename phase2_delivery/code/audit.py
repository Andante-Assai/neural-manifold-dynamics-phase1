from common import *
import importlib.util,types,importlib.metadata as md
def source_audit():
    g=json.loads((P2/'audit/ganguli-tree.json').read_text());b=json.loads((P2/'audit/band-tree.json').read_text())
    item=lambda t,p:next(x for x in t['tree'] if x['path']==p)
    rows=[]
    for name in ['example_trained_weights.npy','example_pc_centers.npy']:
        p=P2/'pretrained/ganguli'/name;h=hashlib.sha1();h.update(f'blob {p.stat().st_size}\0'.encode())
        with p.open('rb') as f:
            for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
        assert h.hexdigest()==item(g,'models/'+name)['sha']
        rows.append(dict(name=name,sha256=sha256(p),git_blob_sha=h.hexdigest(),bytes=p.stat().st_size))
    versions={'numpy':np.__version__,'torch':torch.__version__}
    for name in ['scipy','h5py','scikit-learn','umap-learn','numba','llvmlite','matplotlib','plotly','ripser']:versions[name]=md.version(name)
    save_json(P2/'audit/source_checkpoint_availability.json',dict(official_repo='https://github.com/ganguli-lab/grid-pattern-formation',
       official_commit=g['sha'],trusted_reproduction_repo='https://github.com/yuling-wu/hierarchy_band_grid',reproduction_commit=b['sha'],
       inference_checkpoint_complete=True,arrays_shapes=[[512,4096],[2,4096],[4096,4096],[4096,512]],
       copied_checkpoint_not_independent_seed=item(g,'models/example_trained_weights.npy')['sha']==item(b,'models/example_trained_weights.npy')['sha'],
       optimizer_history_available=False,original_training_seed_verified=False,original_training_budget_verified=False,
       Schoyen_three_environment_checkpoint_available=False,
       scope='Public Sorscher single-environment inference weights, not Schoyen exact checkpoint. Four no-bias matrix transposes implement the same recurrence.',
       artifacts=rows,runtime=versions))
    (P2/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n',encoding='utf8')

@torch.no_grad()
def adapter_check():
    source='official_ganguli_pretrained';model,ck=get_model(source,setup())
    spec=importlib.util.spec_from_file_location('published_model',P2/'pretrained/ganguli/model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    options=types.SimpleNamespace(Ng=4096,Np=512,sequence_length=20,weight_decay=1e-4,activation='relu')
    ref=module.RNN(options,None).cuda()
    ref.load_state_dict({k:v for k,v in model.state_dict().items() if k!='centers'},strict=True)
    pos,u=foraging(np.random.default_rng(73003),32);pt=torch.as_tensor(pos,device='cuda');ut=torch.as_tensor(u,device='cuda')
    p0=model.pc(pt[:,0]);H=model(ut,p0)
    rr=ref.g((ut[:,1:].transpose(0,1),p0)).transpose(0,1)
    difference=float((rr-H[:,1:]).abs().max());assert torch.allclose(rr,H[:,1:],atol=1e-5,rtol=1e-5)
    w=numpy_weights(P2/'pretrained/ganguli/example_trained_weights.npy')
    weights=[torch.as_tensor(x,device='cuda') for x in w];h=p0@weights[0];manual=[h]
    for t in range(1,21):h=torch.relu(h@weights[2]+ut[:,t]@weights[1]);manual.append(h)
    manual=torch.stack(manual,1);direct_delta=float((manual-H).abs().max())
    assert torch.allclose(manual,H,atol=1e-5,rtol=1e-5)
    x=H[:,-1,:64].cpu().numpy();ours=cdist(x,x);gpu=__import__('topology').distances(x)
    distance_error=float(abs(ours-gpu).max());assert distance_error<2e-4
    save_json(P2/'audit/adapter_verification.json',dict(status='passed',published_PyTorch_forward_max_error=difference,
       original_array_manual_recurrence_max_error=direct_delta,float32_distance_vs_float64_max_error=distance_error,
       coordinate_translation_m=[1.1,1.1],saved_place_centers_used=True,no_new_biases=True,no_hidden_reinitialization=True))

def model_diagnostics():
    reports=[]
    for source in ['existing_full4096_fast','official_ganguli_pretrained']:
        model,ck=get_model(source,setup());w=model.RNN.weight_hh_l0
        with torch.no_grad():
            gram=w@w.T;norm=float((w*w).sum());pr=norm**2/float((gram*gram).sum())
            _,s,_=torch.pca_lowrank(w,q=32,center=False,niter=4)
            explained=float((s*s).sum())/norm
        reports.append(dict(source=source,recurrent_Frobenius_norm_squared=norm,
           recurrent_singular_value_participation_ratio=pr,top32_singular_energy_fraction_lower_bound=explained,
           note='Weight spectrum is a diagnostic, not intrinsic manifold dimension or an isolated causal attribution.'))
        del model,w,gram
    save_json(P2/'audit/model_spectrum_diagnostics.json',reports)

if __name__=='__main__':source_audit();adapter_check();model_diagnostics();print('Source, adapter and spectrum audit passed',flush=True)

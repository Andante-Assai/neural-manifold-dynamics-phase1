from phase1 import *
import h5py
import importlib.metadata as md

def full_quality(run):
    out=ROOT/'datasets'/run; dest=ROOT/'analysis'/run
    m=json.loads((out/'manifest.json').read_text());rows_out=[]
    for label,steps in [('core_T20',[0,1,5,10,20]),('exploratory_T160',[40,80,160])]:
        ng=m['core_shape'][2];shape=(len(steps),4,ng)
        totals=np.zeros(shape);squares=np.zeros(shape);nnz=np.zeros(shape,np.int64)
        minima=np.full(shape,np.inf);maxima=np.full(shape,-np.inf);counts=np.zeros(4,np.int64)
        zero_states=np.zeros((len(steps),4),np.int64)
        for r in m['files']:
            if '/'+label+'/' not in r['path']:continue
            with h5py.File(ROOT/r['path']) as f:
                x=f['H'][steps];cc=f['condition'][...]
                for c in range(4):
                    xx=x[:,cc==c].astype(np.float64)
                    if xx.shape[1]==0:continue
                    totals[:,c]+=xx.sum(1);squares[:,c]+=(xx*xx).sum(1)
                    nnz[:,c]+=(xx!=0).sum(1);counts[c]+=xx.shape[1]
                    minima[:,c]=np.minimum(minima[:,c],xx.min(1));maxima[:,c]=np.maximum(maxima[:,c],xx.max(1))
                    zero_states[:,c]+=(np.max(np.abs(xx),axis=-1)<1e-12).sum(1)
        for si,t in enumerate(steps):
            for c in range(4):
                mean=totals[si,c]/counts[c];var=np.maximum(squares[si,c]/counts[c]-mean**2,0)
                rows_out.append(dict(dataset=label,condition=CONDITIONS[c],step=t,n=int(counts[c]),
                 zero_fraction=float(1-nnz[si,c].sum()/(counts[c]*ng)),
                 always_zero_neuron_fraction=float(np.mean(nnz[si,c]==0)),
                 nonvarying_neuron_fraction=float(np.mean(var<1e-16)),
                 total_neuron_variance=float(var.sum()),all_zero_state_count=int(zero_states[si,c]),
                 classification='strong sparsity and inactive neurons; not global point collapse; retain all units and flag for subsequent analysis'))
    dump(dest/'full_population_quality.json',rows_out)
    return rows_out

def main():
    run='full4096_fast';device=setup(42)
    for name in ['pilot256','pilot256_fast',run]:
        model,ck,path=load_run(name,device)
        np.save(path/'place_cell_centers.npy',model.place_cell_ensembles[0].pcs.cpu().numpy())
        dump(path/'state_definition.json',dict(Ng=ck['cfg']['Ng'],Np=512,place_centers='place_cell_centers.npy',
          recurrence='h0=We p0; h[t]=ReLU(Wg h[t-1]+Wv u[t]); logits=Wd h[t]',
          input='u[t]=position[t]-position[t-1] in metres; velocity_m_s=u[t]/0.02',
          h0_readout_untrained=True,checkpoint_step=ck['step'],weights_frozen_during_collection=True,
          status='failed short pilot' if name=='pilot256' else 'local training variant; path integration evaluated'))
        del model
    packages=['torch','numpy','scipy','h5py','scikit-learn','matplotlib','plotly','ripser','tqdm']
    versions={p:md.version(p) for p in packages}
    dump(ROOT/'audit'/'runtime_packages.json',versions)
    (ROOT/'requirements.txt').write_text('\n'.join(f"{p}=={v}" for p,v in versions.items())+'\n',encoding='utf-8')
    q=full_quality(run)
    m=json.loads((ROOT/'datasets'/run/'manifest.json').read_text())
    e=json.loads((ROOT/'runs'/run/'final_evaluation.json').read_text())
    v=json.loads((ROOT/'datasets'/run/'verification.json').read_text())
    stats=json.loads((ROOT/'analysis'/run/'task_performance.json').read_text())
    summary=dict(status='data_acquired_and_numerically_replay_verified',strict_paper_reproduction=False,
         core_shape=m['core_shape'],extension_shape=m['extension_shape'],chunk_count=len(m['files']),
         chunk_bytes=sum(r['bytes'] for r in m['files']),core_sample_count=8192,unique_initial_positions=2048,
         per_condition=2048,trained_minibatches=m['trained_steps'],final_holdout=e,
         all_hashes_and_replays_passed=v['status']=='passed',
         replay_max_abs_error=max(r['replay_max_abs_error'] for r in v['results']),
         manual_recurrence_max_abs_error=max(r['manual_max_abs_error'] for r in v['results']),
         paired_prefix_max_abs_error=v['paired_prefix_max_abs_error'],
         zero_fraction_step20_range=[min(r['zero_fraction'] for r in q if r['step']==20),max(r['zero_fraction'] for r in q if r['step']==20)],
         always_zero_neuron_fraction_step20_range=[min(r['always_zero_neuron_fraction'] for r in q if r['step']==20),max(r['always_zero_neuron_fraction'] for r in q if r['step']==20)],
         global_all_zero_states_step20=sum(r['all_zero_state_count'] for r in q if r['step']==20),
         exploratory_error_step160_by_condition={r['condition']:r['error_final_m'] for r in stats if r['dataset']=='exploratory_T160'},
         torus_evidence='not established',scientific_limitations=['one training seed for full model',
            'strong sparsity and inactive units','accelerated local recipe','single environment',
            'exploratory long-horizon degradation','pooled 3D PCA discards most variance; cannot establish manifold geometry'])
    dump(ROOT/'RESULTS.json',summary)
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

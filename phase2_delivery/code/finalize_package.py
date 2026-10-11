from common import *
import csv
def main():
    source='official_ganguli_pretrained';out=P2/'analysis'/source
    result=json.loads((P2/'RESULTS.json').read_text())
    assert result['acceptance_by_condition']=={'stationary':True,'random_foraging':False}
    summary=[]
    for split in ['validation','confirmation']:
        r=json.loads((out/f'v3/topology/stationary_{split}_test.json').read_text());pr=json.loads((out/f'v3/precision_audit/{split}.json').read_text())
        o=r['observed'];summary.append(dict(split=split,n=8192,neurons=483,primary_landmarks=768,
          field=47,field2_and_1024_landmarks_pass=r['field_and_resolution_pass'],second_H1_normalized_lifetime=o['second_H1_normalized_lifetime'],
          first_H2_normalized_lifetime=o['first_H2_normalized_lifetime'],beta121_interval_normalized=o['longest_beta121_interval_normalized'],
          primary_B=199,primary_p=r['p_unadjusted'],primary_Holm_p=.05,precision_B=999,precision_p=pr['p_plus_one'],
          conservative_precision_p_times10=pr['conservative_times10_p'],null_exceedances=pr['exceedances'],
          conservative_95_percent_tail_upper_times10=pr['conservative_times10_tail_upper']))
    with (P2/'STATISTICAL_SUMMARY.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    result['stationary_raw_evidence']=summary
    result['IID_intrinsic_dimensions']=json.loads((out/'geometry_support/IID_intrinsic_dimensions.json').read_text())
    result['full_step_H_shape']=[21,2048,4096];result['frozen_candidate_H_shape']=[21,2048,483]
    verification=json.loads((P2/'audit/raw_data_verification.json').read_text())
    result['raw_chunk_count']=verification['files'];result['max_hidden_replay_error']=max(r['replay_max_error'] for r in verification['records'])
    result['max_readout_replay_error']=max(r['readout_max_error'] for r in verification['records'])
    result['new_training_executed']=False;result['reason_no_new_training']='Validated public full inference weights already provide condition-specific evidence; no topology-guided retraining performed.'
    result['unique_training_checkpoints_in_comparison']=2
    result['independent_training_seed_replication_claim']=False
    save_json(P2/'RESULTS.json',result)
    # Global HDF5 index, relative chunk links, NaN when source data are unavailable.
    time=P2/'data'/source/'timewise';m=json.loads((time/'manifest.json').read_text());rows=sorted(m['files'],key=lambda r:(r['condition']!='stationary',r['path']))
    keys=['H','candidate_H','position','model_input','velocity_m_s','readout_logits','error_m']
    samples=['sample_id','initial_place_activity']
    target=time/'H_global.h5'
    with h5py.File(P2/rows[0]['path']) as first,h5py.File(target,'w',libver='latest') as f:
        for k in first.attrs:f.attrs[k]=first.attrs[k]
        for key in keys+samples:
            axis=1 if key in keys else 0;shape=list(first[key].shape);shape[axis]=2048
            layout=h5py.VirtualLayout(shape=tuple(shape),dtype=first[key].dtype);start=0
            for r in rows:
                with h5py.File(P2/r['path']) as chunk:
                    n=r['n'];v=h5py.VirtualSource((P2/r['path']).relative_to(time).as_posix(),key,shape=chunk[key].shape)
                    sl=[slice(None)]*len(shape);sl[axis]=slice(start,start+n);layout[tuple(sl)]=v;start+=n
            f.create_virtual_dataset(key,layout,fillvalue=np.nan if first[key].dtype.kind=='f' else -1)
        f.create_dataset('H_steps',data=np.arange(21));f.create_dataset('candidate_neuron_ids',data=first['candidate_neuron_ids'][...])
        f.create_dataset('condition',data=np.repeat([0,1],1024))
    with h5py.File(target) as f:
        assert f['H'].shape==(21,2048,4096) and f['candidate_H'].shape==(21,2048,483)
        assert np.isfinite(f['H'][20,:4]).all() and len(np.unique(f['sample_id'][...]))==2048
        assert np.array_equal(f['sample_id'][...],np.arange(2048))
    m['global_index']=dict(path=target.relative_to(P2).as_posix(),sha256=sha256(target),bytes=target.stat().st_size)
    save_json(time/'manifest.json',m)
    records=[]
    for p in sorted(P2.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.name not in ['SHA256SUMS','file_inventory.json']:
            records.append(dict(path=p.relative_to(P2).as_posix(),bytes=p.stat().st_size,sha256=sha256(p)))
    save_json(P2/'file_inventory.json',records)
    (P2/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in records),encoding='utf8')
    print(json.dumps({'files':len(records),'bytes':sum(r['bytes'] for r in records),'raw_chunks':verification['files'],'core_condition':'stationary only'}),flush=True)
if __name__=='__main__':main()

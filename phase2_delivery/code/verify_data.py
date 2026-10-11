from common import *
def main():
    device=setup();reports=[]
    for source in ['existing_full4096_fast','official_ganguli_pretrained']:
        model,ck=get_model(source,device)
        for manifest_path in sorted((P2/'data'/source).rglob('manifest.json')):
            m=json.loads(manifest_path.read_text());checkpoint=sha256(P2/'models'/f'{source}.pt')
            assert m['checkpoint_sha256']==checkpoint
            for r in m['files']:
                p=P2/r['path'];assert sha256(p)==r['sha256'],p
                with h5py.File(p) as f:
                    hs=f['H_steps'][...];pos=f['position'][...];u=f['model_input'][...];p0=f['initial_place_activity'][...]
                    assert np.isfinite(pos).all() and np.isfinite(u).all()
                    assert np.allclose(np.diff(pos,axis=0),u[1:],atol=3e-7)
                    assert np.allclose(p0.sum(-1),1,atol=2e-6)
                    HH=f['H'][...];YY=f['readout_logits'][...]
                    assert HH.shape[-1]==4096 and np.isfinite(HH).all() and np.isfinite(YY).all()
                    with torch.no_grad():
                        ut=torch.as_tensor(np.ascontiguousarray(u.transpose(1,0,2)),device=device)
                        pp=torch.as_tensor(p0,device=device);H=model(ut,pp)
                        replay=H[:,hs].cpu().numpy().transpose(1,0,2)
                        delta=float(np.max(abs(HH-replay)));assert np.allclose(HH,replay,atol=2e-5,rtol=3e-5),(p,delta)
                        steps=f['readout_steps'][...] if 'readout_steps' in f else np.arange(21)
                        readout=model.decoder(H[:,steps]).cpu().numpy().transpose(1,0,2)
                        outdelta=float(np.max(abs(YY-readout)));assert np.allclose(YY,readout,atol=2e-5,rtol=3e-5)
                    reports.append(dict(path=r['path'],hash_verified=True,all_neurons=True,replay_max_error=delta,readout_max_error=outdelta))
                    del H,HH,YY,replay,readout
            print('Verified',manifest_path.relative_to(P2),flush=True)
        del model
    save_json(P2/'audit/raw_data_verification.json',dict(status='passed',files=len(reports),records=reports,
       all_manifest_hashes_verified=True,all_raw_states_replayed=True,all_inputs_aligned=True))
if __name__=='__main__':main()

from common import *
import argparse
def cluster_annotations(source):
    out=P2/'analysis'/source;features=np.load(out/'cell_features.npz');rot=np.load(out/'rotation_clusters.npz')
    cells=rot['cell_ids'];labels=rot['labels'];gcs=np.load(out/'discovery_grid_scores.npy');rows=[]
    for lab in np.unique(labels):
        ix=labels==lab;theta=features['orientation_rad'][ix]
        comp=np.mean(np.exp(2j*theta));angle=(np.angle(comp)/2)%np.pi
        rows.append(dict(cluster=int(lab),n=int(ix.sum()),grid_score_median=float(np.nanmedian(gcs[cells[ix]])),
           band_concentration_median=float(np.median(features['band_concentration'][ix])),
           band_fraction=float(np.mean(features['band_mask'][ix])),orientation_coherence=float(abs(comp)),
           angle_deg=float(np.degrees(angle)),frequency_median=float(np.median(features['frequency_cycles_per_box'][ix])),
           frequency_CV=float(np.std(features['frequency_cycles_per_box'][ix])/np.mean(features['frequency_cycles_per_box'][ix]))))
    save_json(out/'discovery_cluster_annotations.json',rows)
    print(json.dumps(rows),flush=True)

def benchmark():
    dev=setup(72301);rng=np.random.default_rng(72301);pos,u=foraging(rng,200)
    center=np.load(P2/'pretrained/ganguli/example_pc_centers.npy')+1.1
    results=[]
    for mode in ['float32_strict','float32_TF32','autocast_float16']:
        model=PathRNN(center).to(dev)
        for p in model.parameters():torch.nn.init.xavier_uniform_(p)
        torch.backends.cuda.matmul.allow_tf32=mode=='float32_TF32'
        torch.backends.cudnn.allow_tf32=mode=='float32_TF32'
        pt=torch.as_tensor(pos,device=dev);ut=torch.as_tensor(u,device=dev);target=model.pc(pt)
        opt=torch.optim.Adam(model.parameters(),lr=1e-4)
        scaler=torch.amp.GradScaler('cuda',enabled=mode=='autocast_float16')
        torch.cuda.reset_peak_memory_stats();times=[]
        try:
            for i in range(12):
                torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16,enabled=mode=='autocast_float16'):
                    h=model(ut,target[:,0]);pred=model.decoder(h[:,1:])
                    ce=-(target[:,1:]*torch.log_softmax(pred.float(),-1)).sum(-1).mean()
                    loss=ce+1e-4*(model.RNN.weight_hh_l0.float()**2).sum()
                scaler.scale(loss).backward();scaler.step(opt);scaler.update();torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
            row=dict(mode=mode,median_step_s=float(np.median(times[3:])),finite=bool(torch.isfinite(loss)),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        except Exception as e:row=dict(mode=mode,error=str(e))
        results.append(row);print(json.dumps(row),flush=True)
        del model,opt;torch.cuda.empty_cache()
    save_json(P2/'audit/training_precision_benchmark.json',results)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['annotations','benchmark']);p.add_argument('--source');a=p.parse_args()
    benchmark() if a.stage=='benchmark' else cluster_annotations(a.source)

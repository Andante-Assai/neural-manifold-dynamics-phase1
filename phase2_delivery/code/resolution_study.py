"""Diagnostic sampling-resolution study; cannot retroactively overturn the v2 acceptance rule."""
from topology import *
def main():
    source='official_ganguli_pretrained';out=P2/'analysis'/source
    cells=np.load(out/'paper_cluster_candidate_v2.npy');train=np.load(out/'discovery_states_step20.npy',mmap_mode='r')
    mean,rms=transform_fit(train,cells);dest=out/'v2/resolution_diagnostics';dest.mkdir(exist_ok=True)
    reports=[]
    for condition in ['stationary','random_foraging']:
        for split in ['validation','confirmation']:
            raw=np.load(out/f'v2/{split}_{condition}_states.npy',mmap_mode='r');x=(raw[:,cells]-mean)/rms
            for n in [768,1024]:
                r,ds,ii=ph(x,n=n,coeff=47);r.update(condition=condition,split=split,diagnostic_only=True)
                write_dgms(dest/f'{condition}_{split}_n{n}.npz',ds,ii)
                reports.append(r);save_json(dest/'results.json',reports)
                print('resolution diagnostic',condition,split,n,json.dumps(r),flush=True)
if __name__=='__main__':setup();main()

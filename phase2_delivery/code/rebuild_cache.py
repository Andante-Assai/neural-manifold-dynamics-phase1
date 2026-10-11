"""Rebuild convenient endpoint caches from immutable supplied raw HDF5, no trajectory regeneration."""
from common import *
from identify import maps_and_states
def main():
    for source in ['existing_full4096_fast','official_ganguli_pretrained']:
        out=P2/'analysis'/source;out.mkdir(exist_ok=True)
        for split in ['discovery','validation','confirmation']:
            _,_,states,ids=maps_and_states(source,split)
            np.save(out/f'{split}_states_step20.npy',states);np.save(out/f'{split}_sample_ids.npy',ids)
    source='official_ganguli_pretrained';out=P2/'analysis'/source
    for version in ['v2','v3']:
        m=json.loads((P2/'data'/source/version/'manifest.json').read_text())
        for split in ['validation','confirmation']:
            for condition in ['stationary','random_foraging']:
                states=[]
                for r in m['files']:
                    if r['split']!=split or r['condition']!=condition:continue
                    with h5py.File(P2/r['path']) as f:
                        ix=int(np.flatnonzero(f['H_steps'][...]==20)[0]);states.append(f['H'][ix])
                np.save(out/f'{version}/{split}_{condition}_states.npy',np.concatenate(states))
    print('Endpoint caches rebuilt from raw tensors',flush=True)
if __name__=='__main__':main()

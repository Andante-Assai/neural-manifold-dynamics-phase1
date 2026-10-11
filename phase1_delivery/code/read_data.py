"""Minimal CPU-only consumer. H[s, n, neuron] retains original neuron identity."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
deps=ROOT.parent/'work'/'deps'
if deps.exists():sys.path.insert(0,str(deps))
import h5py
import numpy as np

def open_core(run='full4096_fast'):
    return h5py.File(ROOT/'datasets'/run/'core_T20.h5','r')

def trajectory(sample_id,run='full4096_fast',extended=False):
    name='exploratory_T160.h5' if extended else 'core_T20.h5'
    with h5py.File(ROOT/'datasets'/run/name) as f:
        columns=np.flatnonzero(f['sample_id'][...]==sample_id)
        if len(columns)!=1:raise KeyError(sample_id)
        i=int(columns[0])
        return {key:f[key][:,i] for key in ['H','position','model_input','velocity_m_s','output_logits','output_probability','error_m']}

if __name__=='__main__':
    with open_core() as f:
        print('H global shape:',f['H'].shape,'dtype:',f['H'].dtype)
        print('Fixed-step ensemble: H[20, :128, :] =',f['H'][20,:128,:].shape)
        print('Paired trajectory: H[:, 0, :] =',f['H'][:,0,:].shape)
        print('sample_id order is authoritative; match IDs when using the extension.')

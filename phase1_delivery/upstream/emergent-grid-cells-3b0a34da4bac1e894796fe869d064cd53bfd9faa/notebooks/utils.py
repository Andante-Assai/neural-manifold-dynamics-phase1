import sys
import pickle
import tqdm
import torch
import scipy
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import coo_matrix

# note addition to path
sys.path.append("../src") if "../src" not in sys.path else None 

import pickle

import numpy as np
import matplotlib.pyplot as plt
import umap
from umap.umap_ import compute_membership_strengths, smooth_knn_dist
from sklearn.decomposition import PCA

from Experiment import Experiment
from Models import SorscherRNN
from methods import filenames

def load_experiment(path, name):
    experiment = Experiment(name = name, base_path = path)
    experiment.setup()
    return experiment

def load_model(experiment):
    # Load a model attached to an experiment
    loc = experiment.paths
    # load weights
    checkpoint_filenames = filenames(loc['checkpoints'])
    # load model latest (wrt. #epochs trained)
    print(f"Loading model at epoch = {checkpoint_filenames[-1]}", loc['checkpoints'] / checkpoint_filenames[-1])
    checkpoint = torch.load(loc['checkpoints'] / checkpoint_filenames[-1], map_location= "cpu")
    # instantiate trained model this time
    model = SorscherRNN(experiment.pc_ensembles, Ng=experiment.params['Ng'], Np=experiment.params['Np'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_ratemaps(experiment):
    # load grid scores + ratemaps
    ratemaps = []
    for env_i in range(len(experiment.environments)):
        with open(experiment.paths['ratemaps'] / f'env_{env_i}' / 
                  filenames(experiment.paths['ratemaps'] / f'env_{env_i}')[-1], "rb") as f:
            ratemaps.append(pickle.load(f))
    ratemaps = np.array(ratemaps)
    return ratemaps

def create_trajectories(dataset, environment_idx, num_trajectories=1500):
    """Generate trajectories to compute ratemaps with (scipy.stats.binned_statistics_2d)"""
    batch_velocities, batch_init_pc_positions, batch_positions = [], [], []
    for _ in range(num_trajectories):
        (velocities, init_pc_positions), _, positions, _ = dataset[environment_idx]
        batch_velocities.append(velocities)
        batch_init_pc_positions.append(init_pc_positions)
        batch_positions.append(positions)
    batch_inputs = [torch.stack(batch_velocities), torch.stack(batch_init_pc_positions)]
    batch_positions = torch.stack(batch_positions).detach().numpy()
    batch_velocities = torch.stack(batch_velocities).detach().numpy()
    return batch_inputs, batch_positions[:,1:], batch_velocities

def run_model(model, dataset, envs, samples = 1500, start = 0, stop = 20):
    # run model in inference mode on samples generated from dataset across environments.
    """
    model: SorscherRNN model
    dataset: dataset, passed to create_trajectories
    envs: iterable; e.g. list of environment indices. 
    samples: number of trajectories to run model on
    start: start index of returned timeseries; used to skip initial states
    """
    
    activities = []
    r = [] # positions
    v = [] # velocities

    for env in tqdm.tqdm(envs):
        batch_inputs, batch_pos, batch_v = create_trajectories(dataset, env, num_trajectories = samples)
        g = model.g(batch_inputs).detach().cpu().numpy()[:,start:stop]
        g = g.reshape(-1, g.shape[-1])
        activities.append(g)
        r.append(np.reshape(batch_pos[:,start:stop], (-1, batch_pos.shape[-1])))
        v.append(np.reshape(batch_v[:,start:stop], (-1, batch_v.shape[-1])))

    activities = np.stack(activities, axis=0)
    r = np.stack(r, axis=0)
    v = np.stack(v, axis=0)
    return activities, r, v

def create_ratemaps(g, r, res):
    ratemaps = scipy.stats.binned_statistic_2d(r[:,0], r[:,1], g.T, bins=res)[0]
    return ratemaps

def downsample_pointcloud(data,  k = 10, num_sample = 500, metric = 'euclidean'):
    """
    Adapted from Gardner et al Repo; uses methods from UMAP
    """
    n = data.shape[0]
    
    X = squareform(pdist(data, metric)).astype("float32")
    knn_indices = np.argsort(X)[:, :k]
    knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy().astype("float32")

    sigmas, rhos = smooth_knn_dist(knn_dists, k, local_connectivity=0)
    rows, cols, vals, _  = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    result = coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    prod_matrix = result.multiply(transpose)
    result = (result + transpose - prod_matrix)
    result.eliminate_zeros()
    X = result.toarray()
    F = np.sum(X,1)
    Fs = np.zeros(num_sample)
    Fs[0] = np.max(F)
    i = np.argmax(F)
    
    inds_all = np.arange(n)
    inds_left = inds_all>-1
    inds_left[i] = False
    inds = np.zeros(num_sample, dtype = int)
    inds[0] = i
    
    for j in np.arange(1,num_sample):
        F += X[i]
        Fmax = np.argmax(F[inds_left])
        Fs[j] = F[Fmax]
        i = inds_all[inds_left][Fmax]
        
        inds_left[i] = False
        inds[j] = i
    d = np.zeros((num_sample, num_sample))
    
    for j,i in enumerate(inds):
        d[j,:] = X[i, inds]
    return inds, d, Fs

def plot_barcodes(dgms, delta = 0.01, **kwargs):
    fig, ax = plt.subplots(len(dgms), 1, figsize = (3, 4))
    for i, dgm in enumerate(dgms):
        birth = dgm[:,0]
        death = dgm[:,1]
        
        first_birth = np.amin(birth[~np.isinf(death)])
        last_death = np.amax(death[~np.isinf(death)])
        infty = last_death + (last_death - first_birth)*delta
        
        w = death - birth
        y = np.linspace(first_birth, last_death, len(dgm))
        h = 1/len(dgm)
        ax[i].barh(y, height = h, width = w, left = birth, **kwargs)
    return fig, ax
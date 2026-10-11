import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import scipy
import umap
from sklearn.decomposition import PCA
import pickle
import tqdm

import os

from ratsimulator import Agent


def filenames(path):
    """Return (sorted) names of all files in given path"""
    fnames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    fnames.sort()
    return fnames


def compute_ratemaps(
    model,
    dataset,
    num_trajectories=1000,
    res=np.array([20, 20]),
    idxs=slice(0, 64, 1),
    **kwargs,
):
    activities, x, y = [], [], []
    for _ in tqdm.trange(num_trajectories):
        activities.append(model(dataset[0][0])[0, :, idxs].detach().cpu().numpy())
        x.append(dataset.agents[0].positions[1:, 0])
        y.append(dataset.agents[0].positions[1:, 1])
    activities = np.array(activities)
    return scipy.stats.binned_statistic_2d(
        np.ravel(x),
        np.ravel(y),
        activities.reshape(-1, activities.shape[-1]).T,
        bins=res,
        **kwargs,
    )


def multicontourf(xx, yy, zz, titles=None, axs=None):
    """plot multiple contourf plots on a grid"""
    if axs is None:
        ncells = int(np.sqrt(zz.shape[0]))
        fig, axs = plt.subplots(
            figsize=(10, 10), nrows=ncells, ncols=ncells, squeeze=False
        )
    else:
        fig = None
        ncells = axs.shape[0]

    # plot response maps using contourf
    for k in range(zz.shape[0]):
        ax = axs[k // ncells, k % ncells]
        ax.axis("off")
        if titles is not None:
            ax.set_title(f"{titles[k]}")
        # ax.set_aspect('equal')
        ax.contourf(xx, yy, zz[k], cmap="jet")

    return fig, axs


def multiimshow(zz, nrows=None, ncols=None, axs=None, titles=None, figsize=None, figsize_i=(0.7, 0.7), normalize=True, **kwargs):
    """plot multiple imshow plots on a grid"""
    if axs is None:
        if nrows is not None:
            # choose the size of one axes of the grid layout, filling
            # in the other wrt. the amount of images to include
            ncols = int(np.ceil(zz.shape[0] / nrows))
        elif ncols is not None:
            nrows = int(np.ceil(zz.shape[0] / ncols))
        else:
            nrows = int(np.ceil(np.sqrt(zz.shape[0])))
            ncols = int(round(np.sqrt(zz.shape[0])))
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize if figsize is not None else np.array(figsize_i) * np.array([ncols,nrows]),
            #squeeze=False,
            **kwargs,
        )
        [ax.axis('off') for ax in axs.flat]
    else:
        fig = None
        nrows, ncols = axs.shape

    # plot response maps using imshow
    # get vmin, vmax 
    vmin, vmax = np.nanmin(zz), np.nanmax(zz)
    for k in range(zz.shape[0]):
        ax = axs[k // ncols, k % ncols]
        if normalize:
            ax.imshow(zz[k], vmin=vmin, vmax=vmax)
        else:
            ax.imshow(zz[k])
        if titles is not None:
            ax.set_title(f"{titles[k]}")
    return fig, axs


def multiimshow2(zz, figsize=(1,1), normalize=True, add_colorbar=True, rect=(0,0,1,0.87), axes_pad=0, **kwargs):
    # prepare figure
    ncols = int(np.ceil(np.sqrt(zz.shape[0])))
    nrows = int(round(np.sqrt(zz.shape[0])))
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure(figsize=figsize)
    if add_colorbar and normalize:
        grid = ImageGrid(fig, rect=rect, nrows_ncols=(nrows, ncols), axes_pad=axes_pad, cbar_mode='single', cbar_location='right', cbar_pad=0.1, cbar_size='5%')
    else:
        grid = ImageGrid(fig, rect=rect, nrows_ncols=(nrows, ncols), axes_pad=axes_pad)
    vmin, vmax = (np.nanmin(zz), np.nanmax(zz)) if normalize else (None, None)
    # plot response maps using imshow
    for ax, data in zip(grid, zz):
        im = ax.imshow(data, vmin=vmin, vmax=vmax, **kwargs)
    [ax.axis('off') for ax in grid]
    fig.colorbar(im, cax=grid.cbar_axes[0]) if (normalize and add_colorbar) else None
    return fig, grid.axes_all


def PCA_UMAP(states, n_neighbors, metric = "cosine", min_dist = 0.8):
    """ Run PCA followed by UMAP on states; Similar to Gardner et al. (2022)
    
    states.shape = (Nsamples, Nfeatures)
    PCA down to 6 features/principal components 
    UMAP of PCA componenents to 3 features.
    
    Note: Gardner et al. use n_neighbors = 5000 (more than number of samples in our case...)
    """
    pca_fit = PCA(n_components = 6).fit(states)
    pca_result = pca_fit.transform(states)
    umap_fit = umap.UMAP(n_components = 3, min_dist = min_dist,  # Almost Gardner et al. params
                         n_neighbors = n_neighbors, metric = metric, init = "spectral")
    umap_result = umap_fit.fit_transform(pca_result)
    return pca_result, umap_result


def interpolate_missing_pixels(
    image: np.ndarray, mask: np.ndarray, method: str = "cubic", fill_value: int = 0
):
    """
    Taken from:
    https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python

    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """
    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y),
        known_v,
        (missing_x, missing_y),
        method=method,
        fill_value=fill_value,
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def _find_inner_circle_dist(sac, topk=10):
    """
    Helper function for grid_score()
    Args:
        sac: autocorrellogram
        topk: number of peak transitions to look for (in any direction)
    Returns:
        robust_dist: robust (median) l2-dist to transition peaks from center
        idxs: the idxs of the peaks used to calculate the (robust) dist to transition peaks
    """
    center_x = sac.shape[0] / 2
    center_y = sac.shape[1] / 2
    # smooth sac
    smooth_sac = signal.correlate2d(
        sac, np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), mode="same"
    )
    # differential filtering (could experiment with different filter widths for stability)
    dx_sac = signal.correlate2d(
        smooth_sac, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), mode="same"
    )
    dy_sac = signal.correlate2d(
        smooth_sac, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), mode="same"
    )
    # double differentiantial filtering
    dx2_sac = dx_sac = signal.correlate2d(
        dx_sac, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), mode="same"
    )
    dy2_sac = signal.correlate2d(
        dy_sac, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), mode="same"
    )

    # find peak transitions from center mass
    idxs = np.argpartition((dx2_sac + dy2_sac).flatten(), -topk)[-topk:]
    idxs = np.unravel_index(idxs, sac.shape)
    idxs = np.stack(idxs, axis=-1)  # vectorize idxs
    """
    # plot points in transition plot (double-differentiated sac)
    fig,ax = plt.subplots()
    ax.imshow(dx2_sac + dy2_sac)
    for idx in idxs:
        ax.add_patch(plt.Circle(idx, 0.2, color='r'))
    """
    # calculate distances
    center = np.array([center_x, center_y])
    l2squared_to_center = np.sum((idxs - center) ** 2, axis=-1)
    # choose distance
    robust_dist = np.median(l2squared_to_center)
    return robust_dist, idxs


def _get_annulus_mask(sac):
    """
    Helper function for grid_score()
    """
    # get center coords and create ogrid
    center_x = sac.shape[0] / 2
    center_y = sac.shape[1] / 2
    x, y = np.ogrid[0 : sac.shape[0], 0 : sac.shape[1]]
    # create annulus mask (outer circle fills the square autocorrellogram)
    outer_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= min(
        center_x**2, center_y**2
    )
    inner_circle_dist, _ = _find_inner_circle_dist(sac)  # automatic inner circle mask
    inner_mask = (x - center_x) ** 2 + (y - center_y) ** 2 >= inner_circle_dist
    return inner_mask * outer_mask


def grid_score(rate_map):
    """
    Self-made grid-score function. Grid score is not standardized and thus,
    different grid_score functions give varying results.
    """
    # autocorrelate
    sac = signal.correlate2d(rate_map, rate_map, mode="full")
    annulus_mask = _get_annulus_mask(sac)
    masked_sac = sac[annulus_mask]

    # correlate with rotated sacs
    angles = np.arange(30, 180, 30)
    distributed_image_transform = lambda angle: np.corrcoef(
        scipy.ndimage.rotate(sac, angle=angle, reshape=False)[annulus_mask].flatten(),
        masked_sac.flatten(),
    )[0, 1]
    masked_rot_sacs = np.array([*map(distributed_image_transform, angles)])
    phase60 = masked_rot_sacs[1::2]
    phase30 = masked_rot_sacs[::2]
    return np.min(phase60) - np.max(phase30)

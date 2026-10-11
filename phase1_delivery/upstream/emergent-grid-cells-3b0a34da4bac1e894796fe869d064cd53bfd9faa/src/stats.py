import numpy as np
import scipy
from scipy.special import i0
import os
import pickle
from scipy.signal import correlate
from scipy import interpolate
import tqdm
from sklearn.cluster import KMeans

from Experiment import Experiment
from synthetic_grid_cells import rotation_matrix


def find_peaks(image):
    """
    Taken from cinpla/spatial-maps. But corrects center.

    Find peaks sorted by distance from center of image.
    Returns
    -------
    peaks : array
        coordinates for peaks in image as [row, column], iow (y,x)
    """
    import scipy.ndimage as ndimage
    import scipy.ndimage.filters as filters

    image = image.copy()
    image[~np.isfinite(image)] = 0
    image_max = filters.maximum_filter(image, 3)
    is_maxima = image == image_max
    labels, num_objects = ndimage.label(is_maxima)
    indices = np.arange(1, num_objects + 1)
    peaks = ndimage.maximum_position(image, labels=labels, index=indices)
    peaks = np.array(peaks)
    center = (np.array(image.shape) - 1) / 2
    distances = np.linalg.norm(peaks - center, axis=1)
    peaks = peaks[distances.argsort()]
    return peaks


def calculate_phase_shift(rate_map1, rate_map2, boxsize):
    """
    This function calculates the phase shift between two rate maps.

    :param rate_map1: 2D array, first rate map.
    :param rate_map2: 2D array, second rate map.
    :param boxsize: tuple or list, physical dimensions of the box.
    :return: array, dominant peak indicating phase shift of the patterns.
    """
    # Perform cross correlation
    cross_corr_2d = correlate(rate_map1, rate_map2, mode="same")
    if np.isclose(np.std(cross_corr_2d), 0.0):
        return np.array([np.nan, np.nan])
    # Find dominant peak in cross correlation, indicating the center of pattern
    peak = find_peaks(cross_corr_2d)[0]
    # center peak
    peak = peak - (np.array(rate_map1.shape) - 1) / 2
    # convert pixel peak to physical peak
    peak = peak / (np.array(rate_map1.shape) - 1)  # range [-0.5, 0.5]
    # peak is given in (y,x) image coordinates, but we want (x,y) coordinates, so we swap
    peak = peak[::-1] * np.array(boxsize)
    return peak


def grid_score_masks(
    experiment: Experiment, percentile: float = 0.4, mode="intersection"
) -> np.array:
    """
    Generate masks for relevant neurons based on their grid score in all environments

    Args:
        experiment (Experiment):    Experiment object containing information about data
        percentile (float):         percentile threshold for discrimination
                                        default: 0.4 =~ top 30 %
        mode (str):                 Key word specifying mode of operation
                                    default:    'intersection' - intersection set: select neurons that
                                                                 suffice percentile threshold in all
                                                                 environments
                                                'union'        - union set: select neurons that suffice
                                                                 percentile threshold in any environment
                                                'separate'     - get masks for each environment separately

    Val:
        mask (np.array):            Mask for the given experiment
                                        shape: (Nx, Ny)
    """

    def load_gc_scores(experiment: Experiment) -> np.array:
        """
        Load the grid cell scores associated with the experiment
        for different environments and return them in one single
        array object

        Args:
            experiment (Experiment):    Experiment object containing the data

        Val:
            gc_scores (np.array):       Grid cell scores for all environments
                                            shape: (Number of Environments, Nx, Ny)
        """

        gc_scores = []
        score_filenames = os.listdir(experiment.paths["grid_scores"])
        for fname in score_filenames:
            if not "novel" in fname:
                with open(experiment.paths["grid_scores"] / f"{fname}", "rb") as f:
                    gc_scores.append(pickle.load(f))

        return np.array(gc_scores)

    grid_cell_scores = load_gc_scores(experiment)
    # Shift grid cell scores to > 0
    shifted_gc_scores = grid_cell_scores - np.amin(grid_cell_scores, axis=-1)[:, None]
    # determine neurons that belong to the selection
    sorted_gc_scores = np.sort(shifted_gc_scores, axis=-1)
    cummulative_distribution = np.cumsum(sorted_gc_scores, axis=-1) / np.sum(
        shifted_gc_scores, axis=-1, keepdims=True
    )
    cum_mask = cummulative_distribution > percentile
    no_envs = len(experiment.environments)
    thresholds = [
        np.amin(sorted_gc_scores[env_i, cum_mask[env_i]]) for env_i in range(no_envs)
    ]
    thresholds += np.amin(grid_cell_scores, axis=-1)  # shift back
    masks = np.array(  # select for scores higher than thresholds per environment
        [grid_cell_scores[env_i] > thresholds[env_i] for env_i in range(no_envs)]
    )
    if mode == "separate":
        return masks
    if mode == "intersection":
        # check if apparent in all envs
        return np.prod(masks, axis=0).astype("bool")
    if mode == "union":
        # ehck if apparent in any env
        return np.sum(masks, axis=0).astype("bool")
    # if key word unknown return None
    return None


def circular_kernel(data, kappa):
    """Generate a von Mises kernel for kernel density estimation
    Based on https://www.stata.com/meeting/mexico14/abstracts/materials/mex14_salgado.pdf
    Call this function to obtain a kernel, which can then subsequently be used on new data

    Args:
        data (np.array): circular data to be used for the kernel; shape (N,).
        kappa (float): shape parameter, sets smoothness of KDE
    """

    def kernel(x):
        """Create the actual kernel to be returned

        Args:
            x (np.array): circular data to be evaluated vs kernel; shape (M,)
        """
        von_mises = (
            1
            / (2 * np.pi * i0(kappa))
            * np.exp(kappa * np.cos(x[:, None] - data[None, :]))
        )
        return np.mean(von_mises, axis=-1) / np.trapz(
            np.mean(von_mises, axis=-1), x=x
        )  # sum over data dimension

    return kernel


def fill_nan(ratemap, **kwargs):
    # Get the indices of where the NaNs are
    y_nan, x_nan = np.where(np.isnan(ratemap))
    # Get the indices of where the NaNs aren't
    y_not_nan, x_not_nan = np.where(~np.isnan(ratemap))
    # Get the non-NaN values
    ratemap_not_nan = ratemap[y_not_nan, x_not_nan]
    # Interpolate
    ratemap[y_nan, x_nan] = interpolate.griddata(
        (y_not_nan, x_not_nan),  # points we know
        ratemap_not_nan,  # values we know
        (y_nan, x_nan),  # points to interpolate
        **kwargs,  # fill with nearest values
    )
    return ratemap


def mad(x, axis=None):
    """
    mean absolute deviation
    """
    return np.median(np.abs(x - np.median(x, axis)), axis)


def grid_spacing(ratemap, boxsize: tuple = (2.2, 2.2), p=0.1, verbose=False, **kwargs):
    """
    Calculate the median distance to all 6 nearest peaks from center peak.
    """
    autocorr = scipy.signal.correlate(ratemap, ratemap, **kwargs)
    peaks = find_peaks(autocorr)  # px, range = [0, autocorr.shape - 1]
    # indicate the origin as the DC peak of the autocorrelation
    origin = peaks[0]  # px
    # shift peak coordinates relatively to origin
    peaks = (
        peaks - origin
    )  # px, range = [-(autocorr.shape-1) / 2, (autocorr.shape-1) / 2]
    # make dimensionless peak coordinates
    peaks = peaks / ((np.array(autocorr.shape) - 1) / 2)  # dimless, range = [-1, 1]
    # rescale to physical box dimensions
    peaks *= np.array(boxsize)  # m, range = [-boxsize, boxsize]

    idx = num_closest_isodistance_points(ratemap)
    if idx is None:
        return np.nan, np.nan
    hexagonal_dist = np.linalg.norm(peaks[1 : idx + 1], axis=-1)
    median_dist = np.median(hexagonal_dist)
    sigma = mad(hexagonal_dist)
    if sigma > p and verbose:
        print(f"{sigma=}>{p=}. Grid spacing might NOT be ROBUST")

    return median_dist, sigma


def grid_orientation(ratemap, **kwargs):
    """
    calculate orientation of grids in ratemap

    also calculate difference between angles (dangles)
    """
    autocorr = scipy.signal.correlate(ratemap, ratemap, **kwargs)
    center = (np.array(autocorr.shape) - 1) / 2
    peaks = find_peaks(autocorr)
    peaks[:, 0] = center[0] - peaks[:, 0]
    # peaks = peaks[:, ::-1] arctan2 takes (y, x), which is why we don't change axis sequence here
    idx = num_closest_isodistance_points(ratemap)
    if idx is None:
        return np.nan, np.nan
    centered_peaks = peaks[1 : idx + 1] - peaks[0]
    angles = np.sort(np.arctan2(*centered_peaks.T) % (2 * np.pi))  # [::-1]
    dangles = np.diff(angles)
    return angles[0], dangles


def num_closest_isodistance_points(ratemap, **kwargs):
    """
    Determine the number of grid points that are closest to the center
    """
    autocorr = scipy.signal.correlate(ratemap, ratemap, **kwargs)
    peaks = find_peaks(autocorr)
    if len(peaks) < 2:
        # need multiple peaks
        return None
    dists = np.linalg.norm(
        peaks[1:] - peaks[0], axis=1
    )  # shift coordinate system to the center
    # dists = scipy.ndimage.correlate(dists, np.array([1,2,1]))
    dddx = scipy.signal.correlate(dists, np.array([-1, 1]), mode="valid")
    outliers = dddx > np.mean(dddx) + 2 * np.std(dddx)
    return np.argmax(outliers) + 1  # get the index of the first "True" occurence


def make_disk_mask(size):
    """Create a 2D mask of a disk (filled circle) for a square matrix of a given size"""
    # Create a coordinate grid
    y, x = np.ogrid[-size / 2 : size / 2, -size / 2 : size / 2]
    # Create the disk mask
    mask = x**2 + y**2 <= ((size - 1) / 2) ** 2
    return mask


def calculate_orientation_shift(ratemap1, ratemap2, rotation_res=360):
    """
    calculate orientation shift between two ratemaps
    """
    autocorr1 = scipy.signal.correlate(ratemap1, ratemap1, mode="same")
    autocorr2 = scipy.signal.correlate(ratemap2, ratemap2, mode="same")
    # normalize the autocorrelations
    autocorr1 = autocorr1 / np.sum(autocorr1)
    autocorr2 = autocorr2 / np.sum(autocorr2)
    # slice out the largest disk that fits in the autocorrelation
    disk_mask = make_disk_mask(autocorr1.shape[0])
    rotations = np.linspace(-np.pi, np.pi, rotation_res)
    scores = []
    for rotation in rotations:
        rotated_autocorr2 = scipy.ndimage.rotate(
            autocorr2, np.degrees(rotation), reshape=False, order=0
        )
        # score = np.sum(autocorr1[disk_mask] * rotated_autocorr2[disk_mask])
        score = np.corrcoef(autocorr1[disk_mask], rotated_autocorr2[disk_mask])[0, 1]
        scores.append(score)
    # return np.argmax(scores) / rotation_res * 2 * np.pi
    return rotations[np.argmax(scores)]


def cluster_ratemaps(ratemaps, n_clusters=3):
    """
    Cluster ratemaps based on their correlation maps using Kmeans clustering

    Note! This function requires knowing the number of clusters in advance

    Parameters:
        ratemaps (numpy.ndarray): 3D array of ratemaps (ncells, res, res)
    Returns:
        clusters (numpy.ndarray): 1D array of cluster labels
    """
    ncells, res, _ = ratemaps.shape
    # Create an empty array to store the correlation maps
    corr_maps = np.empty((ncells, res, res))
    # Generate correlation maps
    for i in range(ncells):
        corr_maps[i] = correlate(ratemaps[i], ratemaps[i], mode="same")
    # Calculate correlation coefficients between different correlation maps
    corr_coef_matrix = np.empty((ncells, ncells))
    for i in tqdm.trange(ncells):
        for j in range(ncells):
            corr_coef_matrix[i, j] = np.corrcoef(
                corr_maps[i].ravel(), corr_maps[j].ravel()
            )[0, 1]
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(corr_coef_matrix)
    return clusters

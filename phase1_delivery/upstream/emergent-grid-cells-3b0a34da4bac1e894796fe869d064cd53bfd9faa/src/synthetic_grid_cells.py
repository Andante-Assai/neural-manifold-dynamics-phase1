import numpy as np
import numpy.ma as ma
import numpy.linalg as npl
import matplotlib.pyplot as plt


def rotation_matrix(theta, degrees=True, **kwargs) -> np.ndarray:
    """
    Creates a 2D rotation matrix for theta
    Parameters
    ----------
    theta : float
        angle offset wrt. the cardinal x-axis
    degrees : boolean
        Whether to use degrees or radians
    Returns
    -------
    rotmat : np.ndarray
        the 2x2 rotation matrix
    Examples
    --------
    >>> import numpy as np
    >>> x = np.ones(2) / np.sqrt(2)
    >>> rotmat = rotation_matrix(45)
    >>> tmp = rotmat @ x
    >>> eps = 1e-8
    >>> np.sum(np.abs(tmp - np.array([0., 1.]))) < eps
    True
    """
    # convert to radians
    theta = theta * np.pi / 180 if degrees else theta
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


class GridModule:
    """
    Grid cell pattern constructed from three interacting 2D (plane) vectors
    with 60 degrees relative orientational offsets following the paper:
        "From grid cells to place cells: A mathematical model" Solstad2005
    """

    def __init__(self, ncells, center=np.zeros(2), orientation_offset=0, f=1):
        self.center = center
        self.orientation_offset = orientation_offset
        self.f = f
        self.unit_cell = Hexagon(2 / (3*f), orientation_offset, center)
        self.phases = self.unit_cell.sample(ncells)

        # init k-vectors
        relative_R = rotation_matrix(60)
        init_R = rotation_matrix(orientation_offset)
        k1 = np.array([1.0, 0.0])  # init wave vector. unit length in x-direction
        k1 = init_R @ k1
        ks = np.array([npl.matrix_power(relative_R, k) @ k1 for k in range(3)])
        ks *= (
            2 * np.pi * f
        )  # spatial angular frequency (unit-movement in space is one period)
        self.ks = ks

    def __call__(self, rs):
        """
        Parameters:
            r (nsamples,2): spatial samples
            dp (ncells,2): optional phase (jitter) shift
        Returns:
            activity (nsamples,ncells): activity of all cells on spatial samples
        """
        activity = np.cos((rs[:, None] - self.phases[None]) @ self.ks.T)
        activity = np.sum(activity, axis=-1)  # sum plane waves
        activity = (2 / 3) * (
            activity / 3 + 0.5
        )  # Solstad2006 rescaling, range: [-1.5,3] -> [0,1]
        return activity


class Hexagon:
    def __init__(self, radius, orientation_offset, center):
        self.radius = radius
        self.orientation_offset = orientation_offset
        self.center = center
        self.area = 3 * np.sqrt(3) * radius * radius / 2

        # create hexagonal points
        rotmat60 = rotation_matrix(60, degrees=True)
        rotmat_offset = rotation_matrix(orientation_offset, degrees=True)
        hpoints = np.array([radius, 0])  # start vector along cardinal x-axis
        hpoints = rotmat_offset @ hpoints
        hpoints = [hpoints]
        for _ in range(5):
            hpoints.append(rotmat60 @ hpoints[-1])
        self.hpoints = np.array(hpoints)
        self.basis = (np.sqrt(3) * self.hpoints / 2) @ rotation_matrix(30)

    def is_in_hexagon(self, rs):
        """
        Check if a set of points rs is within hexagon.

        Parameters:
            rs (nsamples,2): points to check if are inside hexagon
        Returns:
            in_hexagon (nsamples,): mask array
        """
        projections = (rs - self.center) @ self.basis.T  # (nsamples,2)
        # all basis vectors have equal length
        in_hexagon = np.max(projections, axis=-1) <= np.sum(self.basis[0] ** 2)
        return in_hexagon

    def sample(self, N, seed=None):
        """
        Vectorized uniform rejection sampling of hexagon using a proposal domain
        define by the minimal enclosing square of the minimal enclosing circle
        of the hexagon.

        Parameters:
            N: (int) number of points to sample
            seed: (int) rng seed
        Returns:
            samples (nsamples,2): array of 2d hexagonal uniform samples
        """
        # sample points within hexagon
        rng = np.random.default_rng(seed)
        missing_samples = N
        samples = np.zeros((N, 2))
        while missing_samples != 0:
            sample_square = (
                rng.uniform(-self.radius, self.radius, size=(missing_samples, 2))
                + self.center
            )
            in_hexagon = self.is_in_hexagon(sample_square)
            sample_square = sample_square[in_hexagon]
            samples[
                (N - missing_samples) : (N - missing_samples) + sample_square.shape[0]
            ] = sample_square
            missing_samples -= sample_square.shape[0]
        return samples

    def plot(self, fig=None, ax=None, center=None, colors=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        center = self.center if center is None else center
        hpoints = self.hpoints + center
        for i in range(len(hpoints)):
            line_segment = np.stack([hpoints[i], hpoints[(i + 1) % 6]])
            if not (colors is None):
                ax.plot(*line_segment.T, color=colors[i], **kwargs)
            else:
                ax.plot(*line_segment.T, **kwargs)
        # ax.set_aspect("equal")
        return fig, ax
    

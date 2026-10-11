import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def scatter3d(data, tags, ncols=2, nrows=2, s=1, alpha=0.5, azim_elev_title=True, edgecolor='none', **kwargs):
    assert data.shape[-1] == 3, "data must have three axes. No more, no less."
    if data.ndim > 2:
        data = data.reshape(-1, 3)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, subplot_kw={"projection": "3d"}, **kwargs)
    num_plots = ncols * nrows
    
    azims = np.linspace(0, 180, ncols + 1)[:-1]
    elevs = np.linspace(0, 90, nrows + 1)[:-1]
    view_angles = np.stack(np.meshgrid(azims, elevs), axis=-1).reshape(-1, 2)
    norm = matplotlib.colors.Normalize(np.amin(tags), np.amax(tags))
    color = matplotlib.cm.viridis(norm(tags))
    for i, ax in enumerate(axs.flat):
        ax.scatter(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2], color=color, s=s, alpha=alpha, edgecolor =edgecolor)
        ax.azim = view_angles[i, 0]
        ax.elev = view_angles[i, 1]
        ax.axis("off")
        if azim_elev_title:
            ax.set_title(f"azim={ax.azim}, elev={ax.elev}")
    return fig, axs


def set_size(width=345.0, fraction=1, mode="wide"):
    """Set figure dimensions to avoid scaling in LaTeX.
    Taken from:
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    To get the width of a latex document, print it with:
    \the\textwidth
    (https://tex.stackexchange.com/questions/39383/determine-text-width)
    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    mode: str
            Whether figure should be scaled by the golden ratio in height
            or width
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width  # * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if mode == "wide":
        fig_height_in = fig_width_in * golden_ratio
    elif mode == "tall":
        fig_height_in = fig_width_in / golden_ratio
    elif mode == "square":
        fig_height_in = fig_width_in
    fig_height_max = 550.0 / 72.27
    if mode == "max" or fig_height_in > fig_height_max:
        # standard max-height of latex document
        fig_height_in = fig_height_max
    fig_dim = (fig_width_in, fig_height_in)
    if isinstance(fraction, (int, float)):
        fraction = (fraction, fraction)
    fig_dim = (fig_width_in * fraction[0], fig_height_in * fraction[1])
    return fig_dim


def axis_off_labels_on(ax):
    """Turn of ticks etc, but leaving labels which .axis('off') doesnt"""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

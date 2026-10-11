import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle

import sys

sys.path.append("../")

from methods import filenames, multiimshow


def baround(x, base, *args, **kwargs):
    """
    Round to nearest base. e.g. 
    10 -> 10
    12 -> 10
    13 -> 15

    Taken from:
    https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
    """
    out = base * np.around(x / base, *args, **kwargs)
    out = out.astype(int) if isinstance(out, np.ndarray) else int(out)
    return out


def training_dynamics_default(
    experiment, loss_history, training_metrics, ps=[0.95, 0.5, 0.05]
):
    """
    3 x len(environments)+1 plot matrix:
    first row: total loss history over different enviroments
    second row: prediction error
    third row: L2-regularization

    The three training time sample points:
    1. ps[0] reduction of (minimum - maximum Cuba-Libre metric)
    2. ps[1] reduction of (minimum - maximum Cuba-Libre metric)
    3. ps[2] reduction of (minimum - maximum Cuba-Libre metric)
    """
    # fig, axs = plt.subplots(nrows=3, ncols=len(experiment.environments) + 1)
    fig = plt.figure(figsize=(10, 10))
    history_grid = plt.GridSpec(6, 2, wspace=0.1, hspace=0.25, left=0.1, right=0.5)
    ratemap_grid = plt.GridSpec(
        6,
        2 * len(experiment.environments),
        wspace=-0.2,
        hspace=0.25,
        left=0.5,
        right=1,
    )

    # find training phases as a fraction of loss-differences
    ce = np.array(training_metrics["CE"])
    ce_diff = np.max(ce) - np.min(ce)
    ce_phases = ce_diff * np.array(ps)
    ce_phases += np.min(ce)
    ce_phase_idxs = abs(ce_phases[:, None] - ce[None])  # shape: (3, #epochs)
    ce_phase_idxs = np.argmin(ce_phase_idxs, axis=-1)  # shape: (3,)

    # find closest (rounded) model checkpoints to the phase idxs
    checkpoint_filenames = filenames(experiment.paths["checkpoints"])
    checkpoint_nums = [
        int(checkpoint_filename) for checkpoint_filename in checkpoint_filenames
    ]
    save_freq = checkpoint_nums[2] - checkpoint_nums[1]
    checkpoint_phase_nums = baround(ce_phase_idxs, save_freq)
    checkpoint_phase_nums = np.minimum(checkpoint_phase_nums, np.max(checkpoint_nums))
    checkpoint_phase_filenames = [
        f"{checkpoint_phase_num:05d}.pkl"
        for checkpoint_phase_num in checkpoint_phase_nums
    ]

    # plot loss history
    ax00 = fig.add_subplot(history_grid[0:2, 0:2])
    ax00.plot(loss_history, label="Loss")
    ax00.plot(training_metrics["CE"], label="CE")
    ax00.plot(training_metrics["entropy"], label="Entropy")
    ax00.set_ylabel("Loss History")
    ax00.legend()
    ax00.xaxis.set_visible(False)
    ax00.text(
        1.1,
        0.5,
        f"t={checkpoint_phase_nums[0]}",
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        transform=ax00.transAxes,
    )
    for checkpoint_phase_num in checkpoint_phase_nums:
        ax00.axvline(checkpoint_phase_num, linestyle=":", c="grey")

    # plot L2-penalty
    ax10 = fig.add_subplot(history_grid[2:4, 0:2])
    ax10.plot(training_metrics["l2_reg"])
    ax10.set_ylabel("L2-Penalty")
    ax10.axhline(0, ls=":", c="black")
    ax10.xaxis.set_visible(False)
    ax10.text(
        1.1,
        0.5,
        f"t={checkpoint_phase_nums[1]}",
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        transform=ax10.transAxes,
    )
    for checkpoint_phase_num in checkpoint_phase_nums:
        ax10.axvline(checkpoint_phase_num, linestyle=":", c="grey")

    # plot Euclidean prediction error
    ax20 = fig.add_subplot(history_grid[4:6, 0:2])
    ax20.plot(training_metrics["pred_error"], label="Prediction")
    ax20.plot(training_metrics["true_error"], label="Label")
    ax20.set_ylabel("Decoding Error")
    ax20.legend()
    ax20.text(
        1.1,
        0.5,
        f"t={checkpoint_phase_nums[2]}",
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        transform=ax20.transAxes,
    )
    ax20.set_xlabel("t [Epochs]")
    for checkpoint_phase_num in checkpoint_phase_nums:
        ax20.axvline(checkpoint_phase_num, linestyle=":", c="grey")

    # load grid scores for environments
    sorted_scores = []
    sort_idxs = []
    for grid_score_filename in filenames(experiment.paths["grid_scores"]):
        with open(experiment.paths["grid_scores"] / grid_score_filename, "rb") as f:
            score_map = pickle.load(f)

        sort_idxs.append(np.argsort(score_map)[::-1])
        sorted_scores.append(score_map[sort_idxs[-1]])

    # load ratemaps for environment
    ratemaps = []
    selected_sorted_cells = [2, 15, 2048, 4095]
    for j, checkpoint_phase_filename in enumerate(checkpoint_phase_filenames):
        for env_i in range(len(experiment.environments)):
            ratemap_filename = (
                experiment.paths["ratemaps"]
                / f"env_{env_i}"
                / checkpoint_phase_filename
            )
            with open(ratemap_filename, "rb") as f:
                ratemaps_i = pickle.load(f)

            for ratemap_example_k in range(num_ratemap_examples := 4):
                ratemap_idx = sort_idxs[env_i][selected_sorted_cells[ratemap_example_k]]
                ratemap_score = np.around(
                    sorted_scores[env_i][selected_sorted_cells[ratemap_example_k]],
                    decimals=2,
                )
                ratemap = ratemaps_i[ratemap_idx]

                # select correct 'subplotaxes' and plot ratemap there
                ax = fig.add_subplot(
                    ratemap_grid[
                        2 * j + (ratemap_example_k // 2),
                        2 * env_i + (ratemap_example_k % 2),
                    ]
                )
                ax.imshow(ratemap, cmap="jet")
                ax.axis("off")
                if j == 0:
                    ax.set_title(f"id={ratemap_idx}")
                elif j == len(checkpoint_phase_filenames) - 1:
                    ax.set_title(f"gcs={ratemap_score}")
                ax.set_aspect("equal")

    return fig


def training_dynamics_3ME(
    experiment, loss_history, training_metrics, ps=[0.95, 0.5, 0.05]
):
    """
    3 x len(environments)+1 plot matrix:
    first row: total loss history over different enviroments
    second row: prediction error
    third row: L2-regularization

    The three training time sample points:
    1. ps[0] reduction of (minimum - maximum Cuba-Libre metric)
    2. ps[1] reduction of (minimum - maximum Cuba-Libre metric)
    3. ps[2] reduction of (minimum - maximum Cuba-Libre metric)
    """
    # fig, axs = plt.subplots(nrows=3, ncols=len(experiment.environments) + 1)
    fig = plt.figure(figsize=(14,10))
    history_grid = plt.GridSpec(6, 2, wspace=0.1, hspace=0.25, left=0.1, right=0.23)
    ratemap_grids = []
    for env_rate_i in range(n_rate_plt := 4):
        overlap = 0.12*3 / n_rate_plt
        left_pos = .3 + env_rate_i*(0.9 / n_rate_plt - overlap)
        right_pos = left_pos + 0.9 / n_rate_plt
        ratemap_grids.append(
            plt.GridSpec(6, 4, wspace=0.2, hspace=0.25, left=left_pos, right=right_pos,)
        )

    # find training phases as a fraction of loss-differences
    ce = np.array(training_metrics["CE"])
    ce_diff = np.max(ce) - np.min(ce)
    ce_phases = ce_diff * np.array(ps)
    ce_phases += np.min(ce)
    ce_phase_idxs = abs(ce_phases[:, None] - ce[None])  # shape: (3, #epochs)
    ce_phase_idxs = np.argmin(ce_phase_idxs, axis=-1)  # shape: (3,)

    # find closest (rounded) model checkpoints to the phase idxs
    checkpoint_filenames = filenames(experiment.paths["checkpoints"])
    checkpoint_nums = [
        int(checkpoint_filename) for checkpoint_filename in checkpoint_filenames
    ]
    save_freq = checkpoint_nums[2] - checkpoint_nums[1]
    checkpoint_phase_nums = baround(ce_phase_idxs, save_freq)
    checkpoint_phase_nums = np.minimum(checkpoint_phase_nums, np.max(checkpoint_nums))
    checkpoint_phase_filenames = [
        f"{checkpoint_phase_num:05d}.pkl"
        for checkpoint_phase_num in checkpoint_phase_nums
    ]

    # plot loss history
    ax00 = fig.add_subplot(history_grid[0:2, 0:2])
    ax00.plot(loss_history, label="Loss")
    ax00.plot(training_metrics["CE"], label="CE")
    ax00.plot(training_metrics["entropy"], label="Entropy")
    ax00.set_ylabel("Loss History")
    ax00.legend()
    ax00.xaxis.set_visible(False)
    ax00.text(
        1.2,
        0.5,
        f"t={checkpoint_phase_nums[0]}",
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        transform=ax00.transAxes,
    )
    for checkpoint_phase_num in checkpoint_phase_nums:
        ax00.axvline(checkpoint_phase_num, linestyle=":", c="grey")

    # plot L2-penalty
    ax10 = fig.add_subplot(history_grid[2:4, 0:2])
    ax10.plot(training_metrics["l2_reg"])
    ax10.set_ylabel("L2-Penalty")
    ax10.axhline(0, ls=":", c="black")
    ax10.xaxis.set_visible(False)
    ax10.text(
        1.2,
        0.5,
        f"t={checkpoint_phase_nums[1]}",
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        transform=ax10.transAxes,
    )
    for checkpoint_phase_num in checkpoint_phase_nums:
        ax10.axvline(checkpoint_phase_num, linestyle=":", c="grey")

    # plot Euclidean prediction error
    ax20 = fig.add_subplot(history_grid[4:6, 0:2])
    ax20.plot(training_metrics["pred_error"], label="Prediction")
    ax20.plot(training_metrics["true_error"], label="Label")
    ax20.set_ylabel("Decoding Error")
    ax20.legend()
    ax20.text(
        1.2,
        0.5,
        f"t={checkpoint_phase_nums[2]}",
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        transform=ax20.transAxes,
    )
    ax20.set_xlabel("t [Epochs]")
    for checkpoint_phase_num in checkpoint_phase_nums:
        ax20.axvline(checkpoint_phase_num, linestyle=":", c="grey")

    # load grid scores for environments
    score_maps = []
    sorted_idxs = []
    for grid_score_filename in filenames(experiment.paths["grid_scores"]):
        with open(experiment.paths["grid_scores"] / grid_score_filename, "rb") as f:
            score_map = pickle.load(f)
        sorted_idxs.append(np.argsort(score_map)[::-1])
        score_maps.append(score_map)

    selected_sorted_cells_idxs = []
    for env_i in range(len(experiment.environments)):
        selected_sorted_cells_idxs.append(sorted_idxs[env_i][0])
    selected_sorted_cells_idxs.append(0)

    selected_cell_scores = []
    for env_i in range(len(experiment.environments)):
        selected_cell_scores.append([])
        for selected_sorted_cells_idx in selected_sorted_cells_idxs:
            selected_cell_scores[env_i].append(
                np.around(score_maps[env_i][selected_sorted_cells_idx], decimals=2)
            )

    for j, checkpoint_phase_filename in enumerate(checkpoint_phase_filenames):
        for env_i in range(len(experiment.environments)):
            ratemap_filename = (
                experiment.paths["ratemaps"]
                / f"env_{env_i}"
                / checkpoint_phase_filename
            )
            with open(ratemap_filename, "rb") as f:
                ratemaps_i = pickle.load(f)

            for ratemap_example_k in range(num_ratemap_examples := 4):
                ratemap_idx = selected_sorted_cells_idxs[ratemap_example_k]
                ratemap_score = selected_cell_scores[env_i][ratemap_example_k]
                ratemap = ratemaps_i[ratemap_idx]

                # select correct 'subplotaxes' and plot ratemap there
                ax = fig.add_subplot(
                    ratemap_grids[env_i][
                        2 * j + (ratemap_example_k // 2),
                        (ratemap_example_k % 2),
                    ]
                )
                ax.imshow(ratemap, cmap="jet")
                ax.axis("off")
                if j == 0 and env_i == 0:
                    ax.set_title(f"id={ratemap_idx}")
                elif j == len(checkpoint_phase_filenames) - 1:
                    ax.set_title(f"gcs={ratemap_score}")
                ax.set_aspect("equal")

                if j == 0 and ratemap_example_k == 0:
                    ax.text(1.15, 1.4, f"Env={env_i}", transform=ax.transAxes, horizontalalignment="center", verticalalignment="center")

    return fig 
    
def training_dynamics_3CL(
    experiment, loss_history, training_metrics, *args, **kwargs
):
    """
    3 x len(environments)+1 plot matrix:
    first row: total loss history over different enviroments
    second row: prediction error
    third row: L2-regularization

    The three training time sample points:
    1. ps[0] reduction of (minimum - maximum Cuba-Libre metric)
    2. ps[1] reduction of (minimum - maximum Cuba-Libre metric)
    3. ps[2] reduction of (minimum - maximum Cuba-Libre metric)
    """
    # fig, axs = plt.subplots(nrows=3, ncols=len(experiment.environments) + 1)
    fig = plt.figure(figsize=(14,10))
    history_grid = plt.GridSpec(6, 2, wspace=0.1, hspace=0.25, left=0.1, right=0.23)
    ratemap_grids = []
    ratemap_grids.append(
        plt.GridSpec(6, 4, wspace=0.2, hspace=0.25, left=0.3, right=0.6,)
    )
    ratemap_grids.append(
        plt.GridSpec(6, 4, wspace=0.2, hspace=0.25, left=0.48, right=0.78,)
    )
    ratemap_grids.append(
        plt.GridSpec(6, 4, wspace=0.2, hspace=0.25, left=0.65, right=.95,)
    )
    
    checkpoint_phase_nums = [
        990, 1990, 2990
    ]
    checkpoint_phase_filenames = [
        f"{checkpoint_phase_num:05d}.pkl"
        for checkpoint_phase_num in checkpoint_phase_nums
    ]

    # plot loss history
    ax00 = fig.add_subplot(history_grid[0:2, 0:2])
    ax00.plot(loss_history, label="Loss")
    ax00.plot(training_metrics["CE"], label="CE")
    ax00.plot(training_metrics["entropy"], label="Entropy")
    ax00.set_ylabel("Loss History")
    ax00.legend()
    ax00.xaxis.set_visible(False)
    ax00.text(
        1.2,
        0.5,
        f"t={checkpoint_phase_nums[0]}",
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        transform=ax00.transAxes,
    )
    for checkpoint_phase_num in checkpoint_phase_nums:
        ax00.axvline(checkpoint_phase_num, linestyle=":", c="grey")

    # plot L2-penalty
    ax10 = fig.add_subplot(history_grid[2:4, 0:2])
    ax10.plot(training_metrics["l2_reg"])
    ax10.set_ylabel("L2-Penalty")
    ax10.axhline(0, ls=":", c="black")
    ax10.xaxis.set_visible(False)
    ax10.text(
        1.2,
        0.5,
        f"t={checkpoint_phase_nums[1]}",
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        transform=ax10.transAxes,
    )
    for checkpoint_phase_num in checkpoint_phase_nums:
        ax10.axvline(checkpoint_phase_num, linestyle=":", c="grey")

    # plot Euclidean prediction error
    ax20 = fig.add_subplot(history_grid[4:6, 0:2])
    ax20.plot(training_metrics["pred_error"], label="Prediction")
    ax20.plot(training_metrics["true_error"], label="Label")
    ax20.set_ylabel("Decoding Error")
    ax20.legend()
    ax20.text(
        1.2,
        0.5,
        f"t={checkpoint_phase_nums[2]}",
        horizontalalignment="center",
        verticalalignment="center",
        rotation=90,
        transform=ax20.transAxes,
    )
    ax20.set_xlabel("t [Epochs]")
    for checkpoint_phase_num in checkpoint_phase_nums:
        ax20.axvline(checkpoint_phase_num, linestyle=":", c="grey")

    # load grid scores for environments
    score_maps = []
    sorted_idxs = []
    for grid_score_filename in filenames(experiment.paths["grid_scores"]):
        with open(experiment.paths["grid_scores"] / grid_score_filename, "rb") as f:
            score_map = pickle.load(f)
        sorted_idxs.append(np.argsort(score_map)[::-1])
        score_maps.append(score_map)

    selected_sorted_cells_idxs = []
    for env_i in range(len(experiment.environments)):
        selected_sorted_cells_idxs.append(sorted_idxs[env_i][0])
    selected_sorted_cells_idxs.append(0)

    selected_cell_scores = []
    for env_i in range(len(experiment.environments)):
        selected_cell_scores.append([])
        for selected_sorted_cells_idx in selected_sorted_cells_idxs:
            selected_cell_scores[env_i].append(
                np.around(score_maps[env_i][selected_sorted_cells_idx], decimals=2)
            )

    for j, checkpoint_phase_filename in enumerate(checkpoint_phase_filenames):
        for env_i in range(len(experiment.environments)):
            ratemap_filename = (
                experiment.paths["ratemaps"]
                / f"env_{env_i}"
                / checkpoint_phase_filename
            )
            with open(ratemap_filename, "rb") as f:
                ratemaps_i = pickle.load(f)

            for ratemap_example_k in range(num_ratemap_examples := 4):
                ratemap_idx = selected_sorted_cells_idxs[ratemap_example_k]
                ratemap_score = selected_cell_scores[env_i][ratemap_example_k]
                ratemap = ratemaps_i[ratemap_idx]

                # select correct 'subplotaxes' and plot ratemap there
                ax = fig.add_subplot(
                    ratemap_grids[env_i][
                        2 * j + (ratemap_example_k // 2),
                        (ratemap_example_k % 2),
                    ]
                )
                ax.imshow(ratemap, cmap="jet")
                ax.axis("off")
                if j == 0 and env_i == (len(experiment.environments)-1):
                    ax.set_title(f"id={ratemap_idx}")
                if j == env_i:
                    ax.set_title(f"gcs={ratemap_score}")
                ax.set_aspect("equal")

                if j == 0 and ratemap_example_k == 0:
                    ax.text(1.15, 1.4, f"Env={env_i}", transform=ax.transAxes, horizontalalignment="center", verticalalignment="center")

    return fig

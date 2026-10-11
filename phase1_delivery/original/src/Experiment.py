import os
from pathlib import Path
import datetime

import numpy as np
import torch
import pickle

import ratsimulator
from PlaceCells import PlaceCells


class Experiment:
    def __init__(self, name, base_path=None):
        # custom unique experiment name/tag identifier
        self.name = name

        self._set_pathing(base_path)
        # if experiment folder already exists - experiment should be loaded, otherwise created
        self.is_new_experiment = not os.path.exists(self.paths["experiment"])
        if self.is_new_experiment:
            print(
                f"Experiment <{self.name}> is NEW. Loading DEFAULT experiment settings!"
            )
            self._init_default_experiment()
            print(
                "Default <params>, <environments>, <agents> and <pc_ensembles> can now be changed. Finish setup by calling setup()"
            )
        else:
            print(
                f"Experiment <{self.name}> already EXISTS. Loading experiment settings!"
            )
            self._load_experiment()

    def _set_pathing(self, base_path):
        self.paths = {}
        # set base paths
        self.paths["data"] = (
            Path(base_path) / "data" if base_path else Path().home() / "data"
        )
        self.paths["project"] = self.paths["data"] / "emergent-grid-cells"
        if not os.path.exists(self.paths["project"]):
            os.makedirs(self.paths["project"])
        self.paths["experiment"] = self.paths["project"] / self.name

        # set specific paths (results etc)
        self.paths["checkpoints"] = self.paths["experiment"] / "checkpoints"
        self.paths["ratemaps"] = self.paths["experiment"] / "ratemaps"
        self.paths["dynamics"] = self.paths["experiment"] / "dynamics"
        self.paths["grid_scores"] = self.paths["experiment"] / "grid_scores"

    def _init_default_experiment(self):
        # --- init default "global" params ---
        self.params = {}
        # model parameters
        self.params["Ng"] = 4096
        self.params["Np"] = 512
        # training parameters
        self.params["sampler"] = "MESampler"
        self.params["weight_decay"] = 1e-4
        self.params["lr"] = 1e-4
        self.params["seq_len"] = 20
        self.params["batch_size"] = 200
        self.params["nsteps"] = 100
        self.params["nepochs"] = 1000
        # metadata
        self.params["save_freq"] = 10
        self.params["date"] = datetime.datetime.now()

        # init default environments
        self.environments = [
            ratsimulator.Environment.Rectangle(boxsize=(2.2, 2.2), soft_boundary=0.03)
        ]

        # init default agents
        self.agents = [
            ratsimulator.Agent(
                environment=self.environments[0],
                angle0=None,
                p0=None,
                dt=0.02,
                turn_angle=5.76 * 2,
                b=0.13 * 2 * np.pi,
                mu=0,
                boundary_mode="sorschers",
            )
        ]

        # init default place cell ensembles
        self.pc_ensembles = [
            PlaceCells(
                environment=self.environments[0],
                npcs=self.params["Np"],
                pc_width=0.12,
                DoG=True,
                surround_scale=2,
                p=2.0,
                seed=0,
            )
        ]
        # overload place cell layout with sorscher's sampled pc layout
        self.pc_ensembles[0].pcs = torch.tensor(
            np.load(self.paths["project"] / "example_pc_centers.npy") + 1.1,
            dtype=torch.float32,
        )
        self.pc_ensembles[0].seed = "sorsher loaded place cells"

    @staticmethod
    def get_default_ecology(seed):
        # init default environments
        environments = [
            ratsimulator.Environment.Rectangle(boxsize=(2.2, 2.2), soft_boundary=0.03)
        ]

        # init default agents
        agents = [
            ratsimulator.Agent(
                environment=environments[0],
                angle0=None,
                p0=None,
                dt=0.02,
                turn_angle=5.76 * 2,
                b=0.13 * 2 * np.pi,
                mu=0,
                boundary_mode="sorschers",
            )
        ]

        # init default place cell ensembles
        pc_ensembles = [
            PlaceCells(
                environment=environments[0],
                npcs=512,
                pc_width=0.12,
                DoG=True,
                surround_scale=2,
                p=2.0,
                seed=seed,
            )
        ]
        return environments, agents, pc_ensembles

    def _load_experiment(self):
        print("Loading experiment details")
        with open(self.paths["experiment"] / "params.pkl", "rb") as f:
            self.params = pickle.load(f)
        with open(self.paths["experiment"] / "environments.pkl", "rb") as f:
            self.environments = pickle.load(f)
        with open(self.paths["experiment"] / "agents.pkl", "rb") as f:
            self.agents = pickle.load(f)
        with open(self.paths["experiment"] / "pc_ensembles.pkl", "rb") as f:
            self.pc_ensembles = pickle.load(f)

    def setup(self):
        if not self.is_new_experiment:
            print("This experiment has ALREADY been setup - SKIPPING.")
            return False

        print("Creating directories")
        for path in self.paths.values():
            if not os.path.exists(path):
                os.makedirs(path)
        for env_i in range(len(self.environments)):
            os.makedirs(self.paths["dynamics"] / f"env_{env_i}")
            os.makedirs(self.paths["ratemaps"] / f"env_{env_i}")

        print("Saving experiment details")
        with open(self.paths["experiment"] / "params.pkl", "wb") as f:
            pickle.dump(self.params, f)
        with open(self.paths["experiment"] / "environments.pkl", "wb") as f:
            pickle.dump(self.environments, f)
        with open(self.paths["experiment"] / "agents.pkl", "wb") as f:
            pickle.dump(self.agents, f)
        with open(self.paths["experiment"] / "pc_ensembles.pkl", "wb") as f:
            pickle.dump(self.pc_ensembles, f)
        return True

    def __sub__(self, other):
        """Returns the (LEFT-hand) difference between two experiments"""
        if not isinstance(other, type(self)):
            raise RuntimeError(
                "Subtraction is only defined between two Experiment() objects"
            )
        difference = None
        return difference

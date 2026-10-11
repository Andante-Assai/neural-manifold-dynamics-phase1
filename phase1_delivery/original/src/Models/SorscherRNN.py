import torch
import numpy as np
import tqdm
import pickle

import copy
import sys

sys.path.append("../") if "../" not in sys.path else None

from Logger import *


class SorscherRNN(torch.nn.Module):
    """
    Model based on:
    https://github.com/ganguli-lab/grid-pattern-formation/blob/master/model.py
    """

    def __init__(
        self, place_cell_ensembles, Ng=4096, Np=512, nonlinearity="relu", **kwargs
    ):
        super(SorscherRNN, self).__init__(**kwargs)
        self.place_cell_ensembles = place_cell_ensembles
        self.Ng, self.Np = Ng, Np

        # define network architecture
        self.init_position_encoder = torch.nn.Linear(Np, Ng, bias=False)
        self.RNN = torch.nn.RNN(
            input_size=2,
            hidden_size=Ng,
            num_layers=1,
            nonlinearity=nonlinearity,
            bias=False,
            batch_first=True,
        )
        # Linear read-out weights
        self.decoder = torch.nn.Linear(Ng, Np, bias=False)

        # initialise model weights
        for param in self.parameters():
            torch.nn.init.xavier_uniform_(param.data, gain=1.0)

        # evaluate model on random place cell basis
        self.novel_room = copy.deepcopy(place_cell_ensembles[0])
        self.novel_room.global_remap(23031994)

        # pruning
        self.original_weights = []
        self._rnnh_prune_mask = torch.ones((Ng, Ng), device=self.device)
        self._rnni_prune_mask = torch.ones((Ng, 2), device=self.device)
        self._prune_mask_idxs = []

    def to(self, device=None, *args, **kwargs):
        """Overwrites: To also add place_cells on same device"""
        # copy place cells before putting on gpu avoid conflict with dataloader
        self.place_cell_ensembles = [
            copy.copy(place_cells) for place_cells in self.place_cell_ensembles
        ]
        if device is not None:
            for i, place_cells in enumerate(self.place_cell_ensembles):
                self.place_cell_ensembles[i].pcs = self.place_cell_ensembles[i].pcs.to(
                    device
                )
        self.novel_room.pcs = self.novel_room.pcs.to(device)
        return super(SorscherRNN, self).to(device, *args, **kwargs)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def prune_mask(self):
        return [self._rnni_prune_mask, self._rnnh_prune_mask]

    @property
    def prune_mask_idxs(self):
        return self._prune_mask_idxs

    @prune_mask.setter
    def prune_mask(self, idxs):
        """
        Set outgoing weights corresponding to output nodes at
        idxs to zero in recurrent matrix - I.e. prune grid cells.
        """
        if not self.original_weights:
            # save weights
            self.original_weights = [
                self.RNN.weight_ih_l0.clone(),
                self.RNN.weight_hh_l0.clone(),
            ]

        # reset weights
        self.RNN.weight_ih_l0, self.RNN.weight_hh_l0 = map(
            lambda x: torch.nn.parameter.Parameter(x.clone()), self.original_weights
        )
        self._rnni_prune_mask, self._rnnh_prune_mask = (
            torch.ones((self.Ng, 2), device=self.device),
            torch.ones((self.Ng, self.Ng), device=self.device),
        )

        maski = torch.ones((self.Ng, 2), device=self.device)
        maskh = torch.ones((self.Ng, self.Ng), device=self.device)
        maski[idxs] = torch.zeros(2, device=self.device)
        maskh[idxs] = torch.zeros(self.Ng, device=self.device)
        self.RNN.weight_ih_l0 = torch.nn.parameter.Parameter(
            self.RNN.weight_ih_l0 * maski
        )
        self.RNN.weight_hh_l0 = torch.nn.parameter.Parameter(
            self.RNN.weight_hh_l0 * maskh
        )
        self._rnni_prune_mask = maski
        self._rnnh_prune_mask = maskh
        self._prune_mask_idxs = idxs

    def g(self, inputs):
        v, p0 = inputs
        if len(v.shape) == 2 and len(p0.shape) == 1:
            # model requires tensor of degree 3 (B,S,N)
            # assume inputs missing empty batch-dim
            v, p0 = v[None], p0[None]

        if self.device != v.device:
            # put v and p0 on same device as model
            v = v.to(self.device, dtype=self.dtype)
            p0 = p0.to(self.device, dtype=self.dtype)

        p0 = self.init_position_encoder(p0)
        p0 = p0[None]  # add dummy (unit) dim for number of stacked rnns (D)
        # output of torch.RNN is a 2d-tuple. First element =>
        # return_sequences=True (in tensorflow). Last element => False.
        out, _ = self.RNN(v, p0)
        return out

    def p(self, g_inputs, log_softmax=False):
        place_preds = self.decoder(g_inputs)
        return (
            torch.nn.functional.log_softmax(place_preds, dim=-1)
            if log_softmax
            else place_preds
        )

    def forward(self, inputs, log_softmax=False):
        gs = self.g(inputs)
        return self.p(gs, log_softmax)


    def l2_reg(self, weight_decay):
        return weight_decay * torch.sum(self.RNN.weight_hh_l0**2)

    def loss_fn(self, log_predictions, labels, weight_decay):
        """
        Args:
            inputs ((B, S, 2), (B, Np)): velocity and position in pc-basis
            labels (B, Np): ground truth pc population activations
        """
        # Actual cross entropy between two distributions p(x) and q(x),
        # rather than classic CE implementations assuming one-hot p(x).
        if labels.device != self.device:
            labels = labels.to(self.device, dtype=self.dtype)
        return CE(log_predictions, labels) + self.l2_reg(weight_decay)

    def position_error(self, pc_pos, cartesian_pos, indices, place_cell_ensembles):
        batch_size = pc_pos.shape[0]
        err = 0
        for env_i, place_cells in enumerate(place_cell_ensembles):
            mask_i = np.array(indices) == env_i
            decoded_pos = place_cells.to_euclid(pc_pos[mask_i])
            # sum over patchy-batch dim (mean after loop), mean over seq_len dim,
            # "sum" (euclidean) over spatial dim
            err += torch.sum(
                torch.mean(
                    torch.sqrt(
                        torch.sum(
                            (decoded_pos - cartesian_pos[mask_i, 1:]) ** 2, axis=-1
                        )
                    ),
                    axis=-1,
                )
            )
        err /= batch_size
        return err

    def train(
        self,
        trainloader,
        optimizer,
        weight_decay,
        nepochs,
        save_freq,
        paths,
        logger=None,
    ):
        logger = Logger(len(trainloader)) if not logger else logger
        start_epoch = len(logger.loss_history["familiar"])

        # save_iter = 0
        pbar = tqdm.tqdm(range(start_epoch, nepochs))
        for epoch in pbar:
            # generic torch training loop
            logger.new_epoch()
            for inputs, labels, positions, indices in trainloader:
                indices = np.array(indices)
                # zero the parameter gradients
                optimizer.zero_grad(set_to_none = True)
                # forward + backward + optimize
                log_predictions = self(inputs, log_softmax=True)
                loss = self.loss_fn(log_predictions, labels, weight_decay)
                loss.backward()
                optimizer.step()

                # update training metrics
                labels = labels.to(self.device, dtype=self.dtype)
                positions = positions.to(self.device, dtype=self.dtype)
                pred_error = self.position_error(
                    log_predictions, positions, indices, self.place_cell_ensembles
                )
                true_error = self.position_error(
                    labels, positions, indices, self.place_cell_ensembles
                )
                logger.update(
                    "familiar",
                    loss,
                    log_predictions,
                    labels,
                    self.l2_reg(weight_decay),
                    pred_error,
                    true_error,
                )

                # measure model in novel environment (global remapping)
                with torch.no_grad():
                    labels = self.novel_room.softmax_response(positions)
                    labels = labels[:, 1:]  # labels are without init pos
                    inputs[1] = labels[:, 0]  # change init pos basis to novel pc basis

                    log_predictions = self(inputs, log_softmax=True)
                    loss = self.loss_fn(log_predictions, labels, weight_decay)
                    indices = torch.zeros(labels.shape[0]) # all mini-batch indices are novel

                    # update training metrics
                    pred_error = self.position_error(
                        log_predictions,
                        positions,
                        indices,
                        [self.novel_room],
                    )
                    true_error = self.position_error(
                        labels, positions, indices, [self.novel_room]
                    )
                    logger.update(
                        "novel",
                        loss,
                        log_predictions,
                        labels,
                        self.l2_reg(weight_decay),
                        pred_error,
                        true_error,
                    )

            # save model history
            if not (epoch % save_freq) or epoch == (nepochs - 1):
                save_dict = {}
                save_dict["optimizer_state_dict"] = optimizer.state_dict()
                save_dict["model_state_dict"] = self.state_dict()
                torch.save(save_dict, paths["checkpoints"] / f"{epoch:05d}")
                with open(paths["experiment"] / "logger.pkl", "wb") as f:
                    pickle.dump(logger, f)

            # update tqdm training-bar description
            pbar.set_description(
                    f"Epoch={epoch}/{nepochs}, loss(F,N)={logger.loss_history['familiar'][-1]:.4f}, {logger.loss_history['novel'][-1]:.4f}, error(P,T)={logger.training_metrics['familiar']['pred_error'][-1]:.4f},{logger.training_metrics['familiar']['true_error'][-1]:.4f}"
            )
        return logger

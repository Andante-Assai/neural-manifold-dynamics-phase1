import torch


class PlaceCells:
    def __init__(
        self,
        environment,
        npcs,
        pc_width=0.12,
        DoG=False,
        surround_scale=2,
        p=2.0,
        seed=0,
        dtype=torch.float32,
        **kwargs
    ):
        """
        Args:
                environment: ABCEnvironment class instance
                npcs: Number of place cells
                pc_width: Array-like or scalar. Tuning curve of place cells
        """
        self.environment = environment
        self.pcs = environment.sample_uniform(npcs, seed=seed)  # sample place cell
        self.pcs = torch.tensor(self.pcs, dtype=dtype)
        self.npcs = npcs
        self.pc_width = pc_width
        self.DoG = DoG
        self.surround_scale = surround_scale
        self.p = p
        self.seed = seed

    def global_remap(self, seed, **kwargs):
        self.pcs = self.environment.sample_uniform(self.npcs, seed=seed)
        self.pcs = torch.tensor(self.pcs, **kwargs)
        self.seed = seed

    def softmax_response(self, pos):
        """Place cell response as modelled by Sorscher"""
        if len(pos.shape) == 1:
            pos = pos[None]
            out_shape = self.npcs
        elif len(pos.shape) == 2:
            out_shape = [pos.shape[0], self.npcs]
        elif len(pos.shape) == 3:
            out_shape = list(pos.shape[:-1]) + [self.npcs]
            pos = pos.reshape(pos.shape[0] * pos.shape[1], pos.shape[-1])

        # distance to place cell center
        dists = torch.sum((pos[:, None] - self.pcs[None]) ** self.p, axis=-1)
        activity = torch.nn.functional.softmax(
            -dists / (2 * self.pc_width ** 2), dim=-1
        )

        if self.DoG:
            activity -= torch.nn.functional.softmax(
                -dists / (2 * self.surround_scale * self.pc_width ** 2), dim=-1
            )

            # after DoG, activity is not a probability dist anymore
            # shift and rescale s.t it becomes a prob dist again.
            activity -= torch.min(
                activity, dim=-1, keepdim=True
            ).values  # returns idxs and values
            activity /= torch.sum(activity, dim=-1, keepdim=True)

        return activity.reshape(out_shape)

    def to_euclid(self, activity, k=3):
        """
        Decode place-cell activity to Euclidean coordinates - following Sorscher.
        OBS! This is an approximation to the actual Euclidean location,
        by considering the top k place-cell activities as if the agent is located
        at the average k place-cell center location
        """
        _, idxs = torch.topk(activity, k, dim=-1)
        return torch.mean(self.pcs[idxs], axis=-2)

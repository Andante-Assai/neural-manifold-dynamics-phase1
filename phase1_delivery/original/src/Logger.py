import torch


def CE(log_predictions, labels):
    """
    Cross entropy. Method assumes predictions are already log()-transformed.
    """
    return torch.mean(-torch.sum(labels * log_predictions, axis=-1))


def entropy(labels):
    """
    Information entropy
    """
    return torch.mean(-torch.sum(torch.nan_to_num(labels * torch.log(labels)), axis=-1))


def KL(cross_entropy_value, entropy_value):
    """
    The Kullback Leibler Divergence
    """
    return cross_entropy_value - entropy_value


class Logger:
    def __init__(self, num_batches_in_epoch):
        self.num_batches_in_epoch = num_batches_in_epoch
        self.training_metrics = {"novel": {}, "familiar": {}}
        self.loss_history = {"novel": [], "familiar": []}

        for key in self.loss_history:
            self.training_metrics[key]["CE"] = []
            self.training_metrics[key]["entropy"] = []
            self.training_metrics[key]["KL"] = []
            self.training_metrics[key]["l2_reg"] = []
            self.training_metrics[key]["pred_error"] = []
            self.training_metrics[key]["true_error"] = []

    def new_epoch(self):
        for key in self.training_metrics["familiar"]:
            self.training_metrics["familiar"][key].append(0)
            self.training_metrics["novel"][key].append(0)

        self.loss_history["familiar"].append(0)
        self.loss_history["novel"].append(0)

    def update(
        self, key, loss, log_predictions, labels, l2_reg, pred_error, true_error
    ):
        cross_entropy_value = CE(log_predictions, labels).item()
        entropy_value = entropy(labels).item()

        self.training_metrics[key]["CE"][-1] += (
            cross_entropy_value / self.num_batches_in_epoch
        )
        self.training_metrics[key]["entropy"][-1] += (
            entropy_value / self.num_batches_in_epoch
        )

        self.training_metrics[key]["KL"][-1] += (
            KL(cross_entropy_value, entropy_value) / self.num_batches_in_epoch
        )
        self.training_metrics[key]["l2_reg"][-1] += l2_reg.item() / self.num_batches_in_epoch
        self.training_metrics[key]["pred_error"][-1] += (
            pred_error.item() / self.num_batches_in_epoch
        )
        self.training_metrics[key]["true_error"][-1] += (
            true_error.item() / self.num_batches_in_epoch
        )

        self.loss_history[key][-1] += loss.item() / self.num_batches_in_epoch







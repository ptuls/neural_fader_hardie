import torch
import torch.functional as F
import torch.nn as nn


class HazardLoss(nn.Module):
    """
    The hazard function here is the log-likelihood of a geometric distribution. Loss takes in
    the probability of churn and outputs the loss. The aim is to get the model to learn the
    probability of churn given a set of features and periods of survivai.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        # additional epsilon to ensure stability of log function
        if eps <= 0.0:
            raise ValueError("eps value must be > 0")

        if eps >= 0.01:
            raise ValueError("recommended eps value < 0.01")
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets are (periods, is_retained)
        # geometric distribution log-likelihood: -t log(1-p) - log(p)
        target_size = targets.size()

        if len(target_size) != 3:
            raise ValueError("target tensor must have 3 dimensions")

        if target_size[2] != 2:
            raise ValueError("labels must have period and churn status")

        bounded_inputs = torch.clamp(inputs, min=0.0, max=1.0)
        log_loss = -(targets[:, :, 0] - targets[:, :, 1]) * torch.log(
            1 - bounded_inputs + self.eps
        ) - targets[:, :, 1] * torch.log(bounded_inputs + self.eps)
        loss = log_loss.mean()

        return loss

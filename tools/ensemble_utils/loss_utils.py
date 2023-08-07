import torch
import torch.nn as nn
import numpy as np

class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta    # 0.11111
        self.weight = code_weights
        if self.weight is not None:
            self.weight = np.array(self.weight, dtype=np.float32)
            self.weight = torch.from_numpy(self.weight).cuda()  # cuda: [1,1,1,1,1,1,1]

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Param:
            input:  (k, 7)
            target: (k, 7)
        Return:
            loss: tensor()
        """
        assert input.shape == target.shape and input.shape > 0
        # target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)    # (k, 7)

        if self.weight is not None:
            assert self.weight.shape[0] == loss.shape[-1]
            loss = loss * self.weight[None, :]

        k = input.shape[0]
        loss = loss.sum() / k   # norm
        return loss


class WeightedMSELoss(nn.Module):
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        super(WeightedMSELoss, self).__init__()
        self.beta = beta    # 0.11111
        self.weight = code_weights
        if self.weight is not None:
            self.weight = np.array(self.weight, dtype=np.float32)
            self.weight = torch.from_numpy(self.weight).cuda()     # cuda: [1,1,1,1,1,1,1]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Param:
            input:  (k, 7)
            target: (k, 7)
        Return:
            loss: tensor()
        """
        assert input.shape == target.shape and input.shape > 0
        loss = (input - target) ** 2

        if self.weight is not None:
            assert self.weight.shape[0] == loss.shape[-1]
            loss = loss * self.weight[None, :]

        k = input.shape[0]
        loss = loss.sum() / k  # norm
        return loss

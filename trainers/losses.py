import torch
from torch import nn


class MaskedL1Loss(nn.Module):
    def __init__(self, threshold=100):
        self.threshold = threshold
        self.e = 1e-10
        super().__init__()

    def forward(self, prediction, target):
        gt = target.clone()
        prediction = prediction.squeeze(dim=1)
        valid_map = gt > 0
        gt[gt > self.threshold] = self.threshold
        gt /= self.threshold
        error = torch.abs(gt[valid_map] - prediction[valid_map]) / torch.sum(valid_map)
        return torch.sum(error)

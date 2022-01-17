import torch
import torch.nn.functional as F
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


class WeightedCrossEntropy(nn.Module):
    def __init__(self, ignore_index, num_classes):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.max_value = 7
        self.class_weight = torch.FloatTensor(self.num_classes).zero_().cuda() + 1
        self.often_weight = torch.FloatTensor(self.num_classes).zero_().cuda() + 1

    def forward(self, prediction, target):
        weight = torch.FloatTensor(self.num_classes).zero_().cuda()
        weight += 1
        count = torch.FloatTensor(self.num_classes).zero_().cuda()
        often = torch.FloatTensor(self.num_classes).zero_().cuda()
        often += 1
        n, h, w = target.shape
        for i in range(self.num_classes):
            count[i] = torch.sum(target == i)
            if count[i] < 64 * 64 * n:  # small objective
                weight[i] = self.max_value

        often[count == 0] = self.max_value

        self.often_weight = 0.9 * self.often_weight + 0.1 * often
        self.class_weight = weight * self.often_weight
        return F.cross_entropy(
            prediction,
            target,
            weight=self.class_weight,
            ignore_index=self.ignore_index,
        )

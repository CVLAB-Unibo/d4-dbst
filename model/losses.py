import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCrossEntropy(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(self, prediction, target, depth):
        # print(depth.size())
        loss = self.loss_fn(prediction, target)
        weights = torch.ones_like(depth)
        weights[depth>depth.mean()] = 10
        # print(weights.size())

        loss = torch.mean(loss * weights.squeeze())
        return loss

class Masked_L1_loss(nn.Module):
    def __init__(self, threshold=100):
        self.threshold = threshold
        self.e = 1e-10
        super().__init__()
        
    def forward(self, prediction, target):
        gt = target.clone()
        prediction = prediction.squeeze(dim=1)
        valid_map = gt>0
        gt[gt>self.threshold] = self.threshold
        gt /= self.threshold
        error = torch.abs(gt[valid_map]-prediction[valid_map])/torch.sum(valid_map)
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
            count[i] = torch.sum(target==i)
            if count[i] < 64*64*n: #small objective
                weight[i] = self.max_value
        
        often[count == 0] = self.max_value

        self.often_weight = 0.9 * self.often_weight + 0.1 * often 
        self.class_weight = weight * self.often_weight
        return F.cross_entropy( prediction, target, weight=self.class_weight, ignore_index=self.ignore_index)

def get_loss_fn(params):

    if params.loss_fn=='crossentropy':
        return nn.CrossEntropyLoss(ignore_index=params.ignore_index)
    if params.loss_fn=='weighted_crossentropy':
        return WeightedCrossEntropy(ignore_index=params.ignore_index, num_classes=params.num_classes)        
    # elif params.loss_name=='beruh':
    #     return BerHu(**kwargs)
    elif params.loss_fn=='l1':
        return Masked_L1_loss(threshold=params.threshold)
    elif params.loss_fn=='l2':
        return nn.MSELoss()  
    elif params.loss_fn=='masked_crossentropy':
        return MaskedCrossEntropy(ignore_index=params.ignore_index)        
    else:
        return nn.CrossEntropyLoss(ignore_index=params.ignore_index)
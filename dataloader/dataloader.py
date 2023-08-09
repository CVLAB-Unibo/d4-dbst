import torch
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from . import transform

import numpy as np
from dataloader.dataloader_depth import DepthDataset
from dataloader.dataloader_semantic import SegmentationDataset
from dataloader.dataloader_depth_semantic import SemanticDepth


def fetch_dataloader(root, txt_file, split, params, sem_depth=False, use_data_augmentation=False):
    # these can be changed. By deafult we use the target dataset statistics (i.e. Cityscapes)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.225, 0.224]
   
    if use_data_augmentation:
        transform_train = [
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std, to_bgr255=False)
            ]
        transform_train = [transform.RandomHorizontalFlip(p=0.5)] + transform_train
        transform_train = [
                    transform.ColorJitter(
                        brightness=0.5,
                        contrast=0.5,
                        saturation=0.5,
                        hue=0.5,
                    ),
                ] + transform_train
        transform_train = transform.Compose(transform_train)
    else:
        transform_train = None

    transform_val = None

    if split == 'train':
        if sem_depth == False:
            if params.task == 'depth':
                dataset = DepthDataset(root, txt_file, transforms=transform_train,
                                       max_depth=params.max_depth, threshold=params.threshold, mean=mean, std=std, use_depth=params.use_depth, size=params.load_size, label_size=params.load_size)
            elif params.task == 'segmentation':
                dataset = SegmentationDataset(
                    root, txt_file, transforms=transform_train, encoding=params.encoding, mean=mean, std=std, size=params.load_size, label_size=params.load_size)
        else:
            dataset = SemanticDepth(root, txt_file, transforms=transform_train,
                                       max_depth=params.max_depth, threshold=params.threshold, mean=mean, std=std, use_depth=params.use_depth, size=params.load_size, label_size=params.load_size)
        return DataLoader(dataset, batch_size=params.batch_size_train, shuffle=True, num_workers=params.num_workers, pin_memory=True, drop_last=True)

    elif split == 'val':
        if sem_depth == False:
            if params.task == 'depth':
                dataset = DepthDataset(root, txt_file, transforms=transform_val,
                                       max_depth=params.max_depth, threshold=params.threshold, mean=mean, std=std, use_depth=params.use_depth, size=params.load_size, label_size=params.label_size)
            elif params.task == 'segmentation':
                dataset = SegmentationDataset(
                    root, txt_file, transforms=transform_val, encoding=params.encoding, mean=mean, std=std, size=params.load_size, label_size=params.label_size, val=True)
        else:
            dataset = SemanticDepth(root, txt_file, transforms=transform_val,
                                       max_depth=params.max_depth, threshold=params.threshold, mean=mean, std=std, use_depth=params.use_depth, size=params.load_size, label_size=params.label_size)

        # reduce validation data to speed up training
        if "split_validation" in params.dict:
            ss = ShuffleSplit(
                n_splits=1, test_size=params.split_validation, random_state=42)
            indexes = range(len(dataset))
            split1, split2 = next(ss.split(indexes))
            dataset = Subset(dataset, split2)

        return DataLoader(dataset, batch_size=params.batch_size_val, shuffle=False, num_workers=params.num_workers, pin_memory=True)

import torch
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from . import transform
import numpy as np
from dataloader.dataloader_depth import DepthDataset
from dataloader.dataloader_semantic_DBST import SegmentationDataset
from dataloader.dataloader_depth_semantic import SemanticDepth


def fetch_dataloader(root, train_augmented_dir, txt_file, split, params, sem_depth=False):
    # these can be changed. By deafult we use the target dataset statistics (i.e. Cityscapes)
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.225, 0.224]
    mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    std = np.array((1, 1, 1), dtype=np.float32) # 

    transform_train = [
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std, to_bgr255=True)
        ]
    transform_train = [transform.RandomHorizontalFlip(p=0.5)] + transform_train
    transform_train = [
                transform.RandomScale(scale=[0.5, 1.5]),
                transform.RandomCrop(size=(params.load_size[1], params.load_size[0]), pad_if_needed=True, label_fill=params.ignore_index),
            ] + transform_train
    transform_train = [
                transform.ColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.5,
                ),
            ] + transform_train
    transform_train = transform.Compose(transform_train)        
    transform_val = None

    if split == 'train':
        dataset = SegmentationDataset(
                    root, train_augmented_dir, txt_file, transforms=transform_train, encoding=params.encoding, mean=mean, std=std, size=params.load_size, label_size=params.load_size)
        return DataLoader(dataset, batch_size=params.batch_size_train, shuffle=True, num_workers=params.num_workers, pin_memory=True)

    elif split == 'val':
        dataset = SegmentationDataset(
                    root, train_augmented_dir, txt_file, transforms=transform_val, encoding=params.encoding, mean=mean, std=std, size=params.load_size, label_size=params.label_size, val=True)

        # reduce validation data to speed up training
        if "split_validation" in params.dict:
            ss = ShuffleSplit(
                n_splits=1, test_size=params.split_validation, random_state=42)
            indexes = range(len(dataset))
            split1, split2 = next(ss.split(indexes))
            dataset = Subset(dataset, split2)

        return DataLoader(dataset, batch_size=params.batch_size_val, shuffle=False, num_workers=params.num_workers, pin_memory=True)

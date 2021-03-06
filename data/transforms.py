from typing import Callable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode

T_SAMPLE = Tuple[Image.Image, Optional[np.ndarray], Optional[np.ndarray]]
T_ITEM = Tuple[Tensor, Optional[Tensor], Optional[Tensor]]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.225, 0.224]


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, sample: T_SAMPLE) -> T_ITEM:
        item = sample
        for t in self.transforms:
            item = t(item)
        return cast(T_ITEM, item)


class RandomHorizontalFlip:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, item: T_ITEM) -> T_ITEM:
        img, sem, dep = item

        if torch.rand(1) < self.p:
            img = F.hflip(img)
            if sem is not None:
                sem = F.hflip(sem)
            if dep is not None:
                dep = F.hflip(dep)

        return img, sem, dep

class RandomScale(object):
    def __init__(
        self,
        scale_range: Tuple[float, float],
        img_interp_mode: InterpolationMode = InterpolationMode.BICUBIC,
        sem_interp_mode: InterpolationMode = InterpolationMode.NEAREST,
        dep_interp_mode: InterpolationMode = InterpolationMode.BILINEAR,
    ) -> None:
        self.scale_range = scale_range
        self.img_interp_mode = img_interp_mode
        self.sem_interp_mode = sem_interp_mode
        self.dep_interp_mode = dep_interp_mode

    def __call__(self, item: T_ITEM) -> T_ITEM:
        img, sem, dep = item

        h, w = img.shape[1], img.shape[2]
        random = torch.rand(1).item()
        random_scale = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * random
        new_size = (int(h * random_scale), int(w * random_scale))

        resized_img = F.resize(img, list(new_size), self.img_interp_mode)
        resized_sem, resized_dep = None, None

        if sem is not None:
            sem.unsqueeze_(0)
            resized_sem = F.resize(sem, list(new_size), self.sem_interp_mode)
            resized_sem.squeeze_()
        if dep is not None:
            dep.unsqueeze_(0)
            resized_dep = F.resize(dep, list(new_size), self.dep_interp_mode)
            resized_dep.squeeze_()

        return resized_img, resized_sem, resized_dep


class RandomCrop:
    def __init__(self, size: Tuple[int, int]):
        self.size = size

    @staticmethod
    def get_params(
        in_size: Tuple[int, int],
        out_size: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        ih, iw = in_size
        oh, ow = out_size

        if in_size <= out_size:
            return 0, 0, ih, iw

        i = int(torch.randint(ih - oh, (1,)).item())
        j = int(torch.randint(iw - ow, (1,)).item())
        return i, j, oh, ow

    def __call__(self, item: T_ITEM) -> T_ITEM:
        img, sem, dep = item

        img_size = (img.shape[1], img.shape[2])
        i, j, h, w = self.get_params(img_size, self.size)
        img = F.crop(img, i, j, h, w)
        if sem is not None:
            sem = F.crop(sem, i, j, h, w)
        if dep is not None:
            dep = F.crop(dep, i, j, h, w)

        return img, sem, dep


class ColorJitter:
    def __init__(
        self,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        hue: float = 0,
    ):
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, item: T_SAMPLE) -> T_SAMPLE:
        img, sem, dep = item
        img = self.color_jitter(img)
        return img, sem, dep


class ToTensor:
    def __call__(self, sample: T_SAMPLE) -> T_ITEM:
        img, sem, dep = sample
        image = F.to_tensor(img)
        if sem is not None:
            sem = torch.as_tensor(sem, dtype=torch.long)
        if dep is not None:
            dep = torch.as_tensor(dep, dtype=torch.float)
        return (image, sem, dep)


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std

    def __call__(self, item: T_ITEM) -> T_ITEM:
        img, sem, dep = item
        img = F.normalize(img, mean=list(self.mean), std=list(self.std))
        return img, sem, dep

import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode

T_SAMPLE = Tuple[Image.Image, Optional[np.ndarray], Optional[np.ndarray]]
T_ITEM = Tuple[Tensor, Optional[Tensor], Optional[Tensor]]
T_TRANSFORM = Callable[[T_ITEM], T_ITEM]


class Compose:
    def __init__(self, transforms: List[T_TRANSFORM]):
        self.to_tensor = ToTensor()
        self.transforms = transforms

    def __call__(self, sample: T_SAMPLE) -> T_ITEM:
        item = self.to_tensor(sample)
        for t in self.transforms:
            item = t(item)
        return item


class RandomHorizontalFlip:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, item: T_ITEM) -> T_ITEM:
        img, sem, dep = item

        if random.random() < self.p:
            img = F.hflip(img)
            if sem is not None:
                sem = F.hflip(sem)
            if dep is not None:
                dep = F.hflip(dep)

        return img, sem, dep


class Resize:
    def __init__(
        self,
        img_size: Tuple[int, int],
        sem_size: Tuple[int, int],
        dep_size: Tuple[int, int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        self.img_size = img_size
        self.sem_size = sem_size
        self.dep_size = dep_size
        self.interpolation = interpolation

    def __call__(self, item: T_ITEM) -> T_ITEM:
        img, sem, dep = item

        resized_img = F.resize(img, list(self.img_size), self.interpolation)
        resized_sem, resized_dep = None, None
        if sem is not None:
            sem.unsqueeze_(0)
            resized_sem = F.resize(
                sem,
                list(self.sem_size),
                interpolation=InterpolationMode.NEAREST,
            )
            resized_sem.squeeze_()
        if dep is not None:
            dep.unsqueeze_(0)
            resized_dep = F.resize(dep, list(self.dep_size), interpolation=self.interpolation)
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

        i = random.randint(0, ih - oh)
        j = random.randint(0, iw - ow)
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

    def __call__(self, item: T_ITEM) -> T_ITEM:
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

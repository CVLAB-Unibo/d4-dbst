from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data.dataset import Dataset as TorchDataset

from data.colormap import get_cmap
from data.semmap import get_semantic_map
from data.transforms import Compose
from data.utils import img2depth

SAMPLE_T = Sequence[Path]
ITEM_T = Tuple[Tensor, Optional[Tensor], Optional[Tensor]]


class Dataset(TorchDataset, ABC):
    def __init__(self, cfg: Dict[str, Any], transform: Compose) -> None:
        TorchDataset.__init__(self)
        ABC.__init__(self)

        root = Path(cfg["root"])
        input_file = cfg["input_file"]
        self.samples: List[SAMPLE_T] = []

        with open(input_file, "rt") as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                splits = line.split(";")
                image = root / Path(splits[0].strip())
                sem = root / Path(splits[1].strip())
                dep = root / Path(splits[2].strip())
                self.samples.append((image, sem, dep))

        self.sem = cfg["sem"]
        if self.sem:
            self.sem_map = get_semantic_map(cfg["sem_map"])
            self.sem_ignore_index = cfg["sem_ignore_index"]
            self.sem_cmap = get_cmap(cfg["sem_cmap"])

        self.dep = cfg["dep"]
        if self.dep:
            self.dep_min, self.dep_max = cfg["dep_range"]
            self.dep_cmap = get_cmap(cfg["dep_cmap"])

        self.transform = transform

    def encode_sem(self, sem_img: Image.Image) -> np.ndarray:
        sem = np.array(sem_img)
        sem_copy = sem.copy()
        sem_copy = self.sem_ignore_index * np.ones(sem_copy.shape, dtype=np.float32)
        for k, v in self.sem_map.items():
            sem_copy[sem == k] = v
        return sem_copy

    def get_dep_img(self, dep: Tensor) -> np.ndarray:
        inv_dep = 1 / dep
        norm_dep = (inv_dep - inv_dep.min()) / (inv_dep.max() - inv_dep.min())
        colored_dep = self.dep_cmap(np.array(norm_dep))
        return colored_dep[..., :3]

    def get_sem_img(self, sem: Tensor) -> np.ndarray:
        colored_sem = self.sem_cmap(np.array(sem))
        return colored_sem[..., :3]

    def __getitem__(self, index: int) -> ITEM_T:
        image_path, sem_path, dep_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        sem, dep = None, None
        if self.sem:
            sem = self.encode_sem(Image.open(sem_path))
        if self.dep:
            dep = img2depth(Image.open(dep_path))
            dep = np.clip(dep, self.dep_min, self.dep_max)

        sample = (image, sem, dep)
        return self.transform(sample)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def collate_fn(batches: List[ITEM_T]) -> ITEM_T:
        imgs: List[Tensor] = []
        sems: List[Tensor] = []
        deps: List[Tensor] = []

        for batch in batches:
            img, sem, dep = batch
            imgs.append(img)
            if sem is not None:
                sems.append(sem)
            if dep is not None:
                deps.append(dep)

        img_stack = torch.stack(imgs, 0)
        sem_stack = torch.stack(sems, 0) if len(sems) > 0 else None
        dep_stack = torch.stack(deps, 0) if len(deps) > 0 else None

        return img_stack, sem_stack, dep_stack

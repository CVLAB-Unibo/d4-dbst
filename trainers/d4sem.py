from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from hesiod import hcfg
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils import model_zoo
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from data.dataset import Dataset
from data.transforms import IMAGENET_MEAN, IMAGENET_STD, ColorJitter, Compose, Normalize
from data.transforms import RandomHorizontalFlip, Resize, ToTensor
from models.deeplab import Res_Deeplab
from trainers.metrics import IoU


class D4SemanticsTrainer:
    def __init__(self) -> None:
        train_dset_cfg = hcfg("sem.train_dataset", Dict[str, Any])
        img_size = hcfg("sem.img_size", Tuple[int, int])
        train_sem_size = hcfg("sem.train_sem_size", Tuple[int, int])

        train_transforms = [
            ToTensor(),
            Resize(img_size, train_sem_size, (-1, -1)),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5,
            ),
        ]

        train_transform = Compose(train_transforms)
        train_dset = Dataset(train_dset_cfg, train_transform)

        train_bs = hcfg("sem.train_bs", int)
        self.train_loader = DataLoader(
            train_dset,
            train_bs,
            shuffle=True,
            num_workers=8,
            collate_fn=Dataset.collate_fn,
        )

        val_source_dset_cfg = hcfg("sem.val_source_dataset", Dict[str, Any])
        val_target_dset_cfg = hcfg("sem.val_target_dataset", Dict[str, Any])
        val_sem_size = hcfg("sem.val_sem_size", Tuple[int, int])

        val_transforms = [
            ToTensor(),
            Resize(img_size, val_sem_size, (-1, -1)),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        val_transform = Compose(val_transforms)

        val_source_dset = Dataset(val_source_dset_cfg, val_transform)
        val_target_dset = Dataset(val_target_dset_cfg, val_transform)

        val_bs = hcfg("sem.val_bs", int)
        self.val_source_loader = DataLoader(
            val_source_dset,
            val_bs,
            num_workers=8,
            collate_fn=Dataset.collate_fn,
        )
        self.val_target_loader = DataLoader(
            val_target_dset,
            val_bs,
            num_workers=8,
            collate_fn=Dataset.collate_fn,
        )

        num_classes = hcfg("num_classes", int)
        self.model = Res_Deeplab(num_classes=num_classes).cuda()
        self.model.eval()

        url = "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
        saved_state_dict = model_zoo.load_url(url)
        new_params = self.model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = str(i).split(".")
            if not i_parts[0] == "fc":
                new_params[".".join(i_parts[0:])] = saved_state_dict[i]
        self.model.load_state_dict(new_params)

        lr = hcfg("sem.lr", float)
        self.num_epochs = hcfg("sem.num_epochs", int)
        self.optimizer = SGD(self.model.optim_parameters(lr), lr=lr)
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=len(self.train_loader),
            epochs=self.num_epochs,
            div_factor=20,
        )

        ignore_index = hcfg("sem.ignore_index", int)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        logdir = hcfg("sem.logdir", str)
        self.summary_writer = SummaryWriter(logdir)

        self.iou = IoU(num_classes=num_classes + 1, ignore_index=ignore_index, valid=da_capire)

        self.global_step = 0

    def train(self) -> None:
        for epoch in range(self.num_epochs):

            self.iou.reset()

            for batch in tqdm(self.train_loader, f"Epoch {epoch}/{self.num_epochs}"):
                images, labels, _ = batch
                images = images.cuda()
                labels = labels.cuda()

                pred = self.model(images)
                h, w = labels.shape[2], labels.shape[3]
                pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)

                loss = self.loss_fn(pred, labels)

                if self.global_step % 100 == 99:
                    self.summary_writer.add_scalar("train/loss", loss.item(), self.global_step)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                self.iou.add(pred.detach(), labels)

                self.global_step += 1

            train_miou = self.iou.value()[0]
            self.summary_writer.add_scalar("train/miou", train_miou, self.global_step)

            val_source_miou = self.val("source")
            self.summary_writer.add_scalar("val_source/miou", val_source_miou, self.global_step)

            val_target_miou = self.val("target")
            self.summary_writer.add_scalar("val_target/miou", val_target_miou, self.global_step)

    @torch.no_grad()
    def val(self, dataset: str) -> float:
        loader = self.val_source_loader if dataset == "source" else self.val_target_loader

        self.iou.reset()

        for batch in tqdm(loader, f"Validating on {dataset}"):
            images, labels, _ = batch
            images = images.cuda()
            labels = labels.cuda()

            pred = self.model(images)
            h, w = labels.shape[2], labels.shape[3]
            pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)

            self.iou.add(pred.detach(), labels)

        return self.iou.value()[0]

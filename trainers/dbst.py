from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hesiod import get_out_dir, hcfg
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torch.utils import model_zoo
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from data.dataset import Dataset
from data.transforms import IMAGENET_MEAN, IMAGENET_STD, ColorJitter, Compose, Normalize
from data.transforms import RandomCrop, RandomHorizontalFlip, RandomScale, Resize, ToTensor
from data.utils import denormalize
from models.deeplab import Res_Deeplab
from trainers.metrics import IoU
from utils import progress_bar


class D4SemanticsTrainer:
    def __init__(self) -> None:
        img_size = hcfg("img_size", Tuple[int, int])
        sem_cmap = hcfg("sem_cmap", str)
        sem_num_classes = hcfg("sem_num_classes", int)
        sem_ignore_index = hcfg("sem_ignore_index", int)

        train_dset_root = Path(hcfg("train_dataset.root", str))
        train_input_file = Path(hcfg("train_dataset.input_file", str))
        train_sem_size = hcfg("train_sem_size", Tuple[int, int])
        train_crop_size = hcfg("train_crop_size", Tuple[int, int])
        train_sem_map = hcfg("train_dataset.sem_map", str)

        train_transforms = [
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            ToTensor(),
            Resize(img_size, train_sem_size, (-1, -1)),
            RandomHorizontalFlip(p=0.5),
            RandomScale((0.5, 1.5)),
            RandomCrop(train_crop_size),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        train_transform = Compose(train_transforms)

        self.train_dset = Dataset(
            train_dset_root,
            train_input_file,
            True,
            train_sem_map,
            sem_ignore_index,
            sem_cmap,
            False,
            (-1, -1),
            "",
            train_transform,
        )

        train_bs = hcfg("train_bs", int)
        self.train_loader = DataLoader(
            self.train_dset,
            train_bs,
            shuffle=True,
            num_workers=8,
            collate_fn=Dataset.collate_fn,  # type: ignore
        )

        val_dset_root = Path(hcfg("val_dataset.root", str))
        val_input_file = Path(hcfg("val_dataset.input_file", str))
        val_sem_map = hcfg("val_dataset.sem_map", str)
        val_sem_size = hcfg("val_sem_size", Tuple[int, int])

        val_transforms = [
            ToTensor(),
            Resize(img_size, val_sem_size, (-1, -1)),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        val_transform = Compose(val_transforms)

        self.val_dset = Dataset(
            val_dset_root,
            val_input_file,
            True,
            val_sem_map,
            sem_ignore_index,
            sem_cmap,
            False,
            (-1, -1),
            "",
            val_transform,
        )

        val_bs = hcfg("val_bs", int)
        self.val_loader = DataLoader(
            self.val_dset,
            val_bs,
            num_workers=8,
            collate_fn=Dataset.collate_fn,  # type: ignore
        )

        self.model = Res_Deeplab(num_classes=sem_num_classes, resnet101=True).cuda()
        self.model.eval()

        url = "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
        saved_state_dict = model_zoo.load_url(url)  # type: ignore
        new_params = self.model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = str(i).split(".")
            if not i_parts[0] == "fc":
                new_params[".".join(i_parts[0:])] = saved_state_dict[i]
        self.model.load_state_dict(new_params)

        lr = hcfg("lr", float)
        self.num_epochs = hcfg("num_epochs", int)
        self.optimizer = SGD(self.model.optim_parameters(lr), lr=lr, weight_decay=1e-4)
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=len(self.train_loader),
            epochs=self.num_epochs,
            div_factor=20,
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=sem_ignore_index)
        self.iou = IoU(num_classes=sem_num_classes + 1, ignore_index=sem_ignore_index)

        self.logdir = get_out_dir() / "dbst"
        self.summary_writer = SummaryWriter(self.logdir / "tensorboard")
        self.global_step = 0

    def train(self) -> None:
        for epoch in range(self.num_epochs):

            self.iou.reset()

            for batch in progress_bar(self.train_loader, f"Epoch {epoch}/{self.num_epochs}"):
                images, labels, _ = batch
                images = images.cuda()
                labels = labels.cuda()

                pred = self.model(images)
                h, w = labels.shape[2], labels.shape[3]
                pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)

                loss = self.loss_fn(pred, labels)

                if self.global_step % 100 == 0:
                    self.summary_writer.add_scalar("train/loss", loss.item(), self.global_step)

                    img = np.array(images[0].detach().cpu())
                    img = denormalize(img, IMAGENET_MEAN, IMAGENET_STD)
                    sem_img_gt = self.train_dset.get_sem_img(labels[0].detach().cpu())
                    pred_labels = torch.argmax(pred[0].detach().cpu(), dim=0)
                    sem_img_pred = self.train_dset.get_sem_img(pred_labels)

                    self.summary_writer.add_image("train/image", img, self.global_step)
                    self.summary_writer.add_image(
                        "train/gt",
                        sem_img_gt,
                        self.global_step,
                        dataformats="HWC",
                    )
                    self.summary_writer.add_image(
                        "train/pred",
                        sem_img_pred,
                        self.global_step,
                        dataformats="HWC",
                    )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                self.iou.add(pred.detach(), labels)

                self.global_step += 1

            train_miou = self.iou.value()[0]
            self.summary_writer.add_scalar("train/miou", train_miou, self.global_step)

            val_miou = self.val()
            self.summary_writer.add_scalar("val/miou", val_miou, self.global_step)

        ckpt_path = self.logdir / "ckpt.pt"
        ckpt = {"model": self.model}
        torch.save(ckpt, ckpt_path)

    @torch.no_grad()
    def val(self) -> float:
        self.iou.reset()

        for batch in progress_bar(self.val_loader, "Validating"):
            images, labels, _ = batch
            images = images.cuda()
            labels = labels.cuda()

            pred = self.model(images)
            h, w = labels.shape[2], labels.shape[3]
            pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)

            self.iou.add(pred.detach(), labels)

        img = np.array(images[0].detach().cpu())  # type: ignore
        img = denormalize(img, IMAGENET_MEAN, IMAGENET_STD)
        sem_img_gt = self.val_dset.get_sem_img(labels[0].detach().cpu())  # type: ignore
        pred_labels = torch.argmax(pred[0].detach().cpu(), dim=0)  # type: ignore
        sem_img_pred = self.val_dset.get_sem_img(pred_labels)

        self.summary_writer.add_image("val/image", img, self.global_step)
        self.summary_writer.add_image(
            "val/gt",
            sem_img_gt,
            self.global_step,
            dataformats="HWC",
        )
        self.summary_writer.add_image(
            "val/pred",
            sem_img_pred,
            self.global_step,
            dataformats="HWC",
        )

        return self.iou.value()[0]

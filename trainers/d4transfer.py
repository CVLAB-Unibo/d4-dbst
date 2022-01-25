from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hesiod import get_out_dir, hcfg
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from data.dataset import Dataset
from data.transforms import IMAGENET_MEAN, IMAGENET_STD, ColorJitter, Compose, Normalize
from data.transforms import RandomHorizontalFlip, ToTensor
from data.utils import denormalize
from models.d4transfer import Transfer
from models.deeplab import Res_Deeplab
from trainers.metrics import IoU
from utils import progress_bar


class D4TransferTrainer:
    def __init__(self) -> None:
        img_size = hcfg("transfer.img_size", Tuple[int, int])
        sem_cmap = hcfg("transfer.sem_cmap", str)
        sem_num_classes = hcfg("transfer.sem_num_classes", int)
        sem_ignore_index = hcfg("transfer.sem_ignore_index", int)

        train_dset_root = Path(hcfg("transfer.train_dataset.root", str))
        train_input_file = Path(hcfg("transfer.train_dataset.input_file", str))
        train_sem_size = hcfg("transfer.train_sem_size", Tuple[int, int])
        train_sem_map = hcfg("transfer.train_dataset.sem_map", str)

        train_transforms = [
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            ToTensor(),
            RandomHorizontalFlip(p=0.5),
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
            img_size,
            train_sem_size,
        )

        train_bs = hcfg("transfer.train_bs", int)
        self.train_loader = DataLoader(
            self.train_dset,
            train_bs,
            shuffle=True,
            num_workers=8,
            collate_fn=Dataset.collate_fn,  # type: ignore
        )

        val_source_dset_root = Path(hcfg("transfer.val_source_dataset.root", str))
        val_source_input_file = Path(hcfg("transfer.val_source_dataset.input_file", str))
        val_source_sem_map = hcfg("transfer.val_source_dataset.sem_map", str)
        val_target_dset_root = Path(hcfg("transfer.val_target_dataset.root", str))
        val_target_input_file = Path(hcfg("transfer.val_target_dataset.input_file", str))
        val_target_sem_map = hcfg("transfer.val_target_dataset.sem_map", str)
        val_sem_size = hcfg("transfer.val_sem_size", Tuple[int, int])

        val_transforms = [
            ToTensor(),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        val_transform = Compose(val_transforms)

        val_source_dset = Dataset(
            val_source_dset_root,
            val_source_input_file,
            True,
            val_source_sem_map,
            sem_ignore_index,
            sem_cmap,
            False,
            (-1, -1),
            "",
            val_transform,
            img_size,
            val_sem_size,
        )
        val_target_dset = Dataset(
            val_target_dset_root,
            val_target_input_file,
            True,
            val_target_sem_map,
            sem_ignore_index,
            sem_cmap,
            False,
            (-1, -1),
            "",
            val_transform,
            img_size,
            val_sem_size,
        )

        val_bs = hcfg("transfer.val_bs", int)
        self.val_source_loader = DataLoader(
            val_source_dset,
            val_bs,
            num_workers=8,
            collate_fn=Dataset.collate_fn,  # type: ignore
        )
        self.val_target_loader = DataLoader(
            val_target_dset,
            val_bs,
            num_workers=8,
            collate_fn=Dataset.collate_fn,  # type: ignore
        )

        model_dep = Res_Deeplab(num_classes=1, use_sigmoid=True).cuda()
        dep_ckpt_path = get_out_dir() / "d4dep/ckpt.pt"
        dep_ckpt = torch.load(dep_ckpt_path)
        model_dep.load_state_dict(dep_ckpt["model"])

        for p in model_dep.parameters():
            p.requires_grad = False
        model_dep.eval()

        model_sem = Res_Deeplab(num_classes=sem_num_classes).cuda()
        sem_ckpt_path = get_out_dir() / "d4sem/ckpt.pt"
        sem_ckpt = torch.load(sem_ckpt_path)
        model_sem.load_state_dict(sem_ckpt["model"])

        for p in model_sem.parameters():
            p.requires_grad = False
        model_sem.eval()

        self.dep_encoder = torch.nn.Sequential(*(list(model_dep.children())[:-2]))
        self.sem_encoder = torch.nn.Sequential(*(list(model_sem.children())[:-1]))
        self.sem_decoder = list(model_sem.children())[-1]

        self.transfer = Transfer(2048, 1024, 2048).cuda()

        lr = hcfg("transfer.lr", float)
        self.num_epochs = hcfg("transfer.num_epochs", int)
        self.optimizer = AdamW(self.transfer.parameters(), lr=lr)
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=len(self.train_loader),
            epochs=self.num_epochs,
            div_factor=20,
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=sem_ignore_index)
        self.iou = IoU(num_classes=sem_num_classes + 1, ignore_index=sem_ignore_index)

        self.logdir = get_out_dir() / "d4_transfer"
        self.summary_writer = SummaryWriter(self.logdir / "tensorboard")
        self.global_step = 0

    def train(self) -> None:
        for epoch in range(self.num_epochs):
            self.transfer.train()

            for batch in progress_bar(self.train_loader, f"Epoch {epoch}/{self.num_epochs}"):
                images, labels, _ = batch
                images = images.cuda()
                labels = labels.cuda()

                output_dep_encoder = self.dep_encoder(images)
                output_sem_encoder = self.sem_encoder(images)
                output_transfer = self.transfer(output_dep_encoder)

                loss = F.mse_loss(output_transfer, output_sem_encoder)

                if self.global_step % 100 == 0:
                    self.summary_writer.add_scalar("train/loss", loss.item(), self.global_step)

                    with torch.no_grad():
                        pred = self.sem_decoder(output_transfer)

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

                self.global_step += 1

            val_source_miou = self.val("source")
            self.summary_writer.add_scalar("val_source/miou", val_source_miou, self.global_step)

            val_target_miou = self.val("target")
            self.summary_writer.add_scalar("val_target/miou", val_target_miou, self.global_step)

        ckpt_path = self.logdir / "ckpt.pt"
        ckpt = {
            "dep_encoder": self.dep_encoder.state_dict(),
            "sem_decoder": self.sem_decoder.state_dict(),
            "transfer": self.transfer.state_dict(),
        }
        torch.save(ckpt, ckpt_path)

    @torch.no_grad()
    def val(self, dataset: str) -> float:
        loader = self.val_source_loader if dataset == "source" else self.val_target_loader
        self.transfer.eval()
        self.iou.reset()

        for batch in progress_bar(loader, f"Validating on {dataset}"):
            images, labels, _ = batch
            images = images.cuda()
            labels = labels.cuda()

            output_dep_encoder = self.dep_encoder(images)
            output_transfer = self.transfer(output_dep_encoder)
            pred = self.sem_decoder(output_transfer)

            h, w = labels.shape[1], labels.shape[2]
            pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)

            self.iou.add(pred.detach(), labels)

        img = np.array(images[0].detach().cpu())  # type: ignore
        img = denormalize(img, IMAGENET_MEAN, IMAGENET_STD)
        sem_img_gt = loader.dataset.get_sem_img(labels[0].detach().cpu())  # type: ignore
        pred_labels = torch.argmax(pred[0].detach().cpu(), dim=0)  # type: ignore
        sem_img_pred = loader.dataset.get_sem_img(pred_labels)

        self.summary_writer.add_image(f"val_{dataset}/image", img, self.global_step)
        self.summary_writer.add_image(
            f"val_{dataset}/gt",
            sem_img_gt,
            self.global_step,
            dataformats="HWC",
        )
        self.summary_writer.add_image(
            f"val_{dataset}/pred",
            sem_img_pred,
            self.global_step,
            dataformats="HWC",
        )

        return self.iou.value()[0]

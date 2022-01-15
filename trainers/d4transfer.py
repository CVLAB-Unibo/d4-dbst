from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from hesiod import hcfg
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils import model_zoo
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from data.dataset import Dataset
from data.transforms import IMAGENET_MEAN, IMAGENET_STD, ColorJitter, Compose, Normalize
from data.transforms import RandomHorizontalFlip, Resize, ToTensor
from models.d4transfer import Transfer
from models.deeplab import Res_Deeplab
from trainers.metrics import IoU


class D4TransferTrainer:
    def __init__(self) -> None:
        train_dset_cfg = hcfg("transfer.train_dataset", Dict[str, Any])
        img_size = hcfg("transfer.img_size", Tuple[int, int])
        train_sem_size = hcfg("transfer.train_sem_size", Tuple[int, int])

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

        train_bs = hcfg("transfer.train_bs", int)
        self.train_loader = DataLoader(
            train_dset,
            train_bs,
            shuffle=True,
            num_workers=8,
            collate_fn=Dataset.collate_fn,
        )

        val_source_dset_cfg = hcfg("transfer.val_source_dataset", Dict[str, Any])
        val_target_dset_cfg = hcfg("transfer.val_target_dataset", Dict[str, Any])
        val_sem_size = hcfg("transfer.val_sem_size", Tuple[int, int])

        val_transforms = [
            ToTensor(),
            Resize(img_size, val_sem_size, (-1, 1)),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        val_transform = Compose(val_transforms)

        val_source_dset = Dataset(val_source_dset_cfg, val_transform)
        val_target_dset = Dataset(val_target_dset_cfg, val_transform)

        val_bs = hcfg("transfer.val_bs", int)
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

        num_classes = hcfg("sem.train_dataset.sem_num_classes", int)

        model_dep = Res_Deeplab(num_classes=1, use_sigmoid=True).cuda()
        dep_ckpt_path = hcfg("transfer.dep_ckpt_path", str)
        dep_ckpt = torch.load(dep_ckpt_path)
        model_dep.load_state_dict(dep_ckpt["model"])

        for p in model_dep.parameters():
            p.requires_grad = False
        model_dep.eval()

        model_sem = Res_Deeplab(num_classes=num_classes).cuda()
        sem_ckpt_path = hcfg("transfer.sem_ckpt_path", str)
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

        ignore_index = hcfg("sem.train_dataset.sem_ignore_idx", int)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.logdir = hcfg("transfer.logdir", str)
        self.summary_writer = SummaryWriter(self.logdir + "/tensorboard")

        self.iou = IoU(num_classes=num_classes + 1, ignore_index=ignore_index)

        self.global_step = 0

    def train(self) -> None:
        for epoch in range(self.num_epochs):

            for batch in tqdm(self.train_loader, f"Epoch {epoch}/{self.num_epochs}"):
                images, labels, _ = batch
                images = images.cuda()
                labels = labels.cuda()

                output_dep_encoder = self.dep_encoder(images)
                output_sem_encoder = self.sem_encoder(images)
                output_transfer = self.transfer(output_dep_encoder)

                loss = F.mse_loss(output_transfer, output_sem_encoder)

                if self.global_step % 100 == 99:
                    self.summary_writer.add_scalar("train/loss", loss.item(), self.global_step)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                self.global_step += 1

            val_source_miou = self.val("source")
            self.summary_writer.add_scalar("val_source/miou", val_source_miou, self.global_step)

            val_target_miou = self.val("target")
            self.summary_writer.add_scalar("val_target/miou", val_target_miou, self.global_step)

        ckpt_path = self.logdir + "ckpt.pt"
        ckpt = {
            "dep_encoder": self.dep_encoder,
            "sem_decoder": self.sem_decoder,
            "trasnfer": self.transfer,
        }
        torch.save(ckpt, ckpt_path)

    @torch.no_grad()
    def val(self, dataset: str) -> float:
        loader = self.val_source_loader if dataset == "source" else self.val_target_loader

        self.iou.reset()

        for batch in tqdm(loader, f"Validating on {dataset}"):
            images, labels, _ = batch
            images = images.cuda()
            labels = labels.cuda()

            output_dep_encoder = self.dep_encoder(images)
            output_transfer = self.transfer(output_dep_encoder)
            pred = self.sem_decoder(output_transfer)

            h, w = labels.shape[2], labels.shape[3]
            pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)

            self.iou.add(pred.detach(), labels)

        return self.iou.value()[0]

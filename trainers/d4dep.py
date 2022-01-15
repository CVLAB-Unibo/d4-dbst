from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hesiod import get_out_dir, hcfg
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils import model_zoo
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from data.dataset import Dataset
from data.transforms import IMAGENET_MEAN, IMAGENET_STD, ColorJitter, Compose, Normalize
from data.transforms import RandomHorizontalFlip, Resize
from data.utils import denormalize
from models.deeplab import Res_Deeplab
from trainers.losses import MaskedL1Loss
from trainers.metrics import RMSE


class D4DepthTrainer:
    def __init__(self) -> None:
        train_dset_cfg = hcfg("dep.train_dataset", Dict[str, Any])
        img_size = hcfg("dep.img_size", Tuple[int, int])
        train_dep_size = hcfg("dep.train_dep_size", Tuple[int, int])

        train_transforms = [
            Resize(img_size, (-1, -1), train_dep_size),
            # ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            RandomHorizontalFlip(p=0.5),
        ]

        train_transform = Compose(train_transforms)
        self.train_dset = Dataset(train_dset_cfg, train_transform)

        train_bs = hcfg("dep.train_bs", int)
        self.train_loader = DataLoader(
            self.train_dset,
            train_bs,
            shuffle=True,
            num_workers=8,
            collate_fn=Dataset.collate_fn,
        )

        val_source_dset_cfg = hcfg("dep.val_source_dataset", Dict[str, Any])
        val_target_dset_cfg = hcfg("dep.val_target_dataset", Dict[str, Any])
        val_dep_size = hcfg("dep.val_dep_size", Tuple[int, int])

        val_transforms = [
            Resize(img_size, (-1, -1), val_dep_size),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        val_transform = Compose(val_transforms)

        self.val_source_dset = Dataset(val_source_dset_cfg, val_transform)
        self.val_target_dset = Dataset(val_target_dset_cfg, val_transform)

        val_bs = hcfg("dep.val_bs", int)
        self.val_source_loader = DataLoader(
            self.val_source_dset,
            val_bs,
            num_workers=8,
            collate_fn=Dataset.collate_fn,
        )
        self.val_target_loader = DataLoader(
            self.val_target_dset,
            val_bs,
            num_workers=8,
            collate_fn=Dataset.collate_fn,
        )

        self.model = Res_Deeplab(num_classes=1, use_sigmoid=True).cuda()
        self.model.eval()

        url = "https://download.pytorch.org/models/resnet50-19c8e357.pth"
        saved_state_dict = model_zoo.load_url(url)
        new_params = self.model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = str(i).split(".")
            if not i_parts[0] == "fc":
                new_params[".".join(i_parts[0:])] = saved_state_dict[i]
        self.model.load_state_dict(new_params)

        lr = hcfg("dep.lr", float)
        self.num_epochs = hcfg("dep.num_epochs", int)
        self.optimizer = AdamW(self.model.optim_parameters(lr), lr=lr)
        self.lr_scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=len(self.train_loader),
            epochs=self.num_epochs,
            div_factor=20,
        )

        self.dep_range = hcfg("dep.train_dataset.dep_range", Tuple[float, float])
        self.loss_fn = MaskedL1Loss(self.dep_range[1])

        self.summary_writer = SummaryWriter(get_out_dir() / "d4dep/tensorboard")

        self.rmse = RMSE(min_depth=self.dep_range[0], max_depth=self.dep_range[1])

        self.global_step = 0

    def train(self) -> None:
        for epoch in range(self.num_epochs):
            self.rmse.reset()

            for batch in tqdm(self.train_loader, f"Epoch {epoch}/{self.num_epochs}"):
                images, _, labels = batch
                images = images.cuda()
                labels = labels.cuda()

                pred = self.model(images)
                h, w = labels.shape[1], labels.shape[2]
                pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)
                pred = pred.squeeze(1)

                loss = self.loss_fn(pred, labels)

                if self.global_step % 100 == 0:
                    self.summary_writer.add_scalar("train/loss", loss.item(), self.global_step)

                    img = np.array(images[0].detach().cpu())
                    img = denormalize(img, IMAGENET_MEAN, IMAGENET_STD)
                    dep_img_gt = self.train_dset.get_dep_img(labels[0].detach().cpu())
                    dep_img_pred = self.train_dset.get_dep_img(pred[0].detach().cpu())

                    self.summary_writer.add_image("train/image", img, self.global_step)
                    self.summary_writer.add_image(
                        "train/gt",
                        dep_img_gt,
                        self.global_step,
                        dataformats="HWC",
                    )
                    self.summary_writer.add_image(
                        "train/pred",
                        dep_img_pred,
                        self.global_step,
                        dataformats="HWC",
                    )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                self.rmse.add(pred.detach(), labels)

                self.global_step += 1

            train_rmse = self.rmse.value()[0]
            self.summary_writer.add_scalar("train/rmse", train_rmse, self.global_step)

            val_source_rmse = self.val("source")
            self.summary_writer.add_scalar("val_source/rmse", val_source_rmse, self.global_step)

            val_target_rmse = self.val("target")
            self.summary_writer.add_scalar("val_target/rmse", val_target_rmse, self.global_step)

        ckpt_path = self.logdir + "ckpt.pt"
        ckpt = {"model": self.model}
        torch.save(ckpt, ckpt_path)

    @torch.no_grad()
    def val(self, dataset: str) -> float:
        loader = self.val_source_loader if dataset == "source" else self.val_target_loader

        self.rmse.reset()

        for batch in tqdm(loader, f"Validating on {dataset}"):
            images, _, labels = batch
            images = images.cuda()
            labels = labels.cuda()

            pred = self.model(images)
            h, w = labels.shape[1], labels.shape[2]
            pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)
            pred = pred.squeeze(1)

            self.rmse.add(pred.detach(), labels)

        img = np.array(images[0].detach().cpu())
        img = denormalize(img, IMAGENET_MEAN, IMAGENET_STD)
        dep_img_gt = self.val_source_dset.get_dep_img(labels[0].detach().cpu())
        dep_img_pred = self.val_source_dset.get_dep_img(pred[0].detach().cpu())

        self.summary_writer.add_image(f"val_{dataset}/image", img, self.global_step)
        self.summary_writer.add_image(
            f"val_{dataset}/gt",
            dep_img_gt,
            self.global_step,
            dataformats="HWC",
        )
        self.summary_writer.add_image(
            f"val_{dataset}/pred",
            dep_img_pred,
            self.global_step,
            dataformats="HWC",
        )

        return self.rmse.value()[0]

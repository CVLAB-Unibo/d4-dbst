import os
import io
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple
import random
import cv2
import glob

class SegmentationDataset(Dataset):

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                    'has_instances', 'ignore_in_eval', 'color'])

    cs = [
        CityscapesClass('unlabeled',            0, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle',          1, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi',           3, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static',               4, 19, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic',              5, 19, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground',               6, 19, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking',              9, 19, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track',           10, 19, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail',           14, 19, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge',               15, 19, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel',               16, 19, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup',            18, 19, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan',              29, 19, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer',              30, 19, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('unknown',              34, 19, 'void', 7, True, False, (0, 0, 0)),
        CityscapesClass('license plate',        -1, 19, 'vehicle', 7, False, True, (0, 0, 0)),
    ]

    synthiaSEQ_encoding = {
        CityscapesClass('unknown',              0, 12,  'void', 7, True, False, (0, 0, 0)),
        CityscapesClass('sky',                  1, 0, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('building',             2, 1, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('road',                 3, 2, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk',             4, 3, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('fence',                5, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('vegetation',           6, 5, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('pole',                 7, 6, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('car',                  8, 7, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('traffic sign',         9, 8, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('person',               10, 9, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('bicycle',              11, 10, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('lanemarking',          12, 2, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('unknown2',             13, 12,  'void', 7, True, False, (0, 0, 0)),
        CityscapesClass('unknown3',             14, 12,  'void', 7, True, False, (0, 0, 0)),
        CityscapesClass('traffic light',        15, 11,  'void', 7, True, False, (250, 170, 30)),
    }

    synthiaSEQ2cs_encoding = {
        CityscapesClass('road',                 0, 2, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sky',                  10, 0, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('building',             2, 1, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('sidewalk',             1, 3, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('fence',                4, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('vegetation',           8, 5, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('pole',                 5, 6, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('car',                  13, 7, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('traffic sign',         7, 8, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('person',               11, 9, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('bicycle',              18, 10, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('wall',                 3, 12, 'construction', 2, False, False, (0, 0, 0)),
        CityscapesClass('traffic light',        6, 11, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('terrain',              9, 12, 'nature', 4, False, False, (0, 0, 0)),
        CityscapesClass('rider',                12, 10, 'human', 6, True, False, (119, 11, 32)),
        CityscapesClass('truck',                14, 7, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('bus',                  15, 7, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('train',                16, 12, 'vehicle', 7, True, False, (0, 0, 0)),
        CityscapesClass('motorcycle',           17, 12, 'vehicle', 7, True, False, (0, 0, 0)),
        CityscapesClass('unknown',              19, 12,  'void', 7, True, False, (0, 0, 0)),
    }

    def __init__(self, root, train_augmented_dir, txt_file, transforms=None, encoding='cs',
                mean=[0.286, 0.325, 0.283], std=[0.176, 0.180, 0.177], size=(1024, 512), label_size=(2048, 1024), val=False):
    
        super(SegmentationDataset, self).__init__()
        if encoding == 'carla':
            self.encoding = self.carla
        elif encoding == 'cs2carla':
            self.encoding = self.cs2carla
        elif encoding == 'cs':
            self.encoding = self.cs
        elif encoding == 'synthiaSEQ':
            self.encoding = self.synthiaSEQ_encoding
        elif encoding == 'synthiaSEQ2cs':
            self.encoding = self.synthiaSEQ2cs_encoding

        self.id_to_trainId = {cs_class.id: cs_class.train_id for cs_class in self.encoding}
        self.palette = []
        self.files_txt = txt_file
        self.images = []
        self.labels = []
        self.root = root
        self.transforms = transforms
        self.size = size
        self.label_size = label_size
        self.val = val
        self.lscale, self.hscale = (0.5, 1.5)
        self.ignore_label = 19

        if not self.val:
            for image_name in glob.glob(os.path.join(train_augmented_dir, "*.png")):
                self.images.append(image_name)
                self.labels.append(image_name.replace('train', 'gtFine', 1))
        else:
            for line in open(self.files_txt, 'r').readlines():
                splits = line.split(';')
                self.images.append(os.path.join(root, splits[0].strip()))
                self.labels.append(os.path.join(root, splits[1].strip()))

        self.colors = {cs_class.train_id: cs_class.color for cs_class in self.encoding}
        for train_id, color in sorted(self.colors.items(), key=lambda item: item[0]):
            R, G, B = color
            self.palette.extend((R, G, B))

        zero_pad = 256 * 3 - len(self.palette)
        for i in range(zero_pad):
            self.palette.append(0)

        self.mean = mean
        self.std = std

    def encode_image_train_id(self, mask):
        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainId.items():
            mask_copy[mask == k] = v
        return mask_copy


    def __getitem__(self, index):
        img_path, mask_path = self.images[index], self.labels[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)
            
        if self.size is not None:
            img = img.resize(self.size, Image.LANCZOS)
            mask = mask.resize(self.label_size, Image.NEAREST)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)
        else:
            img = np.asarray(img, np.float32)
            mask = np.array(mask)
            # print(self.mean, self.std)
            # img = img/255.0
            img = img[..., ::-1]
            img -= self.mean
            img /= self.std
            img = img.transpose((2, 0, 1)).copy()
            # mask = self.encode_image_train_id(mask)
            mask = torch.from_numpy(mask)
            # print(img.max(), img.min())

        return img, mask.type(torch.long) 

    def __len__(self):
        return len(self.images)

    def colorize_mask(self, mask, encode_with_train_id=False):
        mask = np.array(mask, dtype=np.uint8)
        if encode_with_train_id:
            mask = self.encode_image_train_id(mask)
        new_mask = Image.fromarray(mask).convert('P')
        new_mask.putpalette(self.palette)

        return new_mask

    def re_normalize (self, image, mean, std):
        image = np.array(image, np.float32)
        image = image.transpose((1, 2, 0))
        image *= self.std
        image += self.mean
        image = image[:, :, ::-1]
        return np.uint8(image) 

    def get_predictions_plot(self, batch_sample, predictions, batch_gt, encode_gt=False):

        num_images = min(batch_sample.size()[0], 4)
        fig, m_axs = plt.subplots(3, num_images, figsize=(12, 8), squeeze=False)
        plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
        if predictions.dim() == 4:
            predictions = torch.argmax(predictions, dim=1)

        for image, prediction, gt, (axis1, axis2, axis3) in zip(batch_sample, predictions, batch_gt, m_axs.T):
            
            image = self.re_normalize(image, self.mean, self.std)
            # image = to_pil_image(image)
            axis1.imshow(image)
            axis1.set_axis_off()

            prediction = self.colorize_mask(prediction)
            axis2.imshow(prediction)
            axis2.set_axis_off()
            
            gt = self.colorize_mask(gt, encode_with_train_id=encode_gt)
            axis3.imshow(gt)
            axis3.set_axis_off()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches = 'tight', pad_inches = 0)
        buf.seek(0)
        im = Image.open(buf)
        figure = np.array(im)
        buf.close()
        plt.close(fig)
        return figure
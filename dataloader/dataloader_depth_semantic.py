import os
import io
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple


class SemanticDepth(Dataset):
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

    def __init__(self, root, txt_file, transforms=None, 
                max_depth=1000, threshold=100,
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.225, 0.224], use_depth=False, size=(2048, 1024), label_size=(2048, 1024)):

        super(SemanticDepth, self).__init__()
        self.encoding = self.cs

        self.id_to_trainId = {cs_class.id: cs_class.train_id for cs_class in self.encoding}

        self.palette = []
        self.files_txt = txt_file
        self.images = []
        self.labels = []
        self.labels_semantic = []
        self.root = root
        self.max_depth = max_depth
        self.treshold = threshold
        self.transforms = transforms
        self.cm = plt.cm.get_cmap('jet')
        self.colors = self.cm(np.arange(256))[:,:3]
        self.mean = mean
        self.std = std
        self.size = size
        self.label_size = label_size
        self.use_depth = use_depth
        
        for line in open(self.files_txt, 'r').readlines():
            splits = line.split(';')
            self.images.append(os.path.join(root, splits[0].strip()))
            self.labels_semantic.append(os.path.join(root, splits[1].strip()))
            self.labels.append(os.path.join(root, splits[2].strip()))

        self.colors_sem = {cs_class.train_id: cs_class.color for cs_class in self.encoding}
        for train_id, color in sorted(self.colors_sem.items(), key=lambda item: item[0]):
            R, G, B = color
            self.palette.extend((R, G, B))

        zero_pad = 256 * 3 - len(self.palette)
        for i in range(zero_pad):
            self.palette.append(0)

    def png2depth(self, depth_png):
        depth_np = np.array(depth_png, dtype=np.float32)
        depth_np = depth_np[..., 0] + depth_np[..., 1]*256 + depth_np[..., 2]*256*256
        depth_np = depth_np/(256*256*256-1)
        depth_np = depth_np.clip(0.01, 1)
        if self.use_depth:
            depth_np = 1/(depth_np)
            # depth_np = 1 / depth_np
            # depth_np = (depth_np-depth_np.min())/(depth_np.max()-depth_np.min())
            # depth_np *= self.max_depth
        else:
            depth_np *= 100.0
            
        return np.array(depth_np, dtype=np.float32)

    def depth2color(self, depth_np):
        depth_np[depth_np>self.treshold] = self.treshold
        depth_np = (depth_np-depth_np.min())/(depth_np.max()-depth_np.min())
        indexes = np.array(depth_np*255, dtype=np.int32)
        color_depth = self.colors[indexes]
        return color_depth

    def re_normalize (self, image, mean, std):
        image = np.array(image, np.float32)
        image = image.transpose((1, 2, 0))
        image *= self.std
        image += self.mean
        # image = image[:, :, ::-1]
        return np.uint8(image*255)

    def colorize_mask(self, mask, encode_with_train_id=False):
        mask = np.array(mask, dtype=np.uint8)
        if encode_with_train_id:
            mask = self.encode_image_train_id(mask)
        new_mask = Image.fromarray(mask).convert('P')
        new_mask.putpalette(self.palette)

        return new_mask

    def __getitem__(self, index):
        # img = Image.open(self.images[index]).convert('RGB')         
        gt = Image.open(self.labels[index])
        gt_sem = Image.open(self.labels_semantic[index])

        # if self.transforms is not None:
        #     transformed = self.transforms(image=np.array(img), mask=np.array(gt))
        #     img = transformed['image']
        #     gt = transformed['mask']

        if self.size is not None:
            # img = img.resize(self.size, Image.LANCZOS)
            gt = gt.resize(self.size, Image.BILINEAR)    
            gt_sem = gt_sem.resize(self.label_size, Image.NEAREST)    

        gt = self.png2depth(gt)
        # img = np.asarray(img, np.float32)
        # img = img/255.0
        # img -= self.mean
        # img /= self.std      
        gt = torch.from_numpy(gt)

        gt_sem = np.array(gt_sem)
        gt_sem = torch.from_numpy(gt_sem).type(torch.long)
        # img = torch.from_numpy(img.transpose((2,0,1)))
        
        # rgbd = torch.cat([img, gt.unsqueeze(0)], dim=0)
        stacked_depth = torch.cat([gt.unsqueeze(0), gt.unsqueeze(0), gt.unsqueeze(0)], dim=0)

        # return rgbd, gt_sem
        return stacked_depth, gt_sem

    def __len__(self):
        return len(self.images)

    def get_predictions_plot_depth(self, batch_sample, predictions, batch_gt):
        num_images = batch_sample.size()[0]
        fig, m_axs = plt.subplots(3, num_images, figsize=(12, 10), squeeze=False)
        plt.subplots_adjust(hspace = 0.1, wspace = 0.1)

        for image, prediction, gt, (axis1, axis2, axis3) in zip(batch_sample, predictions, batch_gt, m_axs.T):
            
            image = self.re_normalize(image, self.mean, self.std)
            # image = to_pil_image(image)
            axis1.imshow(image)
            axis1.set_axis_off()

            prediction = self.depth2color(prediction.squeeze())
            axis2.imshow(prediction)
            axis2.set_axis_off()
            
            gt = self.depth2color(gt)
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
    
    def get_predictions_plot(self, batch_sample, predictions, batch_gt, encode_gt=False):

        num_images = min(batch_sample.size()[0], 4)
        fig, m_axs = plt.subplots(3, num_images, figsize=(12, 8), squeeze=False)
        plt.subplots_adjust(hspace = 0.1, wspace = 0.1)
        if predictions.dim() == 4:
            predictions = torch.argmax(predictions, dim=1)

        for image, prediction, gt, (axis1, axis2, axis3) in zip(batch_sample, predictions, batch_gt, m_axs.T):
            # image = self.re_normalize(image[:-1, :, :], self.mean, self.std)
            # image = to_pil_image(image)
            # axis1.imshow(image)

            image = self.depth2color(image[0])
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
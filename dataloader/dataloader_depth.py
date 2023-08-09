import os
import io
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
from collections import namedtuple


class DepthDataset(Dataset):
    def __init__(self, root, txt_file, transforms=None, 
                max_depth=1000, threshold=100,
                mean=[0.286, 0.325, 0.283], std=[0.176, 0.180, 0.177], use_depth=True, size=(1024, 512), label_size=(2048, 1024)):

        super(DepthDataset, self).__init__()
        self.files_txt = txt_file
        self.images = []
        self.labels = []
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
            self.labels.append(os.path.join(root, splits[2].strip()))

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
        depth_np = (depth_np-depth_np.min())/(depth_np.max()-depth_np.min()+1e-6)
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

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')         
        gt = Image.open(self.labels[index])
        # if self.transforms is not None:
        #     transformed = self.transforms(image=np.array(img), mask=np.array(gt))
        #     img = transformed['image']
        #     gt = transformed['mask']

        if self.size is not None:
            img = img.resize(self.size, Image.LANCZOS)
            gt = gt.resize(self.label_size, Image.BILINEAR)

        if self.transforms is not None:
            img, gt = self.transforms(img, gt)
        else:
            # img = to_tensor(img)     
            img = np.asarray(img, np.float32)
            # img = img[:, :, ::-1]
            img = img/255.0
            img -= self.mean
            img /= self.std      
            # gt = torch.from_numpy(gt)
            img = img.transpose((2,0,1)).copy()

        gt = self.png2depth(gt)
        return img, gt

    def __len__(self):
        return len(self.images)

    def get_predictions_plot(self, batch_sample, predictions, batch_gt):
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
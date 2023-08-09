#%%
import os
import io
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import namedtuple
#%%

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
        CityscapesClass('unknown',              255, 19, 'void', 7, True, False, (0, 0, 0)),
        CityscapesClass('license plate',        -1, 19, 'vehicle', 7, False, True, (0, 0, 0)),
    ]
#%%

id_to_trainId = {cs_class.id: cs_class.train_id for cs_class in cs}

# input_path = "/home/adricarda/projects/atdt/splits/gta1_full/train.txt"
root = "/media/data/datasets/gta1/semantic"

def encode_image_train_id(mask, id_to_trainId):
    mask = np.array(mask)
    mask_copy = mask.copy()
    for k, v in id_to_trainId.items():
        mask_copy[mask == k] = v
    return mask_copy

for line in tqdm(os.listdir(root)):
    # splits = line.split(';')
    path = os.path.join(root, line)
    dest_path = path.replace('.png', '_encoded.png')

    label = Image.open(path)
    label = encode_image_train_id(label, id_to_trainId)
    Image.fromarray(label).save(dest_path)

# %%

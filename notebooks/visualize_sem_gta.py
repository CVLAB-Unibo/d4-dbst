#%%
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import namedtuple


#%%

image_path = '/media/data/datasets/gta1/images/test/03120.png'
img = Image.open(image_path)
label_path = '/media/data/datasets/gta1/semantic/test/03120.png'
label = Image.open(label_path)
plt.imshow(img)

# %%
semantic_map = np.array(label)
print(semantic_map.max())
print(semantic_map.min())
plt.imshow(semantic_map)
# %%
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
        CityscapesClass('unknown',              255, 19, 'void', 7, True, False, (0, 0, 0)),
        CityscapesClass('license plate',        -1, 19, 'vehicle', 7, False, True, (0, 0, 0)),
    ]


# %%

id_to_trainId = {cs_class.id: cs_class.train_id for cs_class in cs}
mask_copy = semantic_map.copy()
for k, v in id_to_trainId.items():
    mask_copy[semantic_map == k] = v
# mask_copy[mask_copy==10] = 19

# %%
palette = []
colors = {cs_class.train_id: cs_class.color for cs_class in cs}
for train_id, color in sorted(colors.items(), key=lambda item: item[0]):
    R, G, B = color
    palette.extend((R, G, B))

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

# %%
test = Image.fromarray(mask_copy).convert('P')
test.putpalette(palette)
print(test.size)
plt.imshow(test)

# %%

label_path = '/media/data_4t/aCardace/datasets/synthia_cityscapes/RAND_CITYSCAPES/GT/semantic_encoded/0000050.png'
label = Image.open(label_path)
test = label.convert('P')
test.putpalette(palette)
print(test.size)
plt.imshow(test)

# %%
img_path = '/media/data_4t/aCardace/datasets/synthia_cityscapes/RAND_CITYSCAPES/RGB/0000300.png'
img = Image.open(img_path)
plt.imshow(img)
# %%

#%%
import os
import glob
import cv2
import numpy as np
from PIL import Image
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
import time

# %%
# n=0
# fout = open('/home/ldeluigi/dev/atdt-da2/splits/synthia_persefone/train_sky_filtered.txt', 'w')
# for f in tqdm(sorted(glob.glob("/data/lDeLuigi/datasets/SYNTHIA_RAND_CITYSCAPES/semantic_encoded/*"))):
#     mask = np.asarray(Image.open(f))
#     num_per_class = np.zeros((19))
#     for c in range(19):
#         num_per_class[c] = np.count_nonzero(mask==c)
#     indexes = np.argsort(num_per_class)
#     ys, xs = np.where(mask == 10)

#     if 0 in indexes[-3:] and len(ys)>0 and ys.max()>180:
#         fout.write('{};{};{}\n'.format(f.replace("semantic_encoded", "RGB"), f, f.replace("semantic_encoded", "Depth")))
# print(n)
# fout.close()


# %%
for f in tqdm(glob.glob("/data/lDeLuigi/datasets/SYNTHIA_RAND_CITYSCAPES/semantic_encoded/*")):
    mask = np.asarray(Image.open(f))
    num_per_class = np.zeros((19))
    for c in range(19):
        num_per_class[c] = np.count_nonzero(mask==c)
    indexes = np.argsort(num_per_class)
    ys, xs = np.where(mask == 10)

    if 0 in indexes[-3:] and len(ys)>0 and ys.max()>180:
        # n+=1
        path = f.replace("semantic_encoded", "RGB")
        img = Image.open(path)
        display.display(img)
        time.sleep(4)
        display.clear_output()
    # break
# %%

#%%

import os
from pathlib import Path
import shutil
#%%

fin = open("/media/data_4t/aCardace/atdt/splits/synthia_seq/train.txt")
rgb_path = Path("/media/data_4t/aCardace/datasets/synthia_sequences/3k_split/rgb")
sem_path = Path("/media/data_4t/aCardace/datasets/synthia_sequences/3k_split/sem")
rgb_path.mkdir(exist_ok=True, parents=True)
sem_path.mkdir(exist_ok=True, parents=True)

#%%

for index, line in enumerate(fin.readlines()):
    print(index)
    rgb, sem, _ = line.split(";")
    rgbp = "/media/data_4t/aCardace/datasets/" + rgb
    semp = "/media/data_4t/aCardace/datasets/" + sem
    shutil.copy(rgbp, str(rgb_path / (str(index) + ".png")))
    shutil.copy(semp, str(sem_path / (str(index) + ".png")))

# %%

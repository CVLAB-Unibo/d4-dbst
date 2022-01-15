import numpy as np
from matplotlib import cm
from matplotlib.colors import Colormap, ListedColormap

cmap19 = [
    (128, 64, 128),  # road
    (244, 35, 232),  # sidewalk
    (70, 70, 70),  # building
    (102, 102, 156),  # wall
    (190, 153, 153),  # fence
    (153, 153, 153),  # pole
    (250, 170, 30),  # traffic light
    (220, 220, 0),  # traffic sign
    (107, 142, 35),  # vegetation
    (152, 251, 152),  # terrain
    (70, 130, 180),  # sky
    (220, 20, 60),  # person
    (255, 0, 0),  # rider
    (0, 0, 142),  # car
    (0, 0, 70),  # truck
    (0, 60, 100),  # bus
    (0, 80, 100),  # train
    (0, 0, 230),  # motorcycle
    (119, 11, 32),  # bicycle
    (0, 0, 0),  # unlabelled
]


def get_cmap(name: str) -> Colormap:
    try:
        cmap = cm.get_cmap(name)
    except ValueError:
        cmaps = {"cmap19": cmap19}
        cmap = ListedColormap(np.array(cmaps[name], dtype=np.float32) / 255)

    return cmap

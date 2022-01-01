from typing import Dict

cs19: Dict[int, int] = {
    0: 19,  # unlabelled
    1: 19,  # ego vehicle
    2: 19,  # rectification border
    3: 19,  # out of roi
    4: 19,  # static
    5: 19,  # dynamic
    6: 19,  # ground
    7: 0,  # road
    8: 1,  # sidewalk
    9: 19,  # parking
    10: 19,  # rail track
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    14: 19,  # guard rail
    15: 19,  # bridge
    16: 19,  # tunnel
    17: 5,  # pole
    18: 19,  # polegroup
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 19,  # caravan
    30: 19,  # trailer
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
    34: 19,  # unknown
    -1: 19,  # license plate
}

gta19: Dict[int, int] = {}

cs12: Dict[int, int] = {
    0: 12,  # unlabelled
    1: 12,  # ego vehicle
    2: 12,  # rectification border
    3: 12,  # out of roi
    4: 12,  # static
    5: 12,  # dynamic
    6: 12,  # ground
    7: 2,  # road
    8: 3,  # sidewalk
    9: 12,  # parking
    10: 12,  # rail track
    11: 1,  # building
    12: 19,  # wall
    13: 4,  # fence
    14: 12,  # guard rail
    15: 12,  # bridge
    16: 12,  # tunnel
    17: 6,  # pole
    18: 12,  # polegroup
    19: 11,  # traffic light
    20: 8,  # traffic sign
    21: 5,  # vegetation
    22: 12,  # terrain
    23: 0,  # sky
    24: 9,  # person
    25: 10,  # rider
    26: 7,  # car
    27: 7,  # truck
    28: 7,  # bus
    29: 12,  # caravan
    30: 12,  # trailer
    31: 12,  # train
    32: 12,  # motorcycle
    33: 10,  # bicycle
    34: 12,  # unknown
    -1: 12,  # license plate
}

synthiaseq12: Dict[int, int] = {
    0: 12,  # unknown
    1: 0,  # sky
    2: 1,  # building
    3: 2,  # road
    4: 3,  # sidewalk
    5: 4,  # fence
    6: 5,  # vegetation
    7: 6,  # pole
    8: 7,  # car
    9: 8,  # traffic sign
    10: 9,  # person
    11: 10,  # bicycle
    12: 2,  # lanemarking
    13: 12,  # unknown2
    14: 12,  # unknown3
    15: 11,  # traffic light
}


def get_semantic_map(name: str) -> Dict[int, int]:
    maps = {"cs19": cs19, "cs12": cs12, "gta19": gta19, "synthiaseq12": synthiaseq12}
    return maps[name]

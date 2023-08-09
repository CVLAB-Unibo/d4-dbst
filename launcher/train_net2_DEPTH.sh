#!/bin/bash

python3 ../train_net2_DEPTH.py \
    --data_dir '/media/data_4t/aCardace/datasets/' \
    --model_dir '../gta2cs/DEPTH' \
    --checkpoint_dir '../gta2cs/DEPTH/ckpt' \
    --tensorboard_dir '../gta2cs/DEPTH/tensorboard' \
    --txt_train '../splits/gta1_full/train.txt' \
    --txt_val1 '../splits/gta1_full/val.txt' \
    --txt_val2 '../splits/cityscapes/val_gta.txt'
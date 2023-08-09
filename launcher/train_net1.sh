#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../train_net1.py \
    --data_dir '/media/data5/acardace/dataset/' \
    --model_dir '../gta2cs/net1' \
    --checkpoint_dir '../gta2cs/net1/ckpt' \
    --tensorboard_dir '../gta2cs/net1/tensorboard' \
    --txt_train '../splits/mixed_gta1_short_cs/train.txt' \
    --txt_val1 '../splits/gta1_full/val.txt' \
    --txt_val2 '../splits/cityscapes/val.txt'

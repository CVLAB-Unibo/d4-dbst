#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../train_net2.py \
    --data_dir '/media/data5/acardace/dataset/' \
    --model_dir '../gta2cs/net2_wc' \
    --checkpoint_dir '../gta2cs/net2_wc/ckpt' \
    --tensorboard_dir '../gta2cs/net2_wc/tensorboard' \
    --txt_train '../splits/gta1_short/train.txt' \
    --txt_val1 '../splits/gta1_short/val.txt' \
    --txt_val2 '../splits/cityscapes/val.txt'
#!/bin/bash

python3 ../train_net1_batch_alternated.py \
    --data_dir '/media/data_4t/aCardace/datasets/' \
    --model_dir '../synthia2cs/net1_r101_alternated' \
    --checkpoint_dir '../synthia2cs/net1_r101_alternated/ckpt' \
    --tensorboard_dir '../synthia2cs/net1_r101_alternated/tensorboard' \
    --txt_train1 '../splits/synthia/train.txt' \
    --txt_train2 '../splits/cityscapes/train.txt' \
    --txt_val1 '../splits/synthia/val_gt.txt' \
    --txt_val2 '../splits/cityscapes/val_gt.txt'
#!/bin/bash
sleep 75m
CUDA_VISIBLE_DEVICES=0 python ../train_transfer.py \
    --data_dir '/media/data5/acardace/dataset/' \
    --model_dir_source '../gta2cs/net1' \
    --checkpoint_dir_source '../gta2cs/net1/ckpt' \
    --model_dir_target '../gta2cs/net2_wc' \
    --checkpoint_dir_target '../gta2cs/net2_wc/ckpt' \
    --model_dir_transfer '../gta2cs/transfer_wc' \
    --checkpoint_dir_transfer '../gta2cs/transfer_wc/ckpt' \
    --tensorboard_dir "../gta2cs/transfer_wc/tensorboard" \
    --txt_train '../splits/gta1_short/train.txt' \
    --txt_val_source '../splits/gta1_short/val.txt' \
    --txt_val_target '../splits/cityscapes/val.txt'
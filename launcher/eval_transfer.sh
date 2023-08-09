#!/bin/bash

python3 ../evaluate_transfer.py \
    --data_dir '/media/data5/acardace/dataset/' \
    --model_dir_source '../gta2cs/net1_original_high_res' \
    --checkpoint_dir_source '../gta2cs/net1_original_high_res/ckpt' \
    --model_dir_target '../gta2cs/net2_r50_wc_strong' \
    --checkpoint_dir_target '../gta2cs/net2_r50_wc_strong/ckpt' \
    --model_dir_transfer '../gta2cs/transfer_net1_original_high_res2net2_r50_wc_strong_long' \
    --checkpoint_dir_transfer '../gta2cs/transfer_net1_original_high_res2net2_r50_wc_strong_long/ckpt' \
    --txt_val '../splits/cityscapes/val.txt'

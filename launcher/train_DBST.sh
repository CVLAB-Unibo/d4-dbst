#!/bin/bash
# sleep 7h

python3 ../train_DBST.py \
    --data_dir '/media/data3/atdt/' \
    --model_dir '../gta2cs/dbst_proda_full_2' \
    --checkpoint_dir '../gta2cs/dbst_proda_full_2/ckpt' \
    --tensorboard_dir '../gta2cs/dbst_proda_full_2/tensorboard' \
    --txt_train '../splits/cityscapes/train.txt' \
    --txt_val_target '../splits/cityscapes/val.txt' \
    --train_augmented_dir '/media/data3/atdt/CityScapes/train_proDA_full/'
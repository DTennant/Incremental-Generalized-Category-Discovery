#!/bin/bash

set -e
set -x

CUDA_VISIBLE_DEVICES=0 python -m model.simgcd_icarl \
    --dataset_name 'inat21_incremental' \
    --batch_size 512 \
    --grad_from_block 10 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.03 \
    --eval_funcs 'v2i' \
    --print_freq 1 \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 10 \
    --memax_weight 2 \
    --use_small_set \
    --split_crit 'loc_year' \
    --runner_name 'incremental' --return_path \
    --exp_name inat21_incre_simgcd_debug_icarl \
    --exp_id inat21_incre_simgcd_icarl1 \
    --tags 'incremental-inat21-debug-icarl'

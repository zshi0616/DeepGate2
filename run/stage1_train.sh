#!/bin/bash
NUM_PROC=2
GPUS=0,1

cd src
shift
python3 -m torch.distributed.launch --nproc_per_node=$NUM_PROC ./main.py prob \
 --exp_id train \
 --data_dir ../data/train \
 --reg_loss l1 --cls_loss bce \
 --arch mlpgnn \
 --Prob_weight 1 --RC_weight 0 --Func_weight 0 \
 --num_rounds 1 \
 --small_train \
 --gpus ${GPUS} --batch_size 16 \


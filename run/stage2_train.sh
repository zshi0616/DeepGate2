#!/bin/bash
NUM_PROC=1
GPUS=0

cd src
shift
python3 -m torch.distributed.run --nproc_per_node=$NUM_PROC ./main.py prob \
 --exp_id train \
 --data_dir ../data/train \
 --reg_loss l1 --cls_loss bce \
 --arch mlpgnn \
 --Prob_weight 3 --RC_weight 1 --Func_weight 2 \
 --num_rounds 1 \
 --gpus ${GPUS} --batch_size 16 \
 --resume \
 --no_rc


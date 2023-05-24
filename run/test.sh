#!/bin/bash
SPC_EXPID=train
DATASET=test
AGGR=tfmlp
NO_ROUNDS=1
TEST_SCRIPT=test_acc_bin

GPU=-1

cd src
python3 ${TEST_SCRIPT}.py prob --exp_id ${SPC_EXPID} --spc_exp_id ${SPC_EXPID} \
 --data_dir ../data/${DATASET} \
 --num_rounds ${NO_ROUNDS} \
 --reg_loss l1 --cls_loss bce \
 --arch mlpgnn \
 --aggr_function ${AGGR} \
 --gpu ${GPU} --batch_size 1 \
 --no_rc \
 --resume

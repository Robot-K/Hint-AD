#!/usr/bin/env bash

T=`date +%m%d%H%M`

# -------------------------------------------------- #
# Usually you only need to customize these variables #
CFG=$1                                               #
CKPT=$2                                              #
GPUS=$3                                              #    
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))
MASTER_PORT=${MASTER_PORT:-28595}
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
# Intermediate files and logs will be saved to UniAD/projects/work_dirs/

if [ ! -d ${WORK_DIR}logs ]; then
    mkdir -p ${WORK_DIR}logs
fi

export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/test.py \
    $CFG \
    $CKPT \
    --launcher pytorch ${@:4} \
    --eval bbox \
    --show-dir ${WORK_DIR} \
    --results /mnt/kding/Daimler/Hint-UniAD/test/base_caption/Thu_Sep_26_11_12_18_2024/results.pkl \
    2>&1 | tee ${WORK_DIR}logs/eval.$T
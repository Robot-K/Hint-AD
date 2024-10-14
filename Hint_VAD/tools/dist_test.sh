#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29504}
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox \
    --out projects/work_dirs/eval_tod_ablation_QA/output1.pkl --json_dir projects/work_dirs/eval_tod_ablation_QA --show

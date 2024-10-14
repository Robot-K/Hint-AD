# !/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28514}
current_dir=$(dirname "$0")

# export NCCL_IB_DISABLE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT\
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic --work-dir $(dirname "$current_dir")/projects/work_dirs/train_tod_ablationQA
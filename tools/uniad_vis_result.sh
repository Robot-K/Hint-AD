#!/bin/bash
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python ./tools/analysis_tools/visualize/run.py \
    --predroot test/base_caption/Mon_Jun_10_15_58_16_2024/results.pkl \
    --out_folder test/base_caption/Mon_Jun_10_15_58_16_2024/vis_output \
    --demo_video test/base_caption/Mon_Jun_10_15_58_16_2024/video/test_demo*.avi \
    --project_to_cam True
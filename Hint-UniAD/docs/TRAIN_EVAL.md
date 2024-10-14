# Train/Eval Models

## Evaluation Example <a name="example"></a>
Please make sure you have prepared the environment and the nuScenes dataset. You can check it by simply evaluating the pre-trained first-stage(track_map) model as follows:
```shell
cd HintAD
./tools/hintad_dist_eval.sh ./projects/configs/stage3_caption/base_caption.py ./ckpts/6_9_newcmdcount_epoch_3.pth 8

# For slurm users:
# ./tools/hintad_slurm_eval.sh YOUR_PARTITION ./projects/configs/stage3_caption/base_caption.py ./ckpts/6_9_newcmdcount_epoch_3.pth 8
```

## GPU Requirements <a name="gpu"></a>

Hint-UniAD initializes the weights trained from UniAD and optimizes all task modules together. It's recommended to use at least 8 GPUs for training in all stages. Training with fewer GPUs is fine but would cost more time.

The training takes ~ 70 GB GPU memory, ~ 6 hours for 1 epoch on 8 A100 GPUs. (remember to modify max_qa_num in config file to adjust member usage)

##  Train <a name="train"></a>

### Training Command
```shell
# N_GPUS is the number of GPUs used. Recommended >=8.
./tools/hintad_dist_train.sh ./projects/configs/stage3_caption/base_caption.py N_GPUS

# For slurm users:
# ./tools/hintad_slurm_train.sh YOUR_PARTITION ./projects/configs/stage3_caption/base_caption.py N_GPUS
```

## Evaluation <a name="eval"></a>

### Eval Command
```shell
# N_GPUS is the number of GPUs used.  Recommended =8.
# If you evaluate with different number of GPUs rather than 8, the results might be slightly different.

./tools/hintad_dist_eval.sh ./projects/configs/stage3_caption/base_caption.py /PATH/TO/YOUR/CKPT.pth N_GPUS

# For slurm users:
# ./tools/hintad_slurm_eval.sh YOUR_PARTITION ./projects/configs/stage3_caption/base_caption.py /PATH/TO/YOUR/CKPT.pth N_GPUS
```

## Visualization <a name="vis"></a>

### visualization Command


```shell
# please refer to  ./tools/hintad_vis_result.sh
./tools/hintad_vis_result.sh
```
---
<- Last Page: [Prepare The Dataset](./DATA_PREP.md)
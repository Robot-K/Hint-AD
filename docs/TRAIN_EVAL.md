# Train/Eval Models

## Evaluation Example <a name="example"></a>
Please make sure you have prepared the environment and the nuScenes dataset. You can check it by simply evaluating the pre-trained first-stage(track_map) model as follows:
```shell
cd UniAD
./tools/uniad_dist_eval.sh ./projects/configs/stage3_caption/base_caption.py ./ckpts/6_9_newcmdcount_epoch_3.pth 8

# For slurm users:
# ./tools/uniad_slurm_eval.sh YOUR_PARTITION ./projects/configs/stage3_caption/base_caption.py ./ckpts/6_9_newcmdcount_epoch_3.pth 8
```

## GPU Requirements <a name="gpu"></a>

IntCap is trained in three stages. The first stage is to train the perception modules (e.g., track and map), and the second stage initializes the weights trained from last stage and optimizes all task modules together. The third stage is to train caption module independently. It's recommended to use at least 8 GPUs for training in all stages. Training with fewer GPUs is fine but would cost more time.

The first-stage training takes ~ 50 GB GPU memory, ~ 2 days for 6 epochs on 8 A100 GPUs.
* **HINT**: To save GPU memory, you can change `queue_length=5` to `3` which will slightly degrade the tracking performance. Then the training would take ~30 GB GPU memory and is affordable for `V100 GPUs (32GB version)`.

The second-stage training takes ~ 17 GB GPU memory, ~ 4 days for 20 epochs on 8 A100 GPUs.
* **NOTE**: Compared to the first-stage, much less GPU memory is required because we freeze the BEV encoder in this stage to focus on learning task-specific queries. Due to this, you can run the second-stage training on `V100 or 3090` devices.

The third-stage training takes ~ 70 GB GPU memory, ~ 6 hours for 1 epoch on 8 A100 GPUs.

##  Train <a name="train"></a>

### Training Command
```shell
# N_GPUS is the number of GPUs used. Recommended >=8.
./tools/uniad_dist_train.sh ./projects/configs/stage3_caption/base_caption.py N_GPUS

# For slurm users:
# ./tools/uniad_slurm_train.sh YOUR_PARTITION ./projects/configs/stage3_caption/base_caption.py N_GPUS
```

## Evaluation <a name="eval"></a>

### Eval Command
```shell
# N_GPUS is the number of GPUs used.  Recommended =8.
# If you evaluate with different number of GPUs rather than 8, the results might be slightly different.

./tools/uniad_dist_eval.sh ./projects/configs/stage3_caption/base_caption.py /PATH/TO/YOUR/CKPT.pth N_GPUS

# For slurm users:
# ./tools/uniad_slurm_eval.sh YOUR_PARTITION ./projects/configs/stage3_caption/base_caption.py /PATH/TO/YOUR/CKPT.pth N_GPUS
```

## Visualization <a name="vis"></a>

### visualization Command


```shell
# please refer to  ./tools/uniad_vis_result.sh
./tools/uniad_vis_result.sh
```
---
<- Last Page: [Prepare The Dataset](./DATA_PREP.md)
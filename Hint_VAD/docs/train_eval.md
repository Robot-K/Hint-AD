# Train/Eval Models

## Prerequisites

**Please ensure you have prepared the environment and the dataset.**

## GPU Requirements <a name="gpu"></a>

Hint-VAD is trained in two stages. The first stage is to train the backbone VAD module and the second stage is to train caption module independently. It's recommended to use at least 8 GPUs for training in all stages. Training with fewer GPUs is fine but would cost more time.

In the first-stage, training takes ~ 50 GB GPU memory, ~ 20 hours for 15 epochs on 8 A100 GPUs.

In the second-stage, the memory usage ranges from 26GB (batch size = 1) to 45GB (batch size = 20). It takes 7 hours on 8 A100 GPUs.

## Evaluation Example <a name="example"></a>
Please make sure you have prepared the environment and the nuScenes dataset. You can check it by simply evaluating the pre-trained model as follows:
```shell
cd Hint-VAD
conda activate vad
./tools/dist_eval.sh ./projects/configs/VAD/VAD_caption.py ./ckpts/HintVAD.pth N_GPUS
# N_GPUS is the number of GPUs used. Recommended 8.
```

## Train Hint-VAD with 8 GPUs 
### Stage 1
Train the base VAD model as following. We provide checkpoint in ```./ckpts/BaseVAD.pth```.
```shell
./tools/dist_train.sh ./projects/configs/VAD/VAD_base_e2e.py N_GPUS
```

### Stage 2
Train the Hint-VAD model as following. We provide checkpoint in ```./ckpts/HintVAD.pth```.
```shell
./tools/dist_train.sh ./projects/configs/VAD/VAD_base_caption.py N_GPUS
```
**Note** If you trained base VAD by yourself, you should change ```load_from``` item in the config file ```./projects/configs/VAD/VAD_base_caption.py``` to your base VAD model checkpoint root. It is default to be ```ckpts/BaseVAD.pth```. 

## Evaluate Hint-VAD
```shell
.tools/dist_test.py projects/configs/VAD/VAD_base_caption.py /path/to/ckpt.pth N_GPUS
```
**Note** The result will be saved to ```projects/work_dirs/eval/output.pkl```. If you want to evaluate existing output.pkl add ```--result /path/to/your/output.pkl``` in ```dist_test.sh``` command.

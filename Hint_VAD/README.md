<div align="center">   
  
# Hint-AD: Holistically Aligned Interpretability in End-to-End Autonomous Driving
</div>
<br>

## Table of Contents:
1. [News](#news)
2. [Getting Started](#start)
   - [Installation](docs/install.md)
   - [Prepare Dataset](docs/prepare_dataset.md)
   - [GPU Requirements](docs/train_eval.md#gpu)
   - [Train and Eval](docs/train_eval.md)
3. [Modifications based on VAD](#mod)
4. [Results and Models](#models)

## News <a name="news"></a>
- **`Version 1 code release`**: Our implementation for Hint-AD firstly released for Daimler project, enjoy!

## Getting Started <a name="start"></a>
   - [Installation](docs/install.md)
   - [Prepare Dataset](docs/prepare_dataset.md)
   - [GPU Requirements](docs/train_eval.md#gpu)
   - [Train and Eval](docs/train_eval.md)

## Modifications based on VAD <a name="mod"></a>
Please view the following files for our key modifications:
```
Hint-VAD
├── ckpts
│   ├── BaseVAD.pth
│   ├── llama-2-7b/
│   ├── resnet50-19c8e357.pth
│   ├── HintVAD.pth
├── data
│   ├── infos
│   │   ├── nuscenes_infos_vad_train.pkl
│   │   ├── nuscenes_infos_vad_val.pkl
│   ├── nuscenes
│   │   ├── video_cap
│   │   │   ├── commands.json
│   │   │   ├── Nuscenes_X_train.json
│   │   │   ├── Nuscenes_X_val.json
│   │   ├── QA/
│   │   ├── tod3/
│   │   │   ├── bevcap-bevformer-trainval_infos_temporal_train.pkl
│   │   │   ├── bevcap-bevformer-trainval_infos_temporal_val.pkl
│   │   │   ├── final_caption_bbox_token.json
├── projects
│   ├── configs
│   │   ├── VAD
│   │   │   ├── VAD_base_caption.py
│   ├── mmdet3d_plugin
│   │   ├── llama
│   │   │   ├── llama_adapter
│   │   │   │   ├── llama_adapter.py
│   │   ├── VAD
│   │   │   ├── VAD_caption.py
│   │   │   ├── VAD.py
│   │   ├── datasets
│   │   │   ├── nuscenes_vad_dataset.py
│   │   │   ├── pipelines
│   │   │   │   ├── caption.py
```

## Results and Pre-trained Models <a name="models"></a>
IntCap is trained in two stages. Pretrained checkpoints of both stages are released and the results of caption model are listed in the following tables. (Other scores are also provided.)

| Method | Narration CIDEr | Reasoning CIDEr | tod3 CIDEr | Q&A hop0 Acc| Q&A hop1 Acc
| :---: | :---: | :---: | :---: | :---: | :---: |
| Hint-VAD | 0.2954 | 0.2309 | 2.9663 | 0.5577 | 0.4964

### Checkpoint Usage
* Download the checkpoints you need into `Hint-VAD/ckpts/` directory.
* You can evaluate these checkpoints to reproduce the results, following the `evaluation` section in [train_eval.md](docs/train_eval.md).
* You can also initialize your own model with the provided weights. Change the `load_from` field to `path/of/ckpt` in the config and follow the `train` section in [train_eval.md](docs/train_eval.md) to start training.

### Model Structure
The overall pipeline of Hint-AD is controlled by [VAD.py](projects/mmdet3d_plugin/VAD/VAD.py) which coordinates base VAD head and caption head. If you are interested in the implementation of a specific task module, please refer to its corresponding file, e.g., [caption_head](projects/mmdet3d_plugin/VAD/VAD_caption.py).

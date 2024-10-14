<div align="center">   
  
# Hint-AD: Holistically Aligned Interpretability in End-to-End Autonomous Driving
</div>

<br>

## Table of Contents:
1. [News](#news)
2. [Getting Started](#start)
   - [Installation](docs/INSTALL.md)
   - [Prepare Dataset](docs/DATA_PREP.md)
   - [Evaluation Example](docs/TRAIN_EVAL.md#example)
   - [GPU Requirements](docs/TRAIN_EVAL.md#gpu)
   - [Train/Eval](docs/TRAIN_EVAL.md)
3. [Modifications based on UniAD](#mod)
4. [Results and Models](#models)

## Getting Started <a name="start"></a>
- [Installation](docs/INSTALL.md)
- [Prepare Dataset](docs/DATA_PREP.md)
- [Evaluation Example](docs/TRAIN_EVAL.md#example)
- [GPU Requirements](docs/TRAIN_EVAL.md#gpu)
- [Train/Eval](docs/TRAIN_EVAL.md)

## Modifications based on UniAD <a name="mod"></a>
Please view the following files for our key modifications:
```
Hint-UniAD
├── projects/
│   ├── configs/
│   │   ├── stage3_caption
│   │   │   ├── base_caption.py
│   ├── mmdet3d_plugin
│   │   ├── llama
│   │   │   ├── llama_adapter
│   │   │   │   ├── llama_adapter.py
│   │   ├── uniad
│   │   │   ├── dense_heads
│   │   │   │   ├── caption_head.py
│   │   │   ├── detectors
│   │   │   │   ├── uniad_e2e.py
│   │   ├── datasets
│   │   │   ├── nuscenes_e2e_dataset.py
│   │   │   ├── pipelines
│   │   │   │   ├── caption.py
├── ckpts/
│   ├── llama-2-7b
│   ├── uniad_base_e2e.pth
│   ├── uniad_base_track_map.pth
│   ├── bevformer_r101_dcn_24ep.pth
├── data/
│   ├── nuscenes/
│   │   ├── video_cap/
│   │   │   ├── commands.json
│   │   │   ├── Nuscenes_X_train.json
│   │   │   ├── Nuscenes_X_val.json
│   │   ├── QA/
│   │   ├── tod3/
│   │   │   ├── bevcap-bevformer-trainval_infos_temporal_train.pkl
│   │   │   ├── bevcap-bevformer-trainval_infos_temporal_val.pkl
│   │   │   ├── final_caption_bbox_token.json
│   ├── infos/
│   │   ├── nuscenes_infos_capQAcmd_train.pkl
│   │   ├── nuscenes_infos_capQAcmd_val.pkl
```

## Results and Pre-trained Models <a name="models"></a>

> Based on pre-trained UniAD model, caption head is trained with a pretrained LLaMA-adapter-v2 model

| Method | config | path |
| :---: | :---: | :---: |
| UniAD | [base-stage3](projects/configs/stage3_caption/base_caption.py) | ./ckpts/uniad_base_e2e.pth |
| LLaMA | [base-stage3](projects/configs/stage3_caption/base_caption.py) | ./ckpts/llama-2-7b |

### Model Structure
The overall pipeline of Hint-AD is controlled by [uniad_e2e.py](projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py) which coordinates all the task modules in `Hint-AD/projects/mmdet3d_plugin/uniad/dense_heads`. If you are interested in the implementation of a specific task module, please refer to its corresponding file, e.g., [caption_head](projects/mmdet3d_plugin/uniad/dense_heads/caption_head.py).


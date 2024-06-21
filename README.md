<div align="center">   
  
# Hint-AD: Holistically Aligned Interpretability in End-to-End Autonomous Driving
</div>

<!-- <h3 align="center">
  <a href="https://arxiv.org/abs/2212.10156">arXiv</a> |
  <a href="https://www.youtube.com/watch?v=cyrxJJ_nnaQ">Video</a> |
  <a href="https://opendrivelab.com/e2ead/UniAD_plenary_talk_slides.pdf">Slides</a>
</h3>
 -->

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
<!-- 7. [Citation](#citation) -->

## News <a name="news"></a>

- **`Version 1 code release`**: Our implementation for IntCap firstly released for Daimler project, enjoy!

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
│   ├── 6_9_newcmdcount_epoch_3.pth
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
│   │   ├── DriveLM/
│   │   ├── tod3/
│   ├── infos/
│   │   ├── nuscenes_infos_capQAcmd_train.pkl
│   │   ├── nuscenes_infos_capQAcmd_val.pkl
```

## Results and Pre-trained Models <a name="models"></a>

> Based on pre-trained UniAD model, caption head is trained with a pretrained LLaMA-adapter-v2 model

| Method | config | path |
| :---: | :---: | :---: |
| Hint-UniAD | [base-stage3](projects/configs/stage3_caption/base_caption.py) | ./ckpts/6_9_newcmdcount_epoch_3.pth |
| UniAD | [base-stage3](projects/configs/stage3_caption/base_caption.py) | ./ckpts/uniad_base_e2e.pth |
| Hint-UniAD | [base-stage3](projects/configs/stage3_caption/base_caption.py) | ./ckpts/llama-2-7b |

### Checkpoint Usage
* Download the checkpoints you need into `UniAD/ckpts/` directory.
* You can evaluate these checkpoints to reproduce the results, following the `evaluation` section in [TRAIN_EVAL.md](docs/TRAIN_EVAL.md).
* You can also initialize your own model with the provided weights. Change the `load_from` field to `path/of/ckpt` in the config and follow the `train` section in [TRAIN_EVAL.md](docs/TRAIN_EVAL.md) to start training.


### Model Structure
The overall pipeline of UniAD is controlled by [uniad_e2e.py](projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py) which coordinates all the task modules in `UniAD/projects/mmdet3d_plugin/uniad/dense_heads`. If you are interested in the implementation of a specific task module, please refer to its corresponding file, e.g., [caption_head](projects/mmdet3d_plugin/uniad/dense_heads/caption_head.py).

<!-- ## Citation <a name="citation"></a>

Please consider citing our paper if the project helps your research with the following BibTex:

```bibtex

``` -->

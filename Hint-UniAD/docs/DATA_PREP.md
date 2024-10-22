
# Dataset

# nuScenes
Download nuScenes V1.0 full dataset data, CAN bus and map(v1.3) extensions [HERE](https://www.nuscenes.org/download), then follow the steps below to prepare the data.


**Download nuScenes, CAN_bus and Map extensions**
```shell
cd Hint-AD
mkdir data
# Download nuScenes V1.0 full dataset data directly to (or soft link to) Hint-AD/data/
# Download CAN_bus and Map(v1.3) extensions directly to (or soft link to) Hint-AD/data/nuscenes/
```

**Prepare Hint-AD data info**

```shell
cd Hint-AD/data
mkdir infos && cd infos
wget https://github.com/Robot-K/Hint-AD/releases/download/v1.0/nuscenes_infos_capQAcmdDriveLMtod3_train.pkl  # train_infos
wget https://github.com/Robot-K/Hint-AD/releases/download/v1.0/nuscenes_infos_capQAcmdDriveLMtod3_val.pkl  # val_infos
```

> You can also create info files by yourself through running ./tools/hintad_create_data.sh and .py files in data_converter to add annotations of Nu-X, NuScenes-QA, and TOD3Cap.

**Prepare Motion Anchors**
```shell
cd Hint-AD/data
mkdir others && cd others
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl
```

**Download Nu-X dataset from release**
```shell
mkdir -p Hint-AD/data/nuscenes/video_cap
cd Hint-AD/data/nuscenes/video_cap
wget https://github.com/Robot-K/Hint-AD/releases/download/v1.0/Nu_X_train.json
wget https://github.com/Robot-K/Hint-AD/releases/download/v1.0/Nu_X_val.json
```

**Download TOD3 dataset from release**
```shell
mkdir -p Hint-AD/data/nuscenes/tod3
cd Hint-AD/data/nuscenes/tod3
wget https://github.com/Robot-K/Hint-AD/releases/download/v1.0/bevcap-bevformer-trainval_infos_temporal_train.pkl
wget https://github.com/Robot-K/Hint-AD/releases/download/v1.0/bevcap-bevformer-trainval_infos_temporal_val.pkl
wget https://github.com/Robot-K/Hint-AD/releases/download/v1.0/final_caption_bbox_token.json
```

**The Overall Structure**

*Please make sure the structure of Hint-AD is as follows:*
```
Hint-AD
├── data/
│   ├── infos/
│   │   ├── nuscenes_infos_capQAcmd_train.pkl
│   │   ├── nuscenes_infos_capQAcmd_val.pkl├── data/
│   ├── nuscenes/
│   │   ├── video_cap/
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
│   ├── others/
│   │   ├── motion_anchor_infos_mode6.pkl
```
---

<- Last Page:  [Installation](./INSTALL.md)

-> Next Page: [Train/Eval Hint-AD](./TRAIN_EVAL.md)
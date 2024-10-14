
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

**The Overall Structure**

*Please make sure the structure of Hint-AD is as follows:*
```
Hint-AD
├── data/
│   ├── nuscenes/
│   │   ├── video_cap/
│   │   │   ├── commands.json
│   │   │   ├── Nuscenes_X_train.json
│   │   │   ├── Nuscenes_X_val.json
│   │   ├── QA/
│   │   ├── tod3/
│   ├── infos/
│   │   ├── nuscenes_infos_capQAcmd_train.pkl
│   │   ├── nuscenes_infos_capQAcmd_val.pkl├── data/
│   ├── nuscenes/
│   │   ├── video_cap/
│   │   │   ├── commands.json
│   │   │   ├── Nuscenes_X_train.json
│   │   │   ├── Nuscenes_X_val.json
│   │   ├── QA/
│   │   ├── tod3/
│   ├── infos/
│   │   ├── nuscenes_infos_capQAcmd_train.pkl
│   │   ├── nuscenes_infos_capQAcmd_val.pkl
│   ├── others/
│   │   ├── motion_anchor_infos_mode6.pkl
```
---
<- Last Page:  [Installation](./INSTALL.md)

-> Next Page: [Train/Eval Hint-AD](./TRAIN_EVAL.md)
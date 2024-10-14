# Dataset

## Captions

For this release, we've already provided necessary datasets including Nuscenes, Drive-LM, Nuscenes-QA, Nu-X, Command and TOD3. Please use ./data/infos/nuscenes_infos_vad_train.pkl and ./data/infos/nuscenes_infos_vad_val.pkl directly for deployment test.

## NuScenes
We use the same NuScenes datasets as Hint-UniAD, so you can just make a soft link to ```Hint-UniAD/data/nuscenes```.
```sh
mkdir -p "data/nuscenes"
ls -s ../Hint-UniAD/data/nuscenes data/nuscenes
```

## Checkpoint
We provide pre-trained models BaseVAD and HintVAD in ```./ckpts```. We use the same llama datasets as Hint-UniAD, so you can just make a soft link to ```Hint-UniAD/data/nuscenes```.
```sh
mkdir -p "ckpts/llama-2-7b"
ls -s ../Hint-UniAD/ckpts/llama-2-7b ckpts/llama-2-7b
```

## Folder structure
```
VAD
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── BaseVAD.pth
│   ├── llama-2-7b
│   ├── resnet50-19c8e357.pth
│   ├── HintVAD.pth
├── data/
│   ├── infos/
│   │   ├── nuscenes_infos_vad_train.pkl
│   │   ├── nuscenes_infos_vad_val.pkl
│   ├── nuscenes/
│   │   ├── basemap/
│   │   ├── can_bus/
│   │   ├── command/
│   │   ├── DriveLM/
│   │   ├── lidarseg/
│   │   ├── map_anns/
│   │   ├── maps/
│   │   ├── prediction/
│   │   ├── QA/
│   │   ├── samples/
│   │   ├── tod3/
│   │   ├── video/
│   │   ├── video_cap/
```

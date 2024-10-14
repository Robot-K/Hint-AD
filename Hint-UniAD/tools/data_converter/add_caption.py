import os
import subprocess
import numpy as np
import pickle
import json

mode = 'train'
pkl_path = f'/mnt/kding/Daimler/UniAD/data/infos/nuscenes_infos_temporal_{mode}.pkl'
output_path = f'/mnt/kding/Daimler/UniAD/data/infos/nuscenes_infos_cap_{mode}.pkl'
input_json = f'/mnt/kding/Daimler/UniAD/data/nuscenes/video_cap/Nu_X_{mode}.json'

with open(input_json, 'r') as f:
    caption_list = json.load(f)

with open(pkl_path, 'rb') as f:
    nuscenes_infos = pickle.load(f)

for item in nuscenes_infos['infos']:
    item['caption'] = caption_list[item['token']]

with open(output_path, 'wb') as f:
    pickle.dump(nuscenes_infos, f)
import os
import subprocess
import numpy as np
import pickle
import json

mode = 'test'
video_dir = 'data/nuscenes/video'
pkl_path = f'data/nuscenes/infos/vad_nuscenes_infos_temporal_{mode}.pkl'
output_path = f'data/nuscenes/infos/nuscenes_infos_cap_{mode}.pkl'
output_json = f'data/nuscenes/video_cap/Nuscenes_X_{mode}.json'
cap_file = 'data/nuscenes/video_cap/total_output.txt'

# make video-index to scene-token dictionary
video_list = os.listdir(video_dir)
token_dict = {}
for video in video_list:
    if video.endswith('.mp4'):
        scene_token = video.split('_')[-1].split('.')[0]
        index = str(int(video.split('_')[0]))
        token_dict[index] = scene_token

# print(token_dict)

def process_annotations(file_path):
    result = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_video = None
        current_time = None
        current_narration = None
        for line in lines:
            try:
                line = line.strip()
                if not line:
                    current_video = None
                    current_time = None
                    current_narration = None
                elif current_video is None:
                        current_video = token_dict[line] 
                        result[current_video] = []
                        current_time = 0
                elif current_time is None:
                    current_time = int(line)
                elif current_narration is None:
                    current_narration = line
                else:
                    result[current_video].append((current_time, current_narration, line))
                    current_time = None
                    current_narration = None
            except:
                print(f"Error processing line: {line}")
            
    return result

cap_dict = process_annotations(cap_file)
# print(cap_dict)

with open(pkl_path, 'rb') as f:
    nuscenes_infos = pickle.load(f)

prev_scene_token = ''
start_timestamp = 0
caption_json = {}
with open(output_path, 'wb') as f:
    for info in nuscenes_infos['infos']:
        if info['scene_token'] != prev_scene_token:
            if info['scene_token'] in cap_dict.keys():
                prev_scene_token = info['scene_token']
                start_timestamp = info['timestamp']
            else:
                print(f"Scene token {info['scene_token']} not found in cap_dict")
                continue
        
        time = int((info['timestamp'] - start_timestamp) / 1e6)
        if time < 0:
            print(f"Negative time: {time}")
            continue
        else:
            for cap_info in cap_dict[info['scene_token']][::-1]:
                if time >= cap_info[0]:
                    narration = cap_info[1]
                    reasoning = cap_info[2]
                    break
            info['caption'] = {'narration': narration, 'reasoning': reasoning}
            caption_json[info['scene_token']] = {'narration': narration, 'reasoning': reasoning}
    pickle.dump(nuscenes_infos, f)
    
with open(output_json, 'w') as json_file:
    json.dump(caption_json, json_file)
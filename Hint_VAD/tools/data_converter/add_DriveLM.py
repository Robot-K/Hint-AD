import json
import pickle

mode = 'val'
later_fix = 'v1_1_val_nus_q_only' if mode == 'val' else 'v1_1_train_nus'
pkl_path = f'data/infos/nuscenes_infos_capQAcmd_{mode}.pkl'
with open(pkl_path, 'rb') as f:
    nuscenes_infos = pickle.load(f)

with open(f'data/nuscenes/DriveLM/{later_fix}.json') as f: # v1_1_val_nus_q_only
    data = json.load(f)
    
def substitute(text, pair):
    for original, replacement in pair.items():
        text = text.replace(original, replacement)
    return text

key_frame_count = 0
miss_scene_count = 0
scenes = set()
miss_scenes = set()
QA_count = 0
for i, info in enumerate(nuscenes_infos['infos']):
    if info['scene_token'] not in scenes:
        print(QA_count)
        scenes.add(info['scene_token'])
        QA_count = 0
    if info['scene_token'] not in data.keys():
        if info['scene_token'] not in scenes:
            print(info['scene_token'], ' not in data.keys()')
            miss_scene_count += 1
            miss_scenes.add(info['scene_token'])
        continue
    scene = data[info['scene_token']]
    if info['token'] not in scene["key_frames"].keys():
        continue
    frame = scene["key_frames"][info['token']]
    # objects = frame["key_object_infos"]
    # substitute_pair = {}
    # for key in objects.keys():
    #     substitute_pair[key] = objects[key]['Visual_description'][:-1]
    
    QA = []
    idx = 0
    for part in ['perception', 'prediction', 'planning', 'behavior']:
        for QA_pair in frame['QA'][part]:
            # if 'What are the important objects in the current scene' in QA_pair['Q']:
            #     # the followings are perceptions of the objects, which are not suitable.
            #     break
            # if 'What object should the ego vehicle notice first when the ego vehicle is getting to the next possible location?' in QA_pair['Q']:
            #     # this question is not suitable.
            #     continue
            # Q = substitute(QA_pair['Q'], substitute_pair)
            # A = substitute(QA_pair['A'], substitute_pair)
            Q = QA_pair['Q']
            A = QA_pair['A']
            qa_idx = info['scene_token'] + "_" + info['token'] + "_" + str(idx)
            QA.append([Q, A, qa_idx])
            idx += 1
    QA_count += len(QA)
    key_frame_count += 1
    nuscenes_infos['infos'][i]['DriveLM'] = QA

print(key_frame_count)
print(miss_scene_count)
print(len(scenes))
print(len(miss_scenes))
with open(f'data/infos/nuscenes_infos_capQAcmdDriveLM_{mode}.pkl', 'wb') as file:
    pickle.dump(nuscenes_infos, file)

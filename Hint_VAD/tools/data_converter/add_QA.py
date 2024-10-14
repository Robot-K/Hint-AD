import json
import pickle

mode = 'val'
pkl_path = f'data/infos/nuscenes_infos_cap_{mode}.pkl'
QA_path = f'data/nuscenes/QA/NuScenes_{mode}_questions.json'
output_path = f'data/infos/nuscenes_infos_capQA_{mode}.pkl'
    
# load nuscenes-QA dataset
with open(QA_path) as f:
    data = json.load(f)
    
QA_dict = {}
for item in data['questions']:
    if item['sample_token'] not in QA_dict:
        QA_dict[item['sample_token']] = [{'question': item['question'], 'answer': item['answer'],
                                          'num_hop': item['num_hop'], 'template_type': item['template_type']}]
    else:
        QA_dict[item['sample_token']].append({'question': item['question'], 'answer': item['answer'],
                                          'num_hop': item['num_hop'], 'template_type': item['template_type']})

# add QA to pkl
with open(pkl_path, 'rb') as f:
    nuscenes_infos = pickle.load(f)

error_count = 0
with open(output_path, 'wb') as f:
    for info in nuscenes_infos['infos']:
        if info['token'] not in QA_dict.keys():
            info['QA'] = {}
            error_count += 1
        else:
            info['QA'] = QA_dict[info['token']]
    pickle.dump(nuscenes_infos, f)
print(f'There are {error_count} frames that do not have QA. {len(QA_dict.keys())} frames in total.')
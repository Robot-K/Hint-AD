import json
import pickle

mode = 'train'
pkl_path = f'data/infos/nuscenes_infos_capQAcmdDriveLM_{mode}.pkl'
tod3_path = f'data/nuscenes/tod3/bevcap-bevformer-trainval_infos_temporal_{mode}.pkl'
output_path = f'data/infos/nuscenes_infos_capQAcmdDriveLMtod3_{mode}.pkl'

with open(tod3_path, 'rb') as f:
    tod3 = pickle.load(f)

with open(pkl_path, 'rb') as f:
    nuscenes_infos = pickle.load(f)

for info, tod in zip(nuscenes_infos['infos'], tod3['infos']):
    if len(info['gt_boxes']) != len(tod['gt_captions']):
        print(info['token'], ' not match')
    # elif len(info['gt_boxes']) != len(info['gt_inds']):
    #     print(info['token'], ' not match')
    else:
        info['tod3_tokens'] = tod['gt_captions']
    
with open(output_path, 'wb') as f:
    pickle.dump(nuscenes_infos, f)
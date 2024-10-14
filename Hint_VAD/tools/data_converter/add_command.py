import json
import pickle

mode = 'val'
pkl_path = f'data/infos/nuscenes_infos_capQAcmdDriveLMtod3_{mode}.pkl'
command_path = 'data/nuscenes/command/command.json'
output_path = f'data/infos/nuscenes_infos_0605_{mode}.pkl'

with open(command_path, 'r') as f:
    command_dict = json.load(f)
            
with open(pkl_path, 'rb') as f:
    nuscenes_infos = pickle.load(f)

not_found_count = 0
for info in nuscenes_infos['infos']:
    if info['token'] in command_dict.keys():
        info['customized_command'] = command_dict[info['token']]['traj_command'] + ' and ' + command_dict[info['token']]['speed_command']
    else:
        not_found_count += 1
        print(f"{info['token']} not found! ", not_found_count)
    
with open(output_path, 'wb') as f:
    pickle.dump(nuscenes_infos, f)
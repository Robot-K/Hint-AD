import numpy as np
import mmcv
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES

import torch
import copy
from projects.mmdet3d_plugin.llama.llama_adapter import utils
from projects.mmdet3d_plugin.llama.llama_adapter.tokenizer import Tokenizer
import random

@PIPELINES.register_module()
class LoadCaption(object):
    def __init__(self, tokenizer_path, caption_anno_path):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.caplist = mmcv.load(caption_anno_path) # modi

    def __call__(self, input_dict): # modi
        
        input1_list = []
        answer_list = []
        ins_list = []
        tod3_centers = []

        if 'DriveLM' in input_dict:
            input_dict_DriveLM = input_dict['DriveLM']
            for pair in input_dict_DriveLM:
                answer = pair[1]
                instruction = f"<DriveLM> Information of bird's eye view, tracks, trajectories and planning are provided, answer the following question: "
                # print("DriveLM_idx: ", pair[2]) # debug
                instruction = instruction + pair[0]
                instruction = utils.format_prompt(instruction, None)
                input1_list.append(instruction)
                answer_list.append(answer)
                ins_list.append(-1)

        if 'tod3_tokens' in input_dict: # modi
            input_dict_tod3 = input_dict['tod3_tokens']
            for index, token in enumerate(input_dict_tod3):
                instruction = "<tod3> Describe the given object in the format of '<attribute> about <distance> meters away <localization> is <motion> <map>', information of bird's eye view, tracks, trajectories and planning are provided."
                instruction = utils.format_prompt(instruction, None)
                # print('depth: ', np.sum(np.array(tod3_centers[index])**2)**0.5, str(round(self.caplist[token]['depth_caption']['depth']))) # for debug
                try:
                    answer = self.caplist[token]['attribute_caption']['attribute_caption'] + \
                            " about " + str(round(self.caplist[token]['depth_caption']['depth'])) + " meters away " + \
                            self.caplist[token]['localization_caption']['localization_caption'] + " is " + \
                            self.caplist[token]['motion_caption']['motion_caption'] + " " + \
                            self.caplist[token]['map_caption']['map_caption']
                    input1_list.append(instruction)
                    answer_list.append(answer)
                    ins_list.append(0)
                    tod3_centers.append(input_dict['tod3_centers'][index])
                except:
                    # print(token, 'not exist!')
                    pass
            if len(tod3_centers) > 0:
                tod3_centers = torch.tensor(tod3_centers)[:, :7]
            else:
                tod3_centers = torch.tensor(tod3_centers)
        
        if 'caption' in input_dict:
            gt_captions = input_dict['caption']
            answer = 'Narration: ' + gt_captions['narration'] + '. Reasoning: ' + gt_captions['reasoning'] + '.'
            instruction = "<caption> Describe the behavior of the ego car and the reason behind it. Information of bird's eye view, tracks, trajectories and planning are provided.\n Example:\n Narration: the car is merging into the left lane. Reasoning: because the lane is moving faster."
            instruction = utils.format_prompt(instruction, None)
            input1_list.append(instruction)
            answer_list.append(answer)
            ins_list.append(-2)
            
        if 'customized_command' in input_dict:
            instruction = "<command> Information of bird's eye view, tracks, trajectories and planning are provided. Please predict the direction and velocity control signal of the ego car."
            instruction = utils.format_prompt(instruction, None)
            input1_list.append(instruction)
            answer_list.append(input_dict['customized_command'])
            ins_list.append(-3)
            
        if 'QA' in input_dict:
            input_dict_qa = input_dict['QA']
            for pair_dict in input_dict_qa:
                answer = pair_dict['answer']
                instruction = f"<QA><{pair_dict['num_hop']}> Information of the car's bird's eye view video frames, tracks, and trajectories are optionally provided, answer the following question: "
                instruction = instruction + pair_dict['question']
                instruction = utils.format_prompt(instruction, None)
                input1_list.append(instruction)
                answer_list.append(answer)
                ins_list.append(-4)

        input2_list = [input1_list[i] + answer_list[i] for i in range(len(input1_list))]

        max_words = 220
        
        input2_cat = None
        labels_cat = None
        input2_mask_cat = None
        for i in range(len(input1_list)):
            input1 = torch.tensor(self.tokenizer.encode(input1_list[i], bos=True, eos=False), dtype=torch.int64)
            input2 = torch.tensor(self.tokenizer.encode(input2_list[i], bos=True, eos=True), dtype=torch.int64)
            padding = max_words - input2.shape[0]
            if padding > 0:
                input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                input2 = input2[:max_words]
            labels = copy.deepcopy(input2)
            labels[:len(input1)] = -1
            input2_mask = input2.ge(0)
            label_mask = labels.ge(0)
            input2[~input2_mask] = 0
            labels[~label_mask] = 0
            input2_mask = input2_mask.float()
            label_mask = label_mask.float()
            if input2_cat is None:
                input2_cat = input2.unsqueeze(0)
                labels_cat = labels.unsqueeze(0)
                input2_mask_cat = input2_mask.unsqueeze(0)
            else:
                input2_cat = torch.cat((input2_cat, input2.unsqueeze(0)))
                labels_cat = torch.cat((labels_cat, labels.unsqueeze(0)))
                input2_mask_cat = torch.cat((input2_mask_cat, input2_mask.unsqueeze(0)))

        ins_list = torch.tensor(ins_list, dtype=torch.int)

        input_dict.update({
            "gt_caption_token": input2_cat, # [Ins, Ans, 0]
            "gt_caplabel": labels_cat, # [0, Ans, 0]
            "gt_capmask": input2_mask_cat, # [1, 1, 0]
            "ins_list": ins_list, # indicate the caption type, modi
            "tod3_centers": tod3_centers, # [num_object, 7] modi
        })

        return input_dict
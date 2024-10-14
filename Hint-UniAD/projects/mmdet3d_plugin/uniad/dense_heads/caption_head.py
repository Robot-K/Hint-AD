# created by modified

import torch
from copy import deepcopy
import numpy as np
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
import mmcv
import pickle
import torch.nn as nn
from mmdet.models import  build_loss
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from projects.mmdet3d_plugin.models.utils.functional import pos2posemb2d
from projects.mmdet3d_plugin.llama.llama_adapter.llama_adapter import LLaMA_adapter
from projects.mmdet3d_plugin.llama.llama_adapter.tokenizer import Tokenizer
# from projects.mmdet3d_plugin.adapt.adapt import ADAPT
from projects.mmdet3d_plugin.llama.llama_adapter import utils
import time
import math
import re
from .caption_head_plugin import *

@HEADS.register_module()
class CaptionHead(nn.Module):
    def __init__(self,
                 *args,
                 llama_ckpt_root=None,
                 llama_tokenzier_root=None,
                 query_dim=256,
                 bev_proj_dim=64,
                 phase='pretrain',
                 captioner_type='llama',
                 train_cfg=None,
                 test_cfg=None,
                 max_qa_num = 7,
                 tod3_ratio = 0.4,
                 caption_ratio = 0.7,
                 iou_threshold = 0.25,
                 tokenizer_path=None,
                 caption_anno_path=None, # modi
                 **kwargs):
        super(CaptionHead, self).__init__()
        
        if captioner_type == 'llama':
            self.llama_ckpt_root = llama_ckpt_root
            self.llama_tokenzier_root = llama_tokenzier_root
            self.phase = phase
            assert llama_ckpt_root is not None and llama_tokenzier_root is not None
            self.llama_adapter = LLaMA_adapter(llama_ckpt_root, llama_tokenzier_root, phase=phase)
        elif captioner_type == 'adapt':
            self.adapt = ADAPT()
        self.query_dim = query_dim
        self.bev_proj_dim = bev_proj_dim
        self.max_qa_num = max_qa_num[0]
        self.tod3_ratio = tod3_ratio
        
        self.multi_query_fuser =  nn.Sequential(
                nn.Linear(query_dim * 2, query_dim * 2),
                nn.LayerNorm(query_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(query_dim * 2, query_dim),
            )
                
        self.planning_query_embedding_layer = nn.Sequential(
            nn.Linear(self.query_dim, self.query_dim * 2),
            nn.ReLU(),
            nn.Linear(self.query_dim * 2, self.query_dim),
        )
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.captioner_type = captioner_type
        self.CLASS = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        self.CLASSES = ['cars', 'trucks', 'construction_vehicles', 'buses', 'trailers', 'barriers', 'motorcycles', 'bicycles', 'pedestrians', 'traffic_cones']
        self.valid_CLASS = [0,1,2,3,7,8]
        self.tokenizer_path = tokenizer_path
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.caption_ratio = caption_ratio
        self.iou_threshold = iou_threshold

    def sample_tensor(self, tensor, n):
        assert n <= tensor.size(0), "n cannot be greater than the number of elements in the tensor"
        permuted_indices = torch.randperm(tensor.size(0))
        selected_indices = permuted_indices[:n]
        return tensor[selected_indices]

    def merge_queries(self, outs_track, outs_motion, outs_planning, mask_dict=None):
        track_query = outs_track['track_query_embeddings'].unsqueeze(0)              # [b, nq, d]
        traj_query = outs_motion['traj_query']       # [n_dec, b, nq, n_mode, d]
        sdc_traj_query = outs_motion['sdc_traj_query']          # [n_dec, b, n_mode, d]
        sdc_track_query = outs_motion['sdc_track_query']        # [b, d]
        traj_query = traj_query[-1].mean(2) # [b, nq, d]
        sdc_traj_query = sdc_traj_query[-1].mean(1) # [b, d]
        vehicle_mask = outs_motion['vehicle_mask']
        patched_traj_query = torch.zeros_like(track_query)
        if torch.any(vehicle_mask>0):
            patched_traj_query[:, vehicle_mask>0] = traj_query
        traj_query = patched_traj_query

        ins_query = self.multi_query_fuser(torch.cat([traj_query, track_query], dim=-1))  # [b, nq, d]
        sdc_query = self.multi_query_fuser(torch.cat([sdc_traj_query, sdc_track_query], dim=-1))  # [b, d]
        planning_query = self.planning_query_embedding_layer(pos2posemb2d(outs_planning['sdc_traj_all'][0, :, :2])).max(0)[0].view(-1, self.query_dim)  # [b, d]
        
        if mask_dict['ins_masked']:
            ins_query = None
        if mask_dict['sdc_masked']:
            sdc_query = None
        if mask_dict['planning_masked']:
            planning_query = None
        
        return ins_query, sdc_query, planning_query

    def match_centers(self, detected_centers, tod3_centers): # modi
        if len(tod3_centers) == 0:
            return torch.zeros((len(detected_centers),), device=detected_centers.device).long()
        matched_idx = torch.zeros((len(detected_centers), ), device=detected_centers.device)
        tod3_centers[:, 2] = tod3_centers[:, 2] - tod3_centers[:, 5] * 0.5
        iou_matrix = get_3d_iou(detected_centers, tod3_centers)
        for i in range(len(detected_centers)):
            # distances = torch.sum((detected_centers[i].repeat(len(tod3_centers), 1) - tod3_centers) ** 2, dim=1)
            # if torch.min(distances, dim=0)[0] > 3:
            #     matched_idx[i] = -1
            # else:
            #     matched_idx[i] = torch.min(distances, dim=0)[1] # modi2
            if torch.max(iou_matrix[i], dim=0)[0].item() > self.iou_threshold:
                matched_idx[i] = torch.max(iou_matrix[i], dim=0)[1]
            else:
                matched_idx[i] = -1

        ## for debug
        # detected_centers_np = detected_centers.detach().cpu().numpy()
        # tod3_centers_np = tod3_centers.detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.scatter(detected_centers_np[:, 0], detected_centers_np[:, 1], c='r', label='Detected Centers')
        # plt.scatter(tod3_centers_np[:, 0], tod3_centers_np[:, 1], c='b', label='ToD3 Centers')
        # plt.legend()
        # plt.savefig(f'./temp/matched_figure/matched_figure_{time.time()}.png', dpi=300)
        # plt.close()
        return matched_idx.long()

    def get_clock_direction(self, x, y):
        angle = math.degrees(math.atan2(y, x))
        angle = (90 - angle) % 360
        clock_direction = round(angle / 30)
        if clock_direction == 0:
            clock_direction = 12
        return clock_direction

    def get_distance_and_clock_direction(self, x, y):
        clock_direction = self.get_clock_direction(x, y)
        distance = (x**2+y**2)**0.5
        return distance, clock_direction

    def get_velocity_and_clock_direction(self, traj):
        steps = 5
        vx = self.linear_fit_slope(torch.arange(steps).float().to(traj), traj[:steps, 0]) * 2
        vy = self.linear_fit_slope(torch.arange(steps).float().to(traj), traj[:steps, 1]) * 2 # modi2 because the planning step is 0.5s
        velocity = (vx**2+vy**2)**0.5
        clock_direction = self.get_clock_direction(vx, vy)
        return velocity, clock_direction

    def choose_caption(self, gt_caption_token, gt_caplabel, ins_query, outs_track, outs_motion, outs_planning, ins_list, tod3_centers): # modi
        selected_idx = None
        tod3_idx = (ins_list == 0).nonzero(as_tuple=True)[0]
        assert len(tod3_centers) == len(tod3_idx), 'the length of tod3_centers and tod3_idx not equal!'
        detected_centers = (outs_track['track_bbox_results'][0][0].tensor).to(tod3_centers) # neglect the ego track
        detected_centers = detected_centers[:-1, :7]
        matched_idx = self.match_centers(detected_centers, tod3_centers).view((-1))
        flag = 0
        pointer = np.random.rand()
        scene_complexity = len(tod3_centers) # modi2
        if (ins_list == -1).any().item():
            selected_idx = (ins_list == -1).nonzero(as_tuple=True)[0]
            if len(selected_idx) > self.max_qa_num:
                selected_idx = self.sample_tensor(selected_idx, self.max_qa_num)
            
        elif pointer < self.tod3_ratio and torch.sum(matched_idx >= 0).item() > 2:
            flag = 1
            gt_caption_token = gt_caption_token[tod3_idx]
            gt_caplabel = gt_caplabel[tod3_idx]
            ins_query = ins_query[:, matched_idx>=0]
            matched_idx = matched_idx[matched_idx>=0]
            
            if len(matched_idx) > self.max_qa_num:
                idx_idx = torch.arange(len(matched_idx)).long().to(gt_caption_token)
                idx_idx = self.sample_tensor(idx_idx, self.max_qa_num)
                matched_idx = matched_idx[idx_idx]
                ins_query = ins_query[:, idx_idx]

            selected_idx = matched_idx
            if ins_query is not None:
                ins_query = ins_query[0].unsqueeze(1)
            
        elif pointer < self.tod3_ratio + self.caption_ratio:
            caption_idx = (ins_list == -2).nonzero(as_tuple=True)[0]
            command_idx = (ins_list == -3).nonzero(as_tuple=True)[0]
            QA_idx = (ins_list == -4).nonzero(as_tuple=True)[0]
            if len(QA_idx) > self.max_qa_num - 2:
                QA_idx = self.sample_tensor(QA_idx, self.max_qa_num - 2)
            selected_idx = torch.cat([caption_idx, command_idx, QA_idx], dim=0)

        else: # align
            flag = 2
            instruction_list = []
            answer_list = []
            if ins_query is not None:
                ins_query_list = []
                ins_query_idx = torch.arange(len(ins_query[0]), device=ins_query.device, dtype=torch.long)
            # object count test
            query_labels = outs_motion['query_label'].long() # query_labels is a tensor
            vehicle_mask = outs_motion['vehicle_mask']
            count_list = torch.zeros((len(self.CLASSES)), dtype=torch.long)
            for label in query_labels:
                count_list[label] += 1
            question = f"<align_count> Information of the car's bird's eye view video frames, tracks, trajectories and planning are provided, answer the following question: How many cars, trucks, construction_vehicles, buses, bicycles, and pedestrains are detected by the model in this frame?"
            answer = "There are"
            for index in range(len(self.CLASSES)):
                if count_list[index] <= 1 and index in self.valid_CLASS:
                    answer += f" {count_list[index]} {self.CLASS[index]}"
                if count_list[index] > 1 and index in self.valid_CLASS:
                    answer += f" {count_list[index]} {self.CLASSES[index]}"
            answer += " detected in this frame."
            instruction_list.append(question)
            answer_list.append(answer)
            if ins_query is not None:
                ins_query_list.append(ins_query)

            # object position test
            if not torch.all(outs_motion['track_query'] == 0):
                matched_idx_motion = matched_idx[vehicle_mask>0] # modi2
                ins_query_idx_motion = ins_query_idx[vehicle_mask>0]
            else:
                matched_idx_motion = torch.ones((1,)).to(matched_idx) * (-1)
                ins_query_idx_motion = torch.zeros((1,)).to(matched_idx)

            tod3_caption_token = gt_caption_token[tod3_idx]
            tod3_caplabel = gt_caplabel[tod3_idx]
            detected_centers = detected_centers[matched_idx>=0]
            ins_query_idx_track = ins_query_idx[matched_idx>=0]
            matched_idx = matched_idx[matched_idx>=0]
            if len(matched_idx) > self.max_qa_num // 3:
                idx_idx = torch.arange(len(matched_idx)).long().to(gt_caption_token)
                idx_idx = self.sample_tensor(idx_idx, self.max_qa_num // 3)
                matched_idx = matched_idx[idx_idx]
                detected_centers = detected_centers[idx_idx]
                ins_query_idx_track = ins_query_idx_track[idx_idx]
            tod3_caption_token_selected = tod3_caption_token[matched_idx]
            tod3_caplabel_selected = tod3_caplabel[matched_idx]
            question_prefix = f"<align_position><{scene_complexity}> Information of the car's bird's eye view video frames, tracks, trajectories and planning are provided, answer the following question: Tell the distance and clock direction of the "
            for index in range(len(tod3_caption_token_selected)):
                caption = tod3_caption_token_selected[index]
                label = tod3_caplabel_selected[index]
                center = detected_centers[index]
                if isinstance(label, list):
                    label = torch.tensor(label)
                label_indices = torch.nonzero(label, as_tuple=True)[0]
                label = [label[idx].item() for idx in label_indices]
                tod3_caplabel_text = self.tokenizer.decode(label)
                tod3_caplabel_text = re.sub(r" about \d+ meters away ", " ", tod3_caplabel_text)
                instruction_list.append(question_prefix + tod3_caplabel_text + ".")
                distance, clock_direction = self.get_distance_and_clock_direction(center[0], center[1])
                answer_list.append(f"It's about {distance:.1f} meters away at {clock_direction} o'clock direction.")
            if ins_query is not None and len(ins_query[0]) > 0:
                ins_query_list.append(ins_query[0, ins_query_idx_track].unsqueeze(1))

            # object motion test
            traj_preds = outs_motion['all_traj_preds'][-1, 0, :, 0, :, :2]
            if torch.sum(matched_idx_motion>=0) > 0:
                traj_preds = traj_preds[matched_idx_motion>=0]
            ins_query_idx_motion = ins_query_idx_motion[matched_idx_motion>=0]
            matched_idx_motion = matched_idx_motion[matched_idx_motion>=0] # modi2
            if len(matched_idx_motion) > self.max_qa_num // 3:
                idx_idx = torch.arange(len(matched_idx_motion)).long().to(gt_caption_token)
                idx_idx = self.sample_tensor(idx_idx, self.max_qa_num // 3)
                matched_idx_motion = matched_idx_motion[idx_idx]
                traj_preds = traj_preds[idx_idx]
                ins_query_idx_motion = ins_query_idx_motion[idx_idx]
            tod3_caption_token_selected = tod3_caption_token[matched_idx_motion]
            tod3_caplabel_selected = tod3_caplabel[matched_idx_motion]
            question_prefix = f"<align_motion><{scene_complexity}> Information of the car's bird's eye view video frames, tracks, trajectories and planning are provided, answer the following question: Tell the future velocity and moving direction of the "
            for index in range(len(tod3_caption_token_selected)):
                caption = tod3_caption_token_selected[index]
                label = tod3_caplabel_selected[index]
                traj = traj_preds[index]
                if isinstance(label, list):
                    label = torch.tensor(label)
                label_indices = torch.nonzero(label, as_tuple=True)[0]
                label = [label[idx].item() for idx in label_indices]
                tod3_caplabel_text = self.tokenizer.decode(label)
                is_position = tod3_caplabel_text.find(" is ")
                if is_position != -1:
                    tod3_caplabel_text = tod3_caplabel_text[:is_position]
                instruction_list.append(question_prefix + tod3_caplabel_text + ".")
                velocity, clock_direction = self.get_velocity_and_clock_direction(traj)
                if velocity > 0.05:
                    answer_list.append(f"The object is moving at {velocity:.2f} m/s to its {clock_direction} o'clock direction.")
                else:
                    answer_list.append("The object is stationary.")
            if ins_query is not None and len(ins_query[0]) > 0:
                try:
                    ins_query_list.append(ins_query[0, ins_query_idx_motion].unsqueeze(1))
                except:
                    print(ins_query.shape, ins_query_idx_motion)

            question = f"<align_plan><{scene_complexity}> Information of the car's bird's eye view video frames, tracks, trajectories and planning are provided, answer the following question: Interpret the future 5 planning steps from the planning query."
            sdc_traj = outs_planning['sdc_traj_all'][0, :, :2]
            instruction_list.append(question)
            answer_planning = "The future trajectory points are ["
            for i in range(5):
                answer_planning += f"[{sdc_traj[i][0]:.2f},{sdc_traj[i][1]:.2f}],"
            answer_planning += "]."
            answer_list.append(answer_planning)

            if ins_query is not None:
                ins_query_list.append(ins_query)

                max_width = max([t.shape[1] for t in ins_query_list])
                total_length = sum(t.shape[0] for t in ins_query_list)
                result_tensor = torch.zeros((total_length, max_width, ins_query.shape[2]), device=ins_query.device, dtype=ins_query.dtype)
                start_index = 0
                for t in ins_query_list:
                    current_length = t.shape[0]
                    result_tensor[start_index : start_index + current_length, :t.shape[1], :] = t
                    start_index += current_length
                ins_query = result_tensor

        if not flag == 2:
            selected_idx = selected_idx.long().view((-1))
            gt_caption_token = gt_caption_token[selected_idx]
            gt_caplabel = gt_caplabel[selected_idx]
        else:
            gt_caption_token, gt_caplabel = self.tokenize(instruction_list, answer_list)
            gt_caption_token = gt_caption_token.to(tod3_centers.device)
            gt_caplabel = gt_caplabel.to(tod3_centers.device)

        if flag != 1 and flag != 2 and ins_query is not None:
            _bsz = len(gt_caption_token)
            ins_query  = ins_query.repeat((_bsz, 1, 1))

        return gt_caption_token, gt_caplabel, ins_query

    def forward_train(self, bev_embed, gt_caption_token, gt_caplabel, ins_list, tod3_centers, outs_track, outs_motion, outs_planning):
        mask_dict = dict(ins_masked = np.random.rand() < self.train_cfg['ins_masked'],
                         sdc_masked = np.random.rand() < self.train_cfg['sdc_masked'],
                         planning_masked = np.random.rand() < self.train_cfg['planning_masked'])

        ins_query, sdc_query, planning_query = self.merge_queries(outs_track, outs_motion, outs_planning, mask_dict)

        caption_loss = {}
        gt_caption_token = gt_caption_token[0]
        gt_caplabel = gt_caplabel[0]
        ins_list = ins_list[0]
        tod3_centers = tod3_centers[0, ..., :7]
        
        gt_caption_token, gt_caplabel, ins_query = self.choose_caption(gt_caption_token, gt_caplabel, ins_query, outs_track, outs_motion, outs_planning, ins_list, tod3_centers)

        with torch.cuda.amp.autocast():
            if self.captioner_type == 'llama':
                c_loss = self.llama_adapter((gt_caption_token, gt_caplabel), (bev_embed, ins_query, sdc_query, planning_query))
            elif self.captioner_type == 'adapt':
                c_loss = self.adapt((gt_caption_token, gt_caplabel), (bev_embed, ins_query, sdc_query, planning_query))
            else:
                raise ValueError('captioner_type should be either llama or adapt')
            
        caption_loss['caption_loss'] = c_loss
        return caption_loss
    
    def forward_test(self, bev_embed, outs_track, outs_motion, outs_planning, gt_caption_token, gt_caplabel, ins_list, tod3_centers):
        ins_query, sdc_query, planning_query = self.merge_queries(outs_track, outs_motion, outs_planning, self.test_cfg)
        answer_token = []
        instruction_token = []
        gt_caption_token = gt_caption_token[0][0]
        gt_caplabel = gt_caplabel[0][0]
        ins_list = ins_list[0][0]
        tod3_centers = tod3_centers[0][0]

        gt_caption_token, gt_caplabel, ins_query = self.choose_caption(gt_caption_token, gt_caplabel, ins_query, outs_track, outs_motion, outs_planning, ins_list, tod3_centers)

        for i in range(len(gt_caption_token)):
            label = gt_caplabel[i]
            answer_token.append(label[label > 0][:-1])
            instruction = gt_caption_token[i] - label
            instruction_token.append(instruction[instruction > 0])
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if self.captioner_type == 'llama':
                    text, gt_text, question = self.llama_adapter.generate((bev_embed, ins_query, sdc_query, planning_query), (instruction_token, answer_token), max_gen_len=100)
                elif self.captioner_type == 'adapt':
                    text, gt_text, question = self.adapt.generate((bev_embed, ins_query, sdc_query, planning_query), (instruction_token, answer_token), max_gen_len=100)
                else:
                    raise ValueError('captioner_type should be either llama or adapt')

        print(text, gt_text)
        
        return dict(question = question, caption=text, gt_caption=gt_text)

    def tokenize(self, instruction_list, answer_list):
        max_words = 200
        input2_cat = None
        labels_cat = None
        input2_mask_cat = None
        for instruction, answer in zip(instruction_list, answer_list):
            instruction = utils.format_prompt(instruction, None)
            input1 = torch.tensor(self.tokenizer.encode(instruction, bos=True, eos=False), dtype=torch.int64)
            input2 = torch.tensor(self.tokenizer.encode(instruction + answer, bos=True, eos=True), dtype=torch.int64)
            padding = max_words - input2.shape[0]
            if padding > 0:
                input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                input2 = input2[:max_words]
            labels = deepcopy(input2)
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
        return input2_cat, labels_cat

    def linear_fit_slope(self, x, y):
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        denominator = torch.sum((x - x_mean) ** 2)
        slope = numerator / denominator
        return slope.item()

import csv
import json
import os
import os.path as op
import re
import ast
import shutil
import subprocess
import tempfile
import time
from collections import OrderedDict, defaultdict
from pprint import pprint
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import json

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def score(cap_list):
    '''
    input: list[[pred1, gt1], [pred2, gt2], ...]
    '''
    ref = dict()
    hypo = dict()
    if len(cap_list) == 0:
        return
    for i, pair in enumerate(cap_list):
        ref[i] = [pair[1]]
        hypo[i] = [pair[0]]
    scorers = [
        (Bleu(4),["Bleu_1","Bleu_2","Bleu_3","Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(),"ROUGE_L"),
        (Cider(), "Cider"),
    ]
    final_scores = {}
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score)==list:
                for m,s in zip(method,score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        except:
            print(f'error in {method}')

    print(final_scores)

def find_positions(lst, element):
    positions = [index for index, value in enumerate(lst) if value == element]
    return positions

def extract_numbers(input_string):
    pattern = r"\d+"
    matches = re.findall(pattern, input_string)
    numbers = [int(match) for match in matches]
    return numbers

def extract_float_and_int(input_string):
    pattern = r"\b\d+\.\d+|\b\d+"
    matches = re.findall(pattern, input_string)
    numbers = [float(match) if '.' in match else int(match) for match in matches]
    return numbers

def extract_object_counts(input_string):
    CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
    count_list = np.zeros((len(CLASSES),), dtype=int)
    pattern = r"(\d+)\s+(\w+)"
    matches = re.findall(pattern, input_string)
    for number, object_class in matches:
        positions = find_positions(CLASSES, object_class)
        if len(positions) > 0:
            count_list[positions[0]] = int(number)
    return count_list

def extract_trajectory_points(input_string):
    pattern = r'\[\s*-?\d+\.?\d*,\s*-?\d+\.?\d*\]'
    array_strings = re.findall(pattern, input_string)
    arrays = [ast.literal_eval(arr_str) for arr_str in array_strings]
    return(arrays)
    # if match:
    #     points = match.group(1).strip()
    #     if points.endswith(','):
    #         points = points[:-1]
    #     point_list = [list(map(float, point.split(','))) for point in points.split('], [')]
    #     return point_list
    # return []

def extract_direction_and_velocity(input_string):
    if " and " in input_string:
        split_string = input_string.split(" and ")
        direction = split_string[0]
        velocity = split_string[1]
    else:
        direction = 'forward'
        velocity = 'keep speed'
    return direction, velocity

def calculate_accuracy(input_list):
    right_count = 0
    for item in input_list:
        if item[0] == item[1]:
            right_count += 1
    return right_count / len(input_list)

def calculate_pairwise_distances(pred_points, gt_points):
    min_length = min(len(pred_points), len(gt_points))
    distances = []
    for i in range(min(min_length, 4)):
        pred_point = np.array(pred_points[i])
        gt_point = np.array(gt_points[i])
        distance = np.linalg.norm(pred_point - gt_point)
        distances.append(distance)
    return distances

def score_align_count(cap_list, json_file):
    '''
    input: list[[q1, pred1, gt1], [q2, pred2, gt2], ...]
    '''
    diff_list = []
    complexity_list = []
    for caption in cap_list:
        question = caption[0]
        pred = caption[1]
        gt = caption[2]
        scene_complexity = extract_numbers(question)
        if len(scene_complexity) == 0:
            continue
        scene_complexity = scene_complexity[0]
        complexity_list.append(scene_complexity)
        count_list_pred = extract_object_counts(pred)
        count_list_gt = extract_object_counts(gt)
        root_mean_square = np.sqrt(np.sum((count_list_pred - count_list_gt)**2) / len(count_list_pred))
        diff_list.append(root_mean_square)
    with open(json_file.replace('.json', '_align_count.json'), 'w') as f:
        align_count_dict = dict(complexity_list=complexity_list, diff_list=diff_list)
        json.dump(align_count_dict, f)
    mean_count_disalignment = np.mean(diff_list)
    print(f"mean_count_disalignment = {mean_count_disalignment}")

def score_align_positions(cap_list, json_file):
    '''
    input: list[[q1, pred1, gt1], [q2, pred2, gt2], ...]
    '''
    distance_diff_list = []
    direction_diff_list = []
    position_diff_list = []
    complexity_list = []
    for caption in cap_list:
        question = caption[0]
        pred = caption[1]
        gt = caption[2]
        scene_complexity = extract_numbers(question)
        if len(scene_complexity) == 0:
            continue
        scene_complexity = scene_complexity[0]
        numbers_pred = extract_float_and_int(pred)
        numbers_gt = extract_float_and_int(gt)
        if len(numbers_pred) < 2 or len(numbers_gt) < 2:
            continue
        distance_pred, direction_pred = numbers_pred[:2]
        distance_gt, direction_gt = numbers_gt[:2]
        complexity_list.append(scene_complexity)
        distance_diff_list.append(abs(distance_pred - distance_gt))
        direction_diff_degrees = min(abs(direction_pred - direction_gt), 12 - abs(direction_pred - direction_gt)) * 30
        direction_diff_list.append(direction_diff_degrees)
        position_diff_list.append((distance_pred**2+distance_gt**2-2*np.cos(direction_diff_degrees*np.pi/180)*distance_pred*distance_gt)**0.5)
    with open(json_file.replace('.json', '_align_position.json'), 'w') as f:
        align_position_dict = dict(complexity_list=complexity_list, distance_diff_list=distance_diff_list, direction_diff_list=direction_diff_list, position_diff_list=position_diff_list)
        json.dump(align_position_dict, f)
    mean_distance_disalignment = np.mean(distance_diff_list)
    mean_direction_disalignment = np.mean(direction_diff_list)
    mean_position_disalignment = np.mean(position_diff_list)
    print(f"mean_distance_disalignment = {mean_distance_disalignment}")
    print(f"mean_direction_disalignment = {mean_direction_disalignment}")
    print(f"mean_position_disalignment = {mean_position_disalignment}")

def score_align_motion(cap_list, json_file):
    '''
    input: list[[q1, pred1, gt1], [q2, pred2, gt2], ...]
    '''
    velocity_list = []
    velocity_diff_list = []
    direction_diff_list = []
    motion_diff_list = []
    complexity_list = []
    for caption in cap_list:
        question = caption[0]
        pred = caption[1]
        gt = caption[2]
        scene_complexity = extract_numbers(question)
        if len(scene_complexity) == 0:
            continue
        scene_complexity = scene_complexity[0]

        if "stationary" in pred and "stationary" in gt:
            velocity_diff_list.append(0)
            direction_diff_list.append(0)
            velocity_list.append([0,0])
        else:
            if "stationary" in pred:
                velocity_pred, direction_pred = 0, 0
            else:
                numbers_pred = extract_float_and_int(pred)
                if len(numbers_pred) < 2:
                    continue
                velocity_pred, direction_pred = numbers_pred[:2]
            if "stationary" in gt:
                velocity_gt, direction_gt = 0, 0
            else:
                numbers_gt = extract_float_and_int(gt)
                if len(numbers_gt) < 2:
                    continue
                velocity_gt, direction_gt = numbers_gt[:2]
            velocity_list.append([velocity_pred, velocity_gt])
            velocity_diff_list.append(abs(velocity_pred - velocity_gt))
            direction_diff_degrees = min(abs(direction_pred - direction_gt), 12 - abs(direction_pred - direction_gt)) * 30
            direction_diff_list.append(direction_diff_degrees)
            motion_diff_list.append((velocity_pred**2+velocity_gt**2-2*np.cos(direction_diff_degrees*np.pi/180)*velocity_pred*velocity_gt)**0.5)

        complexity_list.append(scene_complexity)

    mean_velocity_disalignment = np.mean(velocity_diff_list)
    mean_direction_disalignment = np.mean(direction_diff_list)
    mean_motion_disalignment = np.mean(motion_diff_list)
    with open(json_file.replace('.json', '_align_motion.json'), 'w') as f:
        align_motion_dict = dict(complexity_list=complexity_list, velocity_list=velocity_list, velocity_diff_list=velocity_diff_list, motion_diff_list=motion_diff_list,
            direction_diff_list=direction_diff_list, mean_velocity_disalignment=mean_velocity_disalignment, mean_direction_disalignment=mean_direction_disalignment, mean_motion_disalignment=mean_motion_disalignment)
        json.dump(align_motion_dict, f)

    print(f"mean_velocity_disalignment = {mean_velocity_disalignment}")
    print(f"mean_velocity_direction_disalignment = {mean_direction_disalignment}")
    print(f"mean_motion_disalignment = {mean_motion_disalignment}")

def score_align_plan(cap_list, json_file):
    '''
    input: list[[q1, pred1, gt1], [q2, pred2, gt2], ...]
    '''
    trajectory_diff_list = []
    traj_length = []

    for caption in cap_list:
        question = caption[0]
        pred = caption[1]
        gt = caption[2]
        scene_complexity = extract_numbers(question)
        if len(scene_complexity) == 0:
            continue
        pred_points = extract_trajectory_points(pred)
        gt_points = extract_trajectory_points(gt)
        if len(pred_points) == 0 or len(gt_points) == 0:
            traj_length.append(0)
            continue
        distances = calculate_pairwise_distances(pred_points, gt_points)
        trajectory_diff_list.append(distances)
        traj_length.append(len(distances))
    
    mean_distances = [np.mean(dist) for dist in trajectory_diff_list if len(dist) > 0]
    overall_mean_distance = np.mean(mean_distances) if len(mean_distances) > 0 else 0
    with open(json_file.replace('.json', '_align_plan.json'), 'w') as f:
        align_plan_dict = dict(traj_length=traj_length, trajectory_diff_list=trajectory_diff_list,
            mean_distances=mean_distances)
        json.dump(align_plan_dict, f)
    print(f"mean_trajectory_disalignment = {overall_mean_distance}")

def save_DriveLM(DriveLM_list, json_file):
    output_dict = []
    for caption in DriveLM_list:
        question, answer = caption
        match = re.search(r'<(.*?)>', question)
        if match:
            idx = match.group(1)
        else:
            continue
        output_dict.append(dict(id=idx, question=question, answer=answer))
    with open(json_file.replace('caption_results.json', 'DriveLM_output.json'), 'w') as f:
        json.dump(output_dict, f)

def score_QA(qa_list):
    hop0_list = []
    hop1_list = []
    hop0_count = 0
    hop1_count = 0
    for item in qa_list:
        question, answer, gt_answer = item
        matches = re.findall(r'<(.*?)>', question)
        if len(matches) >= 2:
            hop = matches[1]
        else:
            continue
        if hop == '0':
            hop0_list.append([answer, gt_answer])
            hop0_count += 1
        else:
            hop1_list.append([answer, gt_answer])
            hop1_count += 1
    print("hop0 Acc: ", calculate_accuracy(hop0_list), " hop0_count: ", hop0_count)
    print("hop1 Acc: ", calculate_accuracy(hop1_list), " hop1_count: ", hop1_count)
    
def score_command(command_list):
    direction_list = []
    velocity_list = []
    for command_pair in command_list:
        pred, gt = command_pair
        direction_pred, velocity_pred = extract_direction_and_velocity(pred)
        direction_gt, velocity_gt = extract_direction_and_velocity(gt)
        direction_list.append([direction_pred, direction_gt])
        velocity_list.append([velocity_pred, velocity_gt])
    print("Acc of direction: ", calculate_accuracy(direction_list))
    print("Acc of velocity: ", calculate_accuracy(velocity_list))
    print("Acc of command: ", calculate_accuracy(command_list))

def score_caption(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    narration_pattern = r"Narration: (.*?)\."
    reasoning_pattern = r"Reasoning: (.*?)\."
    
    narration_list = []
    reasoning_list = []
    command_list = []
    qa_list = []
    DriveLM_list = []
    tod3_list = []
    align_count_list = []
    align_position_list = []
    align_motion_list = []
    align_plan_list = []

    for item in data:
        question = item[0]
        answer = item[1]
        gt_answer = item[2]
        batch_size = len(question)
        for batch_index in range(batch_size):
            if '<caption>' in question[batch_index]:
                try:
                    pred_narration = re.search(narration_pattern, answer[batch_index]).group(1)
                    pred_reasoning = re.search(reasoning_pattern, answer[batch_index]).group(1)
                    gt_narraion = re.search(narration_pattern, gt_answer[batch_index]).group(1)
                    gt_reasoning = re.search(reasoning_pattern, gt_answer[batch_index]).group(1)
                    narration_list.append([pred_narration, gt_narraion])
                    reasoning_list.append([pred_reasoning, gt_reasoning])
                except:
                    pass
            if '<command>' in question[batch_index]:
                command_list.append([answer[batch_index], gt_answer[batch_index]])
            if '<QA>' in question[batch_index]:
                qa_list.append([question[batch_index], answer[batch_index], gt_answer[batch_index]])
            if '<DriveLM>' in question[batch_index]:
                DriveLM_list.append([question[batch_index], answer[batch_index], gt_answer[batch_index]])
            if '<tod3>' in question[batch_index]:
                tod3_list.append([answer[batch_index], gt_answer[batch_index]]) 
            if '<align_count>' in question[batch_index]:
                align_count_list.append([question[batch_index], answer[batch_index], gt_answer[batch_index]])
            if '<align_position>' in question[batch_index]:
                align_position_list.append([question[batch_index], answer[batch_index], gt_answer[batch_index]])
            if '<align_motion>' in question[batch_index]:
                align_motion_list.append([question[batch_index], answer[batch_index], gt_answer[batch_index]])
            if '<align_plan>' in question[batch_index]:
                align_plan_list.append([question[batch_index], answer[batch_index], gt_answer[batch_index]])
        
    # save narration and reasoning list to original file location
    with open(json_file.replace('.json', '_split.txt'), 'w') as f:
        for index in range(len(tod3_list)):
            f.write(f"gt_answer: {tod3_list[index][0]}\n")
            f.write(f"pred_answer: {tod3_list[index][1]}\n")
            f.write("\n")
        for index in range(len(DriveLM_list)):
            f.write(f"question: {DriveLM_list[index][0]}\n")
            f.write(f"pred_answer: {DriveLM_list[index][1]}\n")
            f.write(f"gt_answer: {DriveLM_list[index][2]}\n")
            f.write("\n")
        for index in range(len(narration_list)):
            f.write(f"gt:\n{narration_list[index][1]}\n{reasoning_list[index][1]}\n")
            f.write(f"pred:\n{narration_list[index][0]}\n{reasoning_list[index][0]}\n")
            f.write("\n")
        for index in range(len(command_list)):
            f.write(f"gt_command: {command_list[index][1]}\n")
            f.write(f"pred_command: {command_list[index][0]}\n")
            f.write("\n")
        for index in range(len(qa_list)):
            f.write(f"question: {qa_list[index][0]}\n")
            f.write(f"pred_answer: {qa_list[index][1]}\n")
            f.write(f"gt_answer: {qa_list[index][2]}\n")
            f.write("\n")
        for index in range(len(align_count_list)):
            f.write(f"question: {align_count_list[index][0]}\n")
            f.write(f"pred_answer: {align_count_list[index][1]}\n")
            f.write(f"gt_answer: {align_count_list[index][2]}\n")
            f.write("\n")
        for index in range(len(align_position_list)):
            f.write(f"question: {align_position_list[index][0]}\n")
            f.write(f"pred_answer: {align_position_list[index][1]}\n")
            f.write(f"gt_answer: {align_position_list[index][2]}\n")
            f.write("\n")
        for index in range(len(align_motion_list)):
            f.write(f"question: {align_motion_list[index][0]}\n")
            f.write(f"pred_answer: {align_motion_list[index][1]}\n")
            f.write(f"gt_answer: {align_motion_list[index][2]}\n")
            f.write("\n")
        for index in range(len(align_plan_list)):
            f.write(f"question: {align_plan_list[index][0]}\n")
            f.write(f"pred_answer: {align_plan_list[index][1]}\n")
            f.write(f"gt_answer: {align_plan_list[index][2]}\n")
            f.write("\n")
    
    print("=========== Scoring Narration ===========")
    score(narration_list)
    print("=========== Scoring Reasoning ===========")
    score(reasoning_list)
    print("=========== Scoring tod3 ===========")
    score(tod3_list)
    print("=========== Scoring DriveLM ===========")
    save_DriveLM([[pair[0], pair[1]] for pair in DriveLM_list], json_file)
    print("=========== Scoring command ===========")
    score_command(command_list)
    print("=========== Scoring Q&A ===========")
    score_QA(qa_list)
    print("=========== Scoring align ===========")
    score_align_count(align_count_list, json_file)
    score_align_positions(align_position_list, json_file)
    score_align_motion(align_motion_list, json_file)
    score_align_plan(align_plan_list, json_file)

# for debug
# caption = [["There are 5 cars.", "The future trajectory points are [[-0.53,3.03],[-1.54,6.27],[-3.03,9.69],[-4.57,13.55],[-6.48,16.84],].", "The future trajectory points are [[-0.53,3.03],[-1.54,6.27],[-3.03,9.69],[-4.57,13.55],[-6.48,16.84],]."]]
# json_file = "temp/caption.json"
# score_align_plan(caption, json_file)
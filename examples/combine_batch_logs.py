# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

'''
based on https://stackoverflow.com/questions/43068200/how-to-display-the-average-of-multiple-runs-on-tensorboard
'''
import os
from collections import defaultdict

import numpy as np
import shutil
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter
import argparse

tag_mapping = {#'rewards0/frame': 'policy_loss/step', 'rewards0/iter': 'policy_loss/iter', 'rewards0/time': 'policy_loss/time', 
                'rewards0/frame': 'rewards/step', 'rewards0/iter': 'rewards/iter', 'rewards0/time': 'rewards/time', 
                # 'rewards/frame': 'policy_loss/step', 'rewards/iter': 'policy_loss/iter', 'rewards/time': 'policy_loss/time', 
                'rewards/frame': 'rewards/step', 'rewards/step': 'rewards/step', 'rewards/iter': 'rewards/iter', 'rewards/time': 'rewards/time', 
                'policy_loss/step': 'policy_loss/step', 'policy_loss/iter': 'policy_loss/iter', 'policy_loss/time': 'policy_loss/time', 
                'actor_loss/iter': 'actor_loss/iter', 'actor_loss/step': 'actor_loss/step', 
                # 'policy_loss/step': 'rewards/step', 'policy_loss/iter': 'rewards/iter', 'policy_loss/time': 'rewards/time', 
                'training_loss/step': 'training_loss/step', 'training_loss/iter': 'training_loss/iter', 'training_loss/time': 'training_loss/time',
                'best_policy_loss/step': 'best_policy_loss/step',
                'episode_lengths/iter': 'episode_lengths/iter', 'episode_lengths/step': 'episode_lengths/step', 'episode_lengths/frame': 'episode_lengths/step',
                'value_loss/step': 'value_loss/step', 'value_loss/iter': 'value_loss/iter'}

def tabulate_events(dpath):

    summary_iterators = []
    for dname in os.listdir(dpath):
        for subfolder_name in args.subfolder_names:
            if os.path.exists(os.path.join(dpath, dname, subfolder_name)):
                summary_iterators.append(EventAccumulator(os.path.join(dpath, dname, subfolder_name)).Reload())
                break
            
    tags = summary_iterators[0].Tags()['scalars']

    # for it in summary_iterators:
    #     assert it.Tags()['scalars'] == tags

    out_values = dict()
    out_steps = dict()

    for tag in tags:
        if tag not in tag_mapping.keys():
            continue

        # gathering steps
        steps_set = set()
        for summary in summary_iterators:
            for event in summary.Scalars(tag):
                steps_set.add(event.step)

        is_reward = ('reward' in tag)
        is_loss = ('loss' in tag)

        steps = list(steps_set)
        steps.sort()

        # steps = steps[:500]
        
        new_tag_name = tag_mapping[tag]

        out_values[new_tag_name] = np.zeros((len(steps), len(summary_iterators)))
        out_steps[new_tag_name] = np.array(steps)

        for summary_id, summary in enumerate(summary_iterators):
            events = summary.Scalars(tag)
            i = 0
            for step_id, step in enumerate(steps):
                while i + 1 < len(events) and events[i + 1].step <= step:
                    i += 1
                # if events[i].value > 100000. or events[i].value < -100000.:
                #     import IPython
                #     IPython.embed()
                    
                out_values[new_tag_name][step_id, summary_id] = events[i].value

    return out_steps, out_values


def write_combined_events(dpath, acc_steps, acc_values, dname='combined'):
    fpath = os.path.join(dpath, dname)
    
    if os.path.exists(fpath):
        shutil.rmtree(fpath)

    writer = SummaryWriter(fpath)

    tags = acc_values.keys()

    for tag in tags:
        for i in range(len(acc_values[tag])):
            mean = np.array(acc_values[tag][i]).mean()
            writer.add_scalar(tag, mean, acc_steps[tag][i])

    writer.flush()

parser = argparse.ArgumentParser()
parser.add_argument('--batch-folder', type = str, default='path/to/batch/folder')
parser.add_argument('--subfolder-names', type = str, nargs = '+', default=['log', 'runs']) # 'runs' for rl

args = parser.parse_args()

dpath = args.batch_folder

acc_steps, acc_values = tabulate_events(dpath)

write_combined_events(dpath, acc_steps, acc_values)
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os.path as osp

def make_dir(dataset, tag):
    if dataset == 'NTU':
        output_dir = os.path.join(f'./results/NTU{tag}/')
    elif dataset == 'NTU120':
        output_dir = os.path.join(f'./results/NTU120{tag}/')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def get_num_classes(dataset, tag):
    if dataset == 'NTU':
        if tag == 'ar':
            return 60
        if tag == 'ri':
            return 40
        if tag == 'gc':
            return 2
        return 60
    elif dataset == 'NTU120':
        if tag == 'ar':
            return 120
        if tag == 'ri':
            return 106
        if tag == 'gc':
            return 2
        return 120

    

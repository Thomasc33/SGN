# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import os.path as osp
import datetime

def make_dir(dataset, tag):
    # date string
    now = datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')
    if dataset == 'NTU':
        output_dir = os.path.join(f'./results/NTU{tag}/{now}/')
    elif dataset == 'NTU120':
        output_dir = os.path.join(f'./results/NTU120{tag}/{now}/')
    elif dataset == 'ETRI':
        output_dir = os.path.join(f'./results/ETRI{tag}/{now}/')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def get_num_classes(dataset, tag):
    if tag == 'gc': return 2
    if dataset == 'NTU':
        if tag == 'ar':
            return 60
        if tag == 'ri':
            return 40
        return 60
    elif dataset == 'NTU120':
        if tag == 'ar':
            return 120
        if tag == 'ri':
            return 106
        return 120
    elif dataset == 'ETRI':
        if tag == 'ar':
            return 55
        if tag == 'ri':
            return 100
        return 100
    

    

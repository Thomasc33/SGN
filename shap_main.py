# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import time
import shutil
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import os.path as osp
import csv
import numpy as np
import shap
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients


np.random.seed(1337)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from model import SGN
from data import NTUDataLoaders, AverageMeter
import fit
from util import make_dir, get_num_classes




parser = argparse.ArgumentParser(description='Skeleton-Based Action Recognition')
fit.add_fit_args(parser)
parser.set_defaults(
    network='SGN',
    dataset='NTU',
    case=0,
    batch_size=64,
    max_epochs=120,
    monitor='val_acc',
    lr=0.001,
    weight_decay=0.0001,
    lr_factor=0.1,
    workers=16,
    print_freq=20,
    train=0,
    seg=20,
    tag='ar'
)
args = parser.parse_args()


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class CustomSHAPDeepExplainer(shap.DeepExplainer):
    def __init__(self, model, data, *args, **kwargs):
        super().__init__(model, data, *args, **kwargs)
        self._saved_forward_hooks = {}  # Initialize the attribute
        self._register_custom_hooks()

    def _register_custom_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.AdaptiveMaxPool2d):
                module.register_forward_hook(self._adaptive_max_pool2d_hook)

    def _adaptive_max_pool2d_hook(self, module, input, output):
        self._saved_forward_hooks[module] = (input, output)

    def gradient(self, feature_ind, joint_x):
        self.model.zero_grad()
        joint_x = tuple(joint_x)
        outputs = self.model(*joint_x)
        
        selected = [outputs[i, feature_ind[i]] for i in range(len(feature_ind))]
        grad = torch.autograd.grad(selected, joint_x, grad_outputs=torch.ones_like(torch.stack(selected)), create_graph=True)
        
        for module, (input, output) in self._saved_forward_hooks.items():
            if isinstance(module, nn.AdaptiveMaxPool2d):
                # Custom gradient calculation for AdaptiveMaxPool2d
                delta_out = output[0].detach() - output[1].detach()
                delta_in = input[0].detach() - input[1].detach()
                grad_output = grad[0]
                grad_input = grad_output * (delta_out / delta_in).repeat(1, 1, 1, 1)
                grad = (grad_input,) + grad[1:]
        
        return grad
    
def analyze_joint_influence(attention_weights):
    # Sum the attention weights across the batch and time dimensions
    # Assuming attention_weights shape is (batch_size, num_joints, num_joints, time_steps)
    influence_scores = attention_weights.sum(dim=[0, 3]).mean(dim=0)  # Sum over batch and time, mean over joints
    return influence_scores


def plot_joint_influence(influence_scores, num_joints=25):
    plt.bar(range(num_joints), influence_scores)
    plt.xlabel('Joint Index')
    plt.ylabel('Influence Score')
    plt.title('Joint Influence on Action Recognition')
    plt.show()


def main():
    args.num_classes = get_num_classes(args.dataset, args.tag)
    model = SGN(args.num_classes, args.dataset, args.seg, args)

    total = get_n_params(model)
    print('The number of parameters: ', total)
    print('The modes is:', args.network)

    if torch.cuda.is_available():
        print('It is using GPU!')
        model = model.cuda()

    # Load pretrained model
    pretrained_path = 'results/NTUar/SGN/0_best.pth'
    model.load_state_dict(torch.load(pretrained_path)['state_dict'])
    model.eval()

    # Prepare the test data loader with a fixed batch size
    ntu_loaders = NTUDataLoaders(args.dataset, args.case, seg=args.seg, tag=args.tag)
    test_loader = ntu_loaders.get_test_loader(batch_size=32, num_workers=args.workers)
    
    # Get a batch of test data
    inputs, targets = next(iter(test_loader))
    print(inputs.shape, targets.shape)

    inputs = inputs.cuda()
    targets = targets.cuda()

    ig = IntegratedGradients(model)

    attributions, delta = ig.attribute(inputs, target=targets, return_convergence_delta=True)

    attributions = attributions.cpu().detach().numpy()

    average_attributions = np.mean(np.abs(attributions), axis=0)
    print(average_attributions.shape)
    print('Average attributions:', average_attributions)
    return

    # Create shap explainer with a fixed batch size
    explainer = shap.DeepExplainer(model, inputs.cuda()) 

    # Compute shap values
    shap_values = explainer.shap_values(inputs.cuda())

    # Visualize the shap values
    shap.summary_plot(shap_values, inputs.numpy())

if __name__ == '__main__':
    main()

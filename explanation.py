import torch
import numpy as np
import pickle
from model import SGN
from types import SimpleNamespace
from captum.attr import IntegratedGradients

datasets = {
    'NTU': {
        'ar_model': 'results/NTUar/SGN/1_best.pth',
        'ri_model': 'results/NTUri/SGN/1_best.pth',
        'num_classes': 60,
        'num_actors': 40,
        'joints': 25,
    },
    'NTU120': {
        'ar_model': 'results/NTU120ar/SGN/1_best.pth',
        'ri_model': 'results/NTU120ri/SGN/1_best.pth',
        'num_classes': 120,
        'num_actors': 106,
        'joints': 25,
    },
    'ETRI': {
        'ar_model': 'results/ETRIar/SGN/1_best.pth',
        'ri_model': 'results/ETRIri/SGN/1_best.pth',
        'num_classes': 0,
        'num_actors': 0,
        'joints': 25,
    }
}


class Explanation():
    def __init__(self, dataset):
        self.dataset = dataset
        assert dataset in datasets.keys(), 'Dataset not found'
        
        self.num_classes = datasets[dataset]['num_classes']
        self.num_actors = datasets[dataset]['num_actors']
        self.joints = datasets[dataset]['joints']

        args = SimpleNamespace(batch_size=32, train=0)

        self.sgn_ar = SGN(self.num_classes, None, 20, args, 0).cuda()
        self.sgn_priv = SGN(self.num_actors, None, 20, args, 0).cuda()
        self.sgn_ar.load_state_dict(torch.load(datasets[dataset]['ar_model'], weights_only=True)['state_dict'], strict=False)
        self.sgn_priv.load_state_dict(torch.load(datasets[dataset]['ri_model'], weights_only=True)['state_dict'], strict=False)
        self.sgn_ar.eval()
        self.sgn_priv.eval()

        # Define models for Captum
        def model_ar(input_tensor):
            output = self.sgn_ar.eval_single(input_tensor)
            return output

        def model_ri(input_tensor):
            output = self.sgn_priv.eval_single(input_tensor)
            return output

        # Initialize Integrated Gradients
        self.ig_ar = IntegratedGradients(model_ar)
        self.ig_ri = IntegratedGradients(model_ri)

    def importance_score(self, sample, label, is_action=False, alpha=0.1):
        reshaped_skeleton = self.reshape_skeleton(sample)
        input_tensor = torch.tensor(reshaped_skeleton, dtype=torch.float32).unsqueeze(0).cuda()
        
        # Compute attributions for both models
        ar_attribution = self.ig_ar.attribute(input_tensor, target=label if is_action else 0).abs()
        ri_attribution = self.ig_ri.attribute(input_tensor, target=0 if is_action else label).abs()

        joint_importances_ar = {}
        joint_importances_ri = {}

        for joint in range(self.joints):
            joint_indices = [joint * 3 + c for c in range(3)]

            # Get attributions for these indices across all frames
            joint_attributions_ar = ar_attribution[0, :, joint_indices]  # Shape: (20, 3)
            joint_attributions_ri = ri_attribution[0, :, joint_indices]
            
            # Sum over frames and coords
            joint_importance_ar = joint_attributions_ar.sum().item()
            joint_importance_ri = joint_attributions_ri.sum().item()

            joint_importances_ar[joint] = joint_importance_ar
            joint_importances_ri[joint] = joint_importance_ri

        # Normalize the importance scores so that they sum to 1
        total_importance_ar = sum(joint_importances_ar.values())
        total_importance_ri = sum(joint_importances_ri.values())

        # To avoid division by zero in case total importance is zero
        if total_importance_ar == 0:
            total_importance_ar = 1e-8
        if total_importance_ri == 0:
            total_importance_ri = 1e-8

        normalized_importances_ar = {joint: importance / total_importance_ar for joint, importance in joint_importances_ar.items()}
        normalized_importances_ri = {joint: importance / total_importance_ri for joint, importance in joint_importances_ri.items()}

        # Now compute the importance score using normalized attributions
        importance = {
            joint: normalized_importances_ri[joint] + alpha * (1 - normalized_importances_ar[joint])
            for joint in range(self.joints)
        }

        return importance


    def reshape_skeleton(self, skeleton):
        """
        Reshape the skeleton data from shape [300, 150] to [20, 75].
        """
        skeleton = skeleton[:20, :75]
        return skeleton

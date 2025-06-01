"""
Data loading and processing utilities
"""
import numpy as np
import h5py
import os
import random
import torch
from explanation import Explanation
import math

def load_ntu_data(dataset='NTU', case=0, tag='ar', num_samples=5):
    """
    Load sample data from NTU dataset

    Args:
        dataset: 'NTU', 'NTU120', or 'ETRI'
        case: 0 for CS, 1 for CV
        tag: 'ar' for action recognition, 'ri' for re-identification
        num_samples: number of random samples to load

    Returns:
        samples: list of (skeleton_data, label) tuples
    """
    if dataset == 'NTU':
        metric = 'CS' if case == 0 else 'CV'
        path = f'./data/ntu/NTU_{metric}_{tag}.h5'
    elif dataset == 'NTU120':
        metric = 'CS' if case == 0 else 'CV'
        path = f'./data/ntu120/NTU_{metric}_{tag}.h5'
    elif dataset == 'ETRI':
        metric = 'CS' if case == 0 else 'CV'
        path = f'./data/etri/ETRI_{metric}_{tag}.h5'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    with h5py.File(path, 'r') as f:
        X = f['x'][:]
        Y = np.argmax(f['y'][:], -1)

    # Randomly sample data
    indices = random.sample(range(len(X)), min(num_samples, len(X)))
    samples = [(X[i], Y[i]) for i in indices]

    return samples

def reshape_skeleton(skeleton):
    """
    Reshape skeleton data from [300, 150] to [20, 75] and extract single person

    Args:
        skeleton: numpy array of shape [frames, 150]

    Returns:
        reshaped: numpy array of shape [20, 75]
    """
    # Take first 20 frames and first 75 features (first person)
    reshaped = skeleton[:20, :75]
    return reshaped

def extract_joints_from_frame(frame_data):
    """
    Extract 3D joint coordinates from a single frame

    Args:
        frame_data: numpy array of shape [75] representing one frame of skeleton data

    Returns:
        joints: numpy array of shape [25, 3] with joint coordinates
    """
    joints = frame_data.reshape(25, 3)
    return joints

def apply_smart_masking(skeleton, label, explanation, alpha=0.9, beta=0.2, tag='ar'):
    """
    Apply smart masking to skeleton data

    Args:
        skeleton: numpy array of shape [frames, 150]
        label: action/person label
        explanation: Explanation object
        alpha: weighting parameter
        beta: masking percentage
        tag: 'ar' or 'ri'

    Returns:
        masked_skeleton: skeleton with masked joints
        masked_joints: list of masked joint indices
    """
    # Reshape for explanation
    reshaped = reshape_skeleton(skeleton)

    # Convert label to int if needed
    if isinstance(label, np.ndarray):
        label = int(label.item())
    elif isinstance(label, (np.int64, np.int32)):
        label = int(label)

    # Get importance scores
    importance = explanation.importance_score(reshaped, label, is_action=(tag == 'ar'), alpha=alpha)

    # Sort joints by importance
    sorted_joints = sorted(importance.items(), key=lambda item: item[1], reverse=True)

    # Get top beta% of joints to mask
    top_joints = sorted_joints[:max(1, math.floor(len(sorted_joints) * beta))]
    joints_to_mask = [joint for joint, score in top_joints]

    # Create mask indices (x, y, z for each joint)
    mask_indices = []
    for joint in joints_to_mask:
        mask_indices.extend([joint*3, joint*3+1, joint*3+2])

    # Apply masking
    masked_skeleton = skeleton.copy()
    masked_skeleton[:, mask_indices] = 0

    return masked_skeleton, joints_to_mask, importance

def apply_smart_noise(skeleton, label, explanation,
                      alpha=0.9, sigma=0.01, tag='ar'):
    """
    Add *less* noise to important joints, more to unimportant ones.
    Joints with highest importance receive ≈ 0 noise; the total
    variance budget across the whole skeleton is still ≈ sigma.
    """
    reshaped = reshape_skeleton(skeleton)

    # --- importance (already ∑ = 1 from Explanation) --------------
    importance = explanation.importance_score(
        reshaped, int(label), is_action=(tag == 'ar'), alpha=alpha)

    imp = np.array([importance[j] for j in range(25)])
    imp_norm = (imp - imp.min()) / (imp.ptp() + 1e-8)      # [0,1]
    scale_per_joint = 1.0 - imp_norm                       # inverse importance
    scale_per_joint = scale_per_joint / scale_per_joint.sum()  # normalise
    scale_coords = np.repeat(scale_per_joint, 3) * sigma * 25  # keep σ global

    noisy = skeleton.copy()
    for i in range(75):           # first actor only
        if np.all(noisy[:, i] == 0):
            continue
        zero_end = np.argmax(noisy[:, i] == 0)
        zero_end = zero_end if zero_end else noisy.shape[0]
        noisy[:zero_end, i] += np.random.normal(
            0, scale_coords[i], zero_end)

    return noisy, importance

def apply_group_noise(
        skeleton, label, explanation,
        beta=0.2,             # fraction of joints you call "sensitive"
        alpha=0.1,            # alpha for integrated-gradients
        sigma=0.001,          # global noise budget
        tag='ar'              # 'ar' for action recognition, 'ri' for re-identification
):
    """
    Faithful implementation of group noise from data.py
    """
    # Work with copy to avoid modifying original
    seq = skeleton.copy()

    # Get importance scores (same as data.py line 235)
    importance = explanation.importance_score(seq, int(label), is_action=tag == 'ar', alpha=alpha)

    # Create gamma array (same as data.py line 237)
    gamma = np.repeat(np.array(0.03), 75)

    # Extract importance values for joints (same as data.py line 239)
    phi = np.array([importance[joint] for joint in range(25)])

    # Expand to 75 coordinates (same as data.py line 242)
    phi = np.repeat(phi, 3)

    # Group 1: top beta% of joints (same as data.py lines 248-253)
    sorted_joints = sorted(importance.items(), key=lambda item: item[1], reverse=True)
    top_joints = sorted_joints[:max(1, math.floor(len(sorted_joints) * beta))]
    joints = [joint for joint, score in top_joints]

    # Create mask indices for both actors (same as data.py lines 255-262)
    maskidx = []
    for i in joints:
        maskidx.append(i*3)
        maskidx.append(i*3+1)
        maskidx.append(i*3+2)
        maskidx.append(i*3+75)
        maskidx.append(i*3+1+75)
        maskidx.append(i*3+2+75)

    # Calculate epsilon values (same as data.py lines 267-268)
    epsilon_s = (gamma/(3*len(joints)*gamma + (75-3*len(joints))))
    epsilon_n = (1/(3*len(joints)*gamma + (75-3*len(joints))))

    # Tile epsilon for both actors (same as data.py lines 271-272)
    epsilon_s = np.tile(epsilon_s, 2)
    epsilon_n = np.tile(epsilon_n, 2)

    # Apply noise frame by frame (exact copy of data.py lines 274-278)
    for i in range(150):
        # Skip zero padded frames
        if np.all(seq[:, i] == 0):
            continue
        zero_start = np.argmax(seq[:, i])
        # Apply noise with appropriate epsilon (exact formula from data.py)
        noise_std = sigma/75/(epsilon_s[i] if i in maskidx else epsilon_n[i])
        seq[:zero_start, i] = seq[:zero_start, i] + np.random.normal(0, noise_std, seq[:zero_start, i].shape)

    return seq, joints, importance

def apply_naive_noise(skeleton, sigma=0.01):
    """
    Apply naive (random) noise to skeleton data

    Args:
        skeleton: numpy array of shape [frames, 150]
        sigma: noise standard deviation

    Returns:
        noisy_skeleton: skeleton with random noise applied
    """
    noisy_skeleton = skeleton.copy()

    for i in range(75):  # Only first person
        if np.all(noisy_skeleton[:, i] == 0):
            continue
        zero_start = np.argmax(noisy_skeleton[:, i] == 0) if np.any(noisy_skeleton[:, i] == 0) else len(noisy_skeleton[:, i])
        if zero_start == 0:
            zero_start = len(noisy_skeleton[:, i])
        noisy_skeleton[:zero_start, i] += np.random.normal(0, sigma, zero_start)

    return noisy_skeleton

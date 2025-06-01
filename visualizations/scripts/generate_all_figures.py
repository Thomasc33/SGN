"""
Main script to generate all visualization figures for the presentation - UPDATED VERSION
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add parent directory to path to import project modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from explanation import Explanation
from skeleton_utils import *
from data_utils import *
from figure_generators import *

def create_directory_structure():
    """Create the directory structure for outputs"""
    base_dir = os.path.join(os.path.dirname(__file__), '..')

    directories = [
        'figures/teaser',
        'figures/skeleton_sensitivity',
        'figures/skeleton_attribution',
        'figures/pipeline',
        'figures/process_diagram',
        'figures/heatmap',
        'figures/before_after_masking',
        'figures/noise_comparison',
        'output/images',
        'output/gifs'
    ]

    for directory in directories:
        full_path = os.path.join(base_dir, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

    return base_dir

def create_skeleton_animation(samples, explanation, output_dir, figure_type='original'):
    """Create animated GIF of skeleton sequences with improved visualization"""
    print(f"Creating {figure_type} animation...")

    skeleton, label = samples[0]
    reshaped = reshape_skeleton(skeleton)

    # Calculate global bounds for consistent scaling
    bounds_3d, _, _ = calculate_global_bounds(reshaped)

    # Pre-compute transformations and importance scores for full sequence
    if figure_type == 'sensitivity':
        # Convert label to int if needed
        if isinstance(label, np.ndarray):
            label = int(label.item())
        elif isinstance(label, (np.int64, np.int32)):
            label = int(label)
        importance = explanation.importance_score(reshaped, label, is_action=True, alpha=0.9)
    elif figure_type == 'smart_masking':
        masked_skeleton, masked_joints, importance = apply_smart_masking(skeleton, label, explanation, beta=0.2)
    elif figure_type == 'smart_noise':
        smart_noisy, smart_importance = apply_smart_noise(skeleton, label, explanation, sigma=0.01)
    elif figure_type == 'group_noise':
        group_noisy, sensitive_joints, group_importance = apply_group_noise(skeleton, label, explanation, beta=0.2, sigma=0.01, tag='ar')
    elif figure_type == 'naive_noise':
        naive_noisy = apply_naive_noise(skeleton, sigma=0.01)

    frames = []

    # Remove zero padding
    non_zero_frames = np.any(skeleton[:, :75] != 0, axis=1)
    skeleton = skeleton[non_zero_frames]
    num_frames = skeleton.shape[0]

    # Create frames with consistent styling
    for frame_idx in range(num_frames):
        fig, ax = create_skeleton_figure(figsize=(8, 8))

        # Extract joints for current frame, handling full sequence
        if frame_idx < 20:
            # Use reshaped data for first 20 frames
            frame_joints = extract_joints_from_frame(reshaped[frame_idx])
        else:
            # For frames beyond 20, extract directly from full skeleton
            frame_data = skeleton[frame_idx, :75]  # First person only
            frame_joints = extract_joints_from_frame(frame_data)

        if figure_type == 'original':
            draw_skeleton_3d(ax, frame_joints, title=f'Original - Frame {frame_idx+1}',
                           fixed_bounds=bounds_3d)

        elif figure_type == 'sensitivity':
            # Use pre-computed importance scores
            draw_skeleton_3d(ax, frame_joints, importance_scores=importance,
                           title=f'Sensitivity - Frame {frame_idx+1}',
                           colormap='plasma', fixed_bounds=bounds_3d)

        elif figure_type == 'smart_masking':
            # Extract from pre-computed masked skeleton
            if frame_idx < 20:
                masked_reshaped = reshape_skeleton(masked_skeleton)
                masked_frame_joints = extract_joints_from_frame(masked_reshaped[frame_idx])
            else:
                masked_frame_data = masked_skeleton[frame_idx, :75]
                masked_frame_joints = extract_joints_from_frame(masked_frame_data)
            draw_skeleton_3d(ax, masked_frame_joints, masked_joints=masked_joints,
                           title=f'Smart Masking - Frame {frame_idx+1}',
                           fixed_bounds=bounds_3d)

        elif figure_type == 'smart_noise':
            # Extract from pre-computed noisy skeleton
            if frame_idx < 20:
                smart_reshaped = reshape_skeleton(smart_noisy)
                smart_frame_joints = extract_joints_from_frame(smart_reshaped[frame_idx])
            else:
                smart_frame_data = smart_noisy[frame_idx, :75]
                smart_frame_joints = extract_joints_from_frame(smart_frame_data)
            draw_skeleton_3d(ax, smart_frame_joints, importance_scores=smart_importance,
                           title=f'Smart Noise - Frame {frame_idx+1}',
                           colormap='plasma', fixed_bounds=bounds_3d)

        elif figure_type == 'group_noise':
            # Extract from pre-computed group noisy skeleton
            if frame_idx < 20:
                group_reshaped = reshape_skeleton(group_noisy)
                group_frame_joints = extract_joints_from_frame(group_reshaped[frame_idx])
            else:
                group_frame_data = group_noisy[frame_idx, :75]
                group_frame_joints = extract_joints_from_frame(group_frame_data)
            draw_skeleton_3d(ax, group_frame_joints, masked_joints=sensitive_joints,
                           title=f'Group Noise - Frame {frame_idx+1}',
                           fixed_bounds=bounds_3d)

        elif figure_type == 'naive_noise':
            # Extract from pre-computed naive noisy skeleton
            if frame_idx < 20:
                naive_reshaped = reshape_skeleton(naive_noisy)
                naive_frame_joints = extract_joints_from_frame(naive_reshaped[frame_idx])
            else:
                naive_frame_data = naive_noisy[frame_idx, :75]
                naive_frame_joints = extract_joints_from_frame(naive_frame_data)
            draw_skeleton_3d(ax, naive_frame_joints,
                           title=f'Random Noise - Frame {frame_idx+1}',
                           fixed_bounds=bounds_3d)

        # Convert to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)

    # Save as GIF with better quality
    gif_path = os.path.join(output_dir, f'{figure_type}_animation.gif')
    imageio.mimsave(gif_path, frames, duration=0.15, loop=0)
    print(f"Saved animation: {gif_path}")

def main():
    """Main function to generate all figures"""
    print("\n" + "="*60)
    print("STARTING VISUALIZATION GENERATION")
    print("="*60 + "\n")

    # Create directory structure
    base_dir = create_directory_structure()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load explanation system
    print("Loading explanation system...")
    try:
        explanation = Explanation('NTU')
        print("✓ Explanation system loaded successfully")
    except Exception as e:
        print(f"✗ Error loading explanation system: {e}")
        print("Make sure the model files exist in the results directory")
        return

    # Load sample data
    print("\nLoading sample data...")
    try:
        samples = load_ntu_data(dataset='NTU', case=0, tag='ar', num_samples=10)
        print(f"✓ Loaded {len(samples)} samples")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Make sure the data files exist in the data directory")
        return

    # Generate static figures
    print("\n" + "="*60)
    print("GENERATING STATIC FIGURES")
    print("="*60 + "\n")

    figure_generators = [
        ('Teaser Figure', generate_teaser_figure, 'figures/teaser'),
        ('Skeleton Sensitivity', generate_skeleton_sensitivity_figure, 'figures/skeleton_sensitivity'),
        ('Skeleton Attribution', generate_skeleton_attribution_figure, 'figures/skeleton_attribution'),
        ('Pipeline Diagram', generate_pipeline_figure, 'figures/pipeline'),
        ('Process Diagram', generate_process_diagram, 'figures/process_diagram'),
        ('Heatmap Analysis', generate_heatmap_figure, 'figures/heatmap'),
        ('Before/After Masking', generate_before_after_masking_figure, 'figures/before_after_masking'),
        ('Noise Comparison', generate_noise_comparison_figure, 'figures/noise_comparison')
    ]

    for name, func, output_subdir in figure_generators:
        try:
            output_dir = os.path.join(base_dir, output_subdir)
            if 'Diagram' in name:
                func(output_dir)
            else:
                func(samples, explanation, output_dir)
            print(f"✓ {name} generated successfully")
        except Exception as e:
            print(f"✗ Error generating {name}: {e}")

    # Save skeleton file information
    try:
        info_output_dir = os.path.join(base_dir, 'figures')
        save_skeleton_file_info(samples, info_output_dir)
        print("✓ Skeleton file information saved successfully")
    except Exception as e:
        print(f"✗ Error saving skeleton file information: {e}")

    # Generate animations
    print("\n" + "="*60)
    print("GENERATING ANIMATIONS")
    print("="*60 + "\n")

    gif_output_dir = os.path.join(base_dir, 'output', 'gifs')

    animation_types = [
        'original',
        'sensitivity',
        'smart_masking',
        'smart_noise',
        'group_noise',
        'naive_noise'
    ]

    for anim_type in animation_types:
        try:
            create_skeleton_animation(samples, explanation, gif_output_dir, anim_type)
            print(f"✓ {anim_type.replace('_', ' ').title()} animation generated")
        except Exception as e:
            print(f"✗ Error generating {anim_type} animation: {e}")

    # Summary
    print("\n" + "="*60)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*60 + "\n")

    print(f"All outputs saved to: {base_dir}")
    print("\nGenerated files:")
    print("├── figures/")
    print("│   ├── teaser/teaser.png")
    print("│   ├── skeleton_sensitivity/skeleton_sensitivity.png")
    print("│   ├── skeleton_attribution/skeleton_attribution.png")
    print("│   ├── pipeline/pipeline.png")
    print("│   ├── process_diagram/process_diagram.png")
    print("│   ├── heatmap/heatmap.png")
    print("│   ├── before_after_masking/before_after_masking.png")
    print("│   ├── noise_comparison/noise_comparison.png")
    print("│   └── skeleton_file_info.txt")
    print("└── output/")
    print("    └── gifs/")
    print("        ├── original_animation.gif")
    print("        ├── sensitivity_animation.gif")
    print("        ├── smart_masking_animation.gif")
    print("        ├── smart_noise_animation.gif")
    print("        ├── group_noise_animation.gif")
    print("        └── naive_noise_animation.gif")

    print("\n✓ All visualizations generated successfully!")

if __name__ == "__main__":
    main()
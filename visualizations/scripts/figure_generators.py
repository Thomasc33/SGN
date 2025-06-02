"""
Figure generation functions for all visualization types
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from skeleton_utils import *
from data_utils import *
import os
from pathlib import Path
from data_utils import apply_smart_masking

def generate_teaser_figure(samples, explanation, out_dir):
    sample = samples[0]
    skel, lbl = sample[0], sample[1]  # Handle both old and new format
    resh      = reshape_skeleton(skel)
    bounds, *_ = calculate_global_bounds(resh)

    prot, _ = apply_smart_noise(skel, int(lbl), explanation, alpha=0.9, sigma=0.01)
    frm_o = extract_joints_from_frame(resh[0])
    frm_p = extract_joints_from_frame(reshape_skeleton(prot)[0])

    fig = plt.figure(figsize=(14, 4.2), facecolor="white")
    gs  = fig.add_gridspec(1, 7, width_ratios=[3, .3, 2, .3, 3, .4, .1])

    ax_l = fig.add_subplot(gs[0], projection="3d")
    draw_skeleton_3d(ax_l, frm_o, title="Original", fixed_bounds=bounds)

    fig.text(0.37, 0.50, "➜", fontsize=26, ha="center", va="center")

    ax_mid = fig.add_subplot(gs[2])
    ax_mid.axis("off")
    ax_mid.text(0.5, 0.68, "Privacy ↔ Utility", ha="center",
                fontsize=13, weight="bold")
    ax_mid.text(0.5, 0.43, "Re-identification ↓", ha="center",
                color="#DC2626", fontsize=10)
    ax_mid.text(0.5, 0.28, "Action recognition ↑", ha="center",
                color="#059669", fontsize=10)

    fig.text(0.63, 0.50, "➜", fontsize=26, ha="center", va="center")

    ax_r = fig.add_subplot(gs[4], projection="3d")
    draw_skeleton_3d(ax_r, frm_p, title="Protected (Smart Noise)", fixed_bounds=bounds)

    fig.suptitle("End-to-End Concept", y=0.97, fontsize=16, weight="bold")
    fig.savefig(Path(out_dir) / "teaser.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def generate_skeleton_sensitivity_figure(samples, explanation, out_dir):
    sample = samples[0]
    skel, lbl = sample[0], sample[1]  # Handle both old and new format
    lbl = int(lbl)
    resh = reshape_skeleton(skel)
    frm = extract_joints_from_frame(resh[0])
    importance = explanation.importance_score(resh, lbl, is_action=True)

    # normalise to [0, 1] for consistent colour-map
    imp_vals = np.array(list(importance.values()))
    imp_norm = (imp_vals - imp_vals.min()) / (imp_vals.ptp() + 1e-8)
    importance = {j: imp_norm[j] for j in range(25)}

    views = [(18, 35), (18, -35), (18, 145), (90, -90)]
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, wspace=0.02, hspace=0.02)

    for i, view in enumerate(views):
        ax = fig.add_subplot(gs[i // 2, i % 2], projection="3d")
        draw_skeleton_3d(ax, frm, importance_scores=importance,
                         title="", alpha=1.0, colormap="plasma")
        ax.view_init(*view)

    # colour-bar
    sm = plt.cm.ScalarMappable(cmap="plasma",
                               norm=mpl.colors.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=fig.axes, fraction=0.022, pad=0.02)
    cbar.set_label("Importance Score", rotation=90)

    fig.suptitle("Joint Sensitivity Analysis", weight="bold", fontsize=18,
                 y=0.98)
    fig.savefig(Path(out_dir) / "skeleton_sensitivity.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

def generate_skeleton_attribution_figure(samples, explanation, out_dir):
    """Generate skeleton visualization showing separate privacy and utility attribution scores"""
    sample = samples[0]
    skel, lbl = sample[0], sample[1]  # Handle both old and new format
    lbl = int(lbl)
    resh = reshape_skeleton(skel)
    frm = extract_joints_from_frame(resh[0])

    # Get separate attribution scores
    privacy_scores, utility_scores = explanation.attribution_scores(resh, lbl, is_action=True)

    # Normalize to [0, 1] for consistent color mapping
    privacy_vals = np.array(list(privacy_scores.values()))
    utility_vals = np.array(list(utility_scores.values()))

    privacy_norm = (privacy_vals - privacy_vals.min()) / (privacy_vals.ptp() + 1e-8)
    utility_norm = (utility_vals - utility_vals.min()) / (utility_vals.ptp() + 1e-8)

    privacy_scores_norm = {j: privacy_norm[j] for j in range(25)}
    utility_scores_norm = {j: utility_norm[j] for j in range(25)}

    # Use plasma colormap to match joint sensitivity analysis
    cmap = plt.cm.plasma

    views = [(18, 35), (18, -35), (18, 145), (90, -90)]
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 4, wspace=0.05, hspace=0.1)

    # Privacy attribution (top row)
    for i, view in enumerate(views):
        ax = fig.add_subplot(gs[0, i], projection="3d")
        draw_skeleton_3d(ax, frm, importance_scores=privacy_scores_norm,
                         title="", alpha=1.0, colormap=cmap)
        ax.view_init(*view)
        if i == 0:
            ax.text2D(0.02, 0.95, "Privacy Attribution", transform=ax.transAxes,
                     fontsize=12, weight='bold', va='top')

    # Utility attribution (bottom row)
    for i, view in enumerate(views):
        ax = fig.add_subplot(gs[1, i], projection="3d")
        draw_skeleton_3d(ax, frm, importance_scores=utility_scores_norm,
                         title="", alpha=1.0, colormap=cmap)
        ax.view_init(*view)
        if i == 0:
            ax.text2D(0.02, 0.95, "Utility Attribution", transform=ax.transAxes,
                     fontsize=12, weight='bold', va='top')

    # Add colorbars
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1))

    # Privacy colorbar
    cbar1 = fig.colorbar(sm, ax=fig.axes[:4], fraction=0.022, pad=0.02, shrink=0.8)
    cbar1.set_label("Privacy Attribution Score", rotation=90, color='black')
    cbar1.ax.tick_params(colors='black')

    # Utility colorbar
    cbar2 = fig.colorbar(sm, ax=fig.axes[4:8], fraction=0.022, pad=0.02, shrink=0.8)
    cbar2.set_label("Utility Attribution Score", rotation=90, color='black')
    cbar2.ax.tick_params(colors='black')

    fig.suptitle("Joint Attribution Analysis: Privacy vs Utility", weight="bold", fontsize=18, y=0.95)
    fig.savefig(Path(out_dir) / "skeleton_attribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def generate_pipeline_figure(out_dir):
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.axis("off")

    # blocks --------------------------------------------------------------
    boxes = {
        "Input\nSkeleton":      (0.05, 0.35, "#93C5FD"),
        "Smart\nProtection":    (0.35, 0.35, "#FDE047"),
        "Action\nRecognition":  (0.65, 0.6,  "#BBF7D0"),
        "Re-identification":    (0.65, 0.12, "#FCA5A5"),
    }
    for txt, (x, y, colour) in boxes.items():
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y), 0.25, 0.23, boxstyle="round,pad=0.02",
                fc=colour, ec="#374151", lw=1.6,
            )
        )
        ax.text(x + 0.125, y + 0.115, txt, ha="center", va="center",
                weight="bold", fontsize=11)

    # arrows --------------------------------------------------------------
    def arrow(xy1, xy2, color):
        ax.annotate(
            "", xytext=xy1, xy=xy2,
            arrowprops=dict(arrowstyle="->", lw=2, color=color),
        )
    arrow((0.30, 0.46), (0.35, 0.46), "#374151")      # forward
    arrow((0.60, 0.57), (0.65, 0.67), "#059669")      # up
    arrow((0.60, 0.46), (0.65, 0.24), "#DC2626")      # down

    ax.set_title("Privacy-Preserving Skeleton Analysis – Data Flow",
                 fontsize=14, weight="bold", pad=15)
    fig.savefig(Path(out_dir) / "pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def generate_process_diagram(out_dir):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")

    colours = ["#93C5FD", "#FEF08A", "#BBF7D0"]
    labels  = [
        ("Step 1\nTrain models",
         "Utility: Action-Rec\nThreat: Re-ID"),
        ("Step 2\nScore joints",
         "Integrated Gradients\nprivacy ⊕ utility"),
        ("Step 3\nApply protection",
         "Mask / noise β ≈ 20 %"),
    ]
    x0 = 0.05
    for i, (title, desc) in enumerate(labels):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x0 + i * 0.32, 0.35), 0.28, 0.28,
                boxstyle="round,pad=0.02", fc=colours[i],
                ec="#374151", lw=1.6,
            )
        )
        ax.text(x0 + i * 0.32 + 0.14, 0.54, title, ha="center",
                va="center", weight="bold", fontsize=11)
        ax.text(x0 + i * 0.32 + 0.14, 0.43, desc, ha="center",
                va="center", fontsize=9, linespacing=1.3)

        # arrows except after last box
        if i < 2:
            arrow_x = x0 + i * 0.32 + 0.28
            ax.annotate("", xytext=(arrow_x, 0.49),
                        xy=(arrow_x + 0.04, 0.49),
                        arrowprops=dict(arrowstyle="->", lw=2,
                                        color="#374151"))

    ax.set_title("Three-Step Protection Pipeline", fontsize=14,
                 weight="bold", pad=15)
    fig.savefig(Path(out_dir) / "process_diagram.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

def generate_heatmap_figure(samples, explanation, output_dir):
    """Generate heatmap visualization of joint importance"""
    print("Generating heatmap figure...")

    # Collect importance scores for multiple samples
    all_importance = []
    sample_labels = []

    for i, sample in enumerate(samples[:5]):  # Use first 5 samples
        skeleton, label = sample[0], sample[1]  # Handle both old and new format
        reshaped = reshape_skeleton(skeleton)

        # Convert label to int if needed
        if isinstance(label, np.ndarray):
            label = int(label.item())
        elif isinstance(label, (np.int64, np.int32)):
            label = int(label)

        importance = explanation.importance_score(reshaped, label, is_action=True, alpha=0.9)
        importance_values = [importance[j] for j in range(25)]
        all_importance.append(importance_values)
        sample_labels.append(f'Sample {i+1}')

    # Use plasma colormap to match joint sensitivity analysis
    cmap = plt.cm.plasma

    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    importance_matrix = np.array(all_importance)
    joint_names = [f'Joint {i}' for i in range(25)]

    sns.heatmap(importance_matrix,
                xticklabels=joint_names,
                yticklabels=sample_labels,
                annot=False,  # Remove numbers from heatmap
                cmap=cmap,    # Use plasma colormap
                cbar_kws={'label': 'Importance Score'},
                ax=ax)

    # Set text colors to black
    ax.set_title('Joint Importance Heatmap Across Samples', fontsize=14, weight='bold', color='black')
    ax.set_xlabel('Skeleton Joints', fontsize=12, color='black')
    ax.set_ylabel('Samples', fontsize=12, color='black')

    # Set tick label colors to black
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    # Set colorbar label color to black
    cbar = ax.collections[0].colorbar
    cbar.set_label('Importance Score', color='black')
    cbar.ax.tick_params(colors='black')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_before_after_masking_figure(samples, explanation, out_dir, beta=0.2):
    sample = samples[0]
    skel, lbl = sample[0], sample[1]  # Handle both old and new format
    resh = reshape_skeleton(skel)
    frames = [0, 5, 10]                   # three evenly-spaced frames

    masked, masked_joints, _ = apply_smart_masking(
        skel, lbl, explanation, beta=beta)
    mresh = reshape_skeleton(masked)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8),
                             subplot_kw={"projection": "3d"})
    fig.patch.set_facecolor("white")

    for col, fidx in enumerate(frames):
        f_orig = extract_joints_from_frame(resh[fidx])
        f_mask = extract_joints_from_frame(mresh[fidx])

        draw_skeleton_3d(axes[0, col], f_orig, title=f"Frame {fidx + 1}")
        draw_skeleton_3d(axes[1, col], f_mask,
                         masked_joints=masked_joints, title="")

        axes[1, col].set_title(f"Masked (β={beta})", pad=10)

    fig.suptitle("Before and After Smart Masking", fontsize=18, weight="bold",
                 y=0.94)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "before_after_masking.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

def generate_noise_comparison_figure(samples, explanation, output_dir):
    """Generate stunning comparison of smart noise, group noise, and random noise"""
    print("Generating noise comparison figure...")

    sample = samples[0]
    skeleton, label = sample[0], sample[1]  # Handle both old and new format
    reshaped = reshape_skeleton(skeleton)

    # Apply different noise types
    smart_noisy, smart_importance = apply_smart_noise(skeleton, label, explanation, sigma=0.01)
    group_noisy, sensitive_joints, group_importance = apply_group_noise(skeleton, label, explanation, beta=0.2, sigma=0.01, tag='ar')
    naive_noisy = apply_naive_noise(skeleton, sigma=0.01)

    # Calculate global bounds for consistent scaling
    bounds_3d, _, _ = calculate_global_bounds(reshaped)

    fig = plt.figure(figsize=(24, 12), facecolor='white')

    frame_idx = 0  # Use first frame

    # Original
    ax1 = fig.add_subplot(2, 4, 1, projection='3d', facecolor='white')
    original_joints = extract_joints_from_frame(reshaped[frame_idx])
    draw_skeleton_3d(ax1, original_joints, title='Original', fixed_bounds=bounds_3d)

    # Smart noise
    ax2 = fig.add_subplot(2, 4, 2, projection='3d', facecolor='white')
    smart_joints = extract_joints_from_frame(reshape_skeleton(smart_noisy)[frame_idx])
    draw_skeleton_3d(ax2, smart_joints, importance_scores=smart_importance,
                    title='Smart Noise', colormap='plasma', fixed_bounds=bounds_3d)

    # Group noise
    ax3 = fig.add_subplot(2, 4, 3, projection='3d', facecolor='white')
    group_joints = extract_joints_from_frame(reshape_skeleton(group_noisy)[frame_idx])
    draw_skeleton_3d(ax3, group_joints, masked_joints=sensitive_joints,
                    title='Group Noise', fixed_bounds=bounds_3d)

    # Naive noise
    ax4 = fig.add_subplot(2, 4, 4, projection='3d', facecolor='white')
    naive_joints = extract_joints_from_frame(reshape_skeleton(naive_noisy)[frame_idx])
    draw_skeleton_3d(ax4, naive_joints, title='Random Noise', fixed_bounds=bounds_3d)

    # Second row - different viewing angles
    ax5 = fig.add_subplot(2, 4, 5, projection='3d', facecolor='white')
    draw_skeleton_3d(ax5, original_joints, title='Original (Alt View)', fixed_bounds=bounds_3d)
    ax5.view_init(elev=10, azim=-45)

    ax6 = fig.add_subplot(2, 4, 6, projection='3d', facecolor='white')
    draw_skeleton_3d(ax6, smart_joints, importance_scores=smart_importance,
                    title='Smart Noise (Alt View)', colormap='plasma', fixed_bounds=bounds_3d)
    ax6.view_init(elev=10, azim=-45)

    ax7 = fig.add_subplot(2, 4, 7, projection='3d', facecolor='white')
    draw_skeleton_3d(ax7, group_joints, masked_joints=sensitive_joints,
                    title='Group Noise (Alt View)', fixed_bounds=bounds_3d)
    ax7.view_init(elev=10, azim=-45)

    ax8 = fig.add_subplot(2, 4, 8, projection='3d', facecolor='white')
    draw_skeleton_3d(ax8, naive_joints, title='Random Noise (Alt View)', fixed_bounds=bounds_3d)
    ax8.view_init(elev=10, azim=-45)

    fig.suptitle('Noise Application Comparison', fontsize=20, weight='bold', color='black', y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'noise_comparison.png'), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def save_skeleton_file_info(samples, output_dir):
    """
    Save skeleton file information to a text file.
    Now includes actor ID and original skeleton filename information.
    """
    info_file = os.path.join(output_dir, 'skeleton_file_info.txt')

    with open(info_file, 'w') as f:
        f.write("Skeleton File Information\n")
        f.write("=" * 50 + "\n\n")
        f.write("Skeleton files used for all visualizations:\n\n")
        f.write("Indexing: All IDs are 0-indexed (starting from 0)\n\n")

        for i, sample in enumerate(samples[:5]):
            if len(sample) == 3:
                skeleton, label, metadata = sample
                if metadata:
                    # Convert label to int if needed
                    if isinstance(label, np.ndarray):
                        label_int = int(label.item())
                    elif isinstance(label, (np.int64, np.int32)):
                        label_int = int(label)
                    else:
                        label_int = int(label)

                    f.write(f"Sample {i+1}:\n")
                    f.write(f"  Original filename: {metadata['filename']}.skeleton\n")
                    f.write(f"  Actor ID: {metadata['actor_id']} (0-indexed, original: P{metadata['actor_id']+1:03d})\n")
                    f.write(f"  Action ID: {metadata['action_id']} (0-indexed, original: A{metadata['action_id']+1:03d})\n")
                    f.write(f"  Label from h5: {label_int}\n")

                    # Parse additional info from filename
                    filename = metadata['filename']
                    setup_id = int(filename[1:4])
                    camera_id = int(filename[5:8])
                    replication_id = int(filename[13:16])

                    f.write(f"  Setup ID: {setup_id-1} (0-indexed, original: S{setup_id:03d})\n")
                    f.write(f"  Camera ID: {camera_id-1} (0-indexed, original: C{camera_id:03d})\n")
                    f.write(f"  Replication ID: {replication_id-1} (0-indexed, original: R{replication_id:03d})\n\n")
                else:
                    f.write(f"Sample {i+1}: Metadata not available\n\n")
            else:
                # Handle old format for backward compatibility
                skeleton, label = sample
                if isinstance(label, np.ndarray):
                    label_int = int(label.item())
                elif isinstance(label, (np.int64, np.int32)):
                    label_int = int(label)
                else:
                    label_int = int(label)

                f.write(f"Sample {i+1}:\n")
                f.write(f"  Action ID: {label_int} (0-indexed)\n")
                f.write(f"  Note: Actor ID not available in current data structure\n\n")

        f.write("Legend:\n")
        f.write("  S = Setup ID (recording setup)\n")
        f.write("  C = Camera ID (camera viewpoint)\n")
        f.write("  P = Performer/Actor ID (person performing the action)\n")
        f.write("  R = Replication ID (repetition number)\n")
        f.write("  A = Action ID (action class)\n\n")
        f.write("Note: In the original NTU dataset, IDs in filenames are 1-indexed,\n")
        f.write("but in the processed data they are converted to 0-indexed for ML purposes.\n\n")
        f.write("Privacy Attribution Analysis:\n")
        f.write("The Actor ID is crucial for privacy attribution analysis as it represents\n")
        f.write("the identity that the re-identification model tries to predict.\n")

    print(f"Skeleton file information saved to: {info_file}")

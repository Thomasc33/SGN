"""
Skeleton visualization utilities for NTU RGB+D dataset - IMPROVED VERSION
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


# NTU RGB+D 25-joint skeleton structure
NTU_JOINTS = {
    0: "Base of spine",
    1: "Middle of spine", 
    2: "Neck",
    3: "Head",
    4: "Left shoulder",
    5: "Left elbow",
    6: "Left wrist",
    7: "Left hand",
    8: "Right shoulder",
    9: "Right elbow",
    10: "Right wrist",
    11: "Right hand",
    12: "Left hip",
    13: "Left knee",
    14: "Left ankle",
    15: "Left foot",
    16: "Right hip",
    17: "Right knee",
    18: "Right ankle",
    19: "Right foot",
    20: "Spine",
    21: "Left hand tip",
    22: "Left thumb",
    23: "Right hand tip",
    24: "Right thumb"
}

# Skeleton connections (bone structure)
SKELETON_CONNECTIONS = [
    (0, 1), (1, 20), (20, 2), (2, 3),  # Spine to head
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),  # Right arm
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),  # Left arm
    (0, 16), (16, 17), (17, 18), (18, 19),  # Right leg
    (0, 12), (12, 13), (13, 14), (14, 15)  # Left leg
]
def _prep_axis(ax):
    """Turn a Matplotlib 3-D axis into a neutral, grid-free canvas."""
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill       = False
        axis.pane.set_edgecolor("white")
        axis._axinfo["grid"]["linewidth"] = 0
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1.5))

mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.edgecolor":   "#E5E7EB",
    "axes.labelcolor":  "#6B7280",
    "axes.grid":        False,
    "grid.color":       "#E5E7EB",
    "xtick.color":      "#6B7280",
    "ytick.color":      "#6B7280",
    "font.family":      ["DejaVu Sans", "sans-serif"],
})

_BASE_BLUE  = "#2563EB"   # joints & bones when importance==None
_MASK_RED   = "#EF4444"
_BONE_GREY  = "#4B5563"
_COLORMAP   = plt.cm.magma  # perceptually uniform, no neon artefacts

def create_skeleton_figure(figsize=(10, 10)):
    """Create a figure for skeleton visualization with white background"""
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    return fig, ax

def draw_skeleton_3d(
        ax, joints,
        importance_scores=None,
        masked_joints=None,
        title="Skeleton",
        alpha=1.0,
        fixed_bounds=None,
        colormap=None,          # <- NEW (keeps old callers happy)
        **kwargs                # <- NEW (swallows any other legacy arg)
):
    """
    Render a 3-D NTU skeleton with a consistent style.
    – No grids or panes.
    – Deterministic colour scheme.
    – Optional importance colouring & masked-joint highlighting.
    """
    global _COLORMAP
    if colormap is not None:
        _COLORMAP = plt.cm.get_cmap(colormap)

    _prep_axis(ax)
    joints = joints.copy()

    # Re-orient → (X=left/right, Z=height, Y=depth)
    joints = joints[:, [0, 2, 1]]
    nonzero = ~np.all(joints == 0, axis=1)
    if nonzero.any():
        joints[nonzero] -= joints[nonzero].mean(0)

    # ----- colours ----------------------------------------------------------
    if importance_scores:
        imp = np.array([importance_scores.get(i, 0.0) for i in range(25)])
        norm = mcolors.Normalize(vmin=float(imp.min()), vmax=float(imp.max()) or 1)
        joint_cols = _COLORMAP(norm(imp))
        bone_colour = lambda a, b: _COLORMAP(norm((imp[a] + imp[b]) / 2))
    else:
        joint_cols = [_BASE_BLUE] * 25
        bone_colour = lambda *args: _BONE_GREY

    # ----- bones ------------------------------------------------------------
    for a, b in SKELETON_CONNECTIONS:
        if nonzero[a] and nonzero[b]:
            ax.plot(
                joints[[a, b], 0], joints[[a, b], 1], joints[[a, b], 2],
                color=bone_colour(a, b), lw=2.2, alpha=0.85,
                solid_capstyle="round",
            )

    # ----- joints -----------------------------------------------------------
    for idx in range(25):
        if not nonzero[idx]:
            continue
        if masked_joints and idx in masked_joints:
            ax.scatter(
                *joints[idx], s=130, marker="X",
                c=_MASK_RED, ec="#7F1D1D", lw=2.2, zorder=5,
            )
        else:
            ax.scatter(
                *joints[idx],
                s=70 if importance_scores is None else 50 + imp[idx] * 120,
                c=[joint_cols[idx]], ec="#1F2937", lw=1.3,
                alpha=alpha, zorder=4,
            )

    # ----- bounds / camera --------------------------------------------------
    if fixed_bounds:
        x_rng, y_rng, z_rng = fixed_bounds
    else:
        pad = 0.35
        mins = joints[nonzero].min(0) - pad
        maxs = joints[nonzero].max(0) + pad
        x_rng, y_rng, z_rng = (mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2])
    ax.set_xlim(*x_rng); ax.set_ylim(*y_rng); ax.set_zlim(*z_rng)
    ax.view_init(elev=18, azim=35)
    ax.set_title(title, fontsize=13, weight="bold", pad=14)
    
def draw_skeleton_multi_view(fig, joints, importance_scores=None, masked_joints=None,
                           title="Skeleton", colormap='plasma', fixed_bounds=None):
    """
    Draw multiple 3D views of the skeleton in a single figure
    """
    fig.clear()
    fig.patch.set_facecolor('white')
    
    # Create multiple 3D subplots with different viewing angles
    views = [
        {'elev': 15, 'azim': 45, 'title': 'Front-Right View'},
        {'elev': 15, 'azim': -45, 'title': 'Front-Left View'},
        {'elev': 15, 'azim': 135, 'title': 'Back-Right View'},
        {'elev': 90, 'azim': 0, 'title': 'Top View'},
    ]
    
    for i, view in enumerate(views):
        ax = fig.add_subplot(2, 2, i+1, projection='3d', facecolor='white')
        
        draw_skeleton_3d(ax, joints, importance_scores=importance_scores,
                        masked_joints=masked_joints, title=view['title'],
                        colormap=colormap, alpha=1.0, fixed_bounds=fixed_bounds)
        
        ax.view_init(elev=view['elev'], azim=view['azim'])
    
    fig.suptitle(title, fontsize=18, weight='bold', color='#1f2937', y=0.98)
    plt.tight_layout()

def calculate_global_bounds(skeleton_sequence, padding=0.3):
    """
    Calculate global bounds for consistent skeleton visualization across frames
    """
    # Take only first person data (first 75 features)
    if skeleton_sequence.shape[1] > 75:
        skeleton_sequence = skeleton_sequence[:, :75]
    
    # Collect all valid joints with reorientation
    all_joints = []
    for frame in skeleton_sequence:
        joints = frame.reshape(25, 3)
        # Apply same reorientation as in draw_skeleton_3d
        joints_reoriented = np.zeros_like(joints)
        joints_reoriented[:, 0] = joints[:, 0]  # X stays the same
        joints_reoriented[:, 1] = joints[:, 2]  # Y becomes depth
        joints_reoriented[:, 2] = joints[:, 1]  # Z becomes height
        
        # Only include non-zero joints
        valid_joints = joints_reoriented[~np.all(joints_reoriented == 0, axis=1)]
        if len(valid_joints) > 0:
            # Center the joints
            valid_joints -= valid_joints.mean(axis=0)
            all_joints.append(valid_joints)
    
    if not all_joints:
        # Default bounds if no valid joints
        return ([-1, 1], [-1, 1], [-1, 1]), ([-1, 1], [-1, 1]), ([-1, 1], [-1, 1])
    
    all_joints = np.vstack(all_joints)
    
    # Calculate bounds with padding
    x_min, x_max = all_joints[:, 0].min(), all_joints[:, 0].max()
    y_min, y_max = all_joints[:, 1].min(), all_joints[:, 1].max()
    z_min, z_max = all_joints[:, 2].min(), all_joints[:, 2].max()
    
    # Add padding
    x_range = [x_min - padding, x_max + padding]
    y_range = [y_min - padding, y_max + padding]
    z_range = [z_min - padding, z_max + padding]
    
    # Ensure minimum range
    for r in [x_range, y_range, z_range]:
        if r[1] - r[0] < 1.0:
            mid = (r[0] + r[1]) / 2
            r[0] = mid - 0.5
            r[1] = mid + 0.5
    
    bounds_3d = (x_range, y_range, z_range)
    bounds_2d_front = (x_range, z_range)  # X-Z plane for front view
    bounds_2d_side = (y_range, z_range)   # Y-Z plane for side view
    
    return bounds_3d, bounds_2d_front, bounds_2d_side

def create_importance_colorbar(fig, importance_scores, colormap='plasma'):
    """Add a colorbar for importance scores with better positioning"""
    scores = np.array(list(importance_scores.values()))
    if scores.max() - scores.min() > 0:
        norm = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())
        cmap = plt.cm.get_cmap(colormap)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Create colorbar with better positioning
        cbar = fig.colorbar(sm, ax=fig.get_axes(), pad=0.02, fraction=0.02, shrink=0.8)
        cbar.set_label('Importance Score', color='#1f2937', fontsize=12, weight='bold')
        cbar.ax.tick_params(colors='#4b5563', labelsize=10)
        cbar.outline.set_edgecolor('#9ca3af')
        cbar.outline.set_linewidth(1)
        
        return cbar
    return None
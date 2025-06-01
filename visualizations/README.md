# Privacy-Preserving Skeleton Analysis Visualizations

This directory contains scripts to generate all visualizations for the presentation on privacy-preserving skeleton analysis using integrated gradients.

## Overview

The visualization system generates the following figures:

1. **Teaser Image**: Visual showing skeleton data → privacy risks vs utility
2. **Skeleton Sensitivity**: Skeleton with joints colored by sensitivity scores
3. **Pipeline Diagram**: Process flow showing skeleton → check/x → AR and privacy attacks
4. **Process Diagram**: 3-step process (Train Models → Identify Sensitive Joints → Apply Protection)
5. **Heatmap**: Joint importance visualization across multiple samples
6. **Before/After Masking**: Comparison with β=0.2 masking
7. **Noise Comparison**: Smart noise vs group noise vs random noise
8. **Animations**: GIF sequences showing skeleton motion with different processing

## Directory Structure

```
visualizations/
├── scripts/
│   ├── generate_all_figures.py    # Main orchestration script
│   ├── skeleton_utils.py          # Skeleton drawing utilities
│   ├── data_utils.py              # Data loading and processing
│   └── figure_generators.py       # Individual figure generation functions
├── figures/                       # Generated static images
│   ├── teaser/
│   ├── skeleton_sensitivity/
│   ├── pipeline/
│   ├── process_diagram/
│   ├── heatmap/
│   ├── before_after_masking/
│   └── noise_comparison/
├── output/                        # Generated animations and other outputs
│   ├── images/
│   └── gifs/
├── run_visualizations.py          # Simple runner script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Prerequisites

1. **Data Files**: Ensure the following data files exist:
   - `./data/ntu/NTU_CS_ar.h5`
   - `./data/ntu/NTU_CS_ri.h5`

2. **Model Files**: Ensure the trained models exist:
   - `results/NTUar/SGN/1_best.pth`
   - `results/NTUri/SGN/1_best.pth`

3. **Python Dependencies**: Install required packages:
   ```bash
   pip install -r visualizations/requirements.txt
   ```

## Usage

### Quick Start

Run all visualizations with a single command:

```bash
python visualizations/run_visualizations.py
```

### Advanced Usage

You can also run the main script directly:

```bash
cd visualizations/scripts
python generate_all_figures.py
```

### Individual Figure Generation

To generate specific figures, you can modify the main script or import individual functions:

```python
from figure_generators import generate_teaser_figure
from data_utils import load_ntu_data
from explanation import Explanation

# Load data and explanation system
explanation = Explanation('NTU')
samples = load_ntu_data(dataset='NTU', case=0, tag='ar', num_samples=5)

# Generate specific figure
generate_teaser_figure(samples, explanation, './output')
```

## Generated Outputs

After running the visualization script, you will find:

### Static Images (PNG format, 300 DPI)
- `figures/teaser/teaser.png`
- `figures/skeleton_sensitivity/skeleton_sensitivity.png`
- `figures/pipeline/pipeline.png`
- `figures/process_diagram/process_diagram.png`
- `figures/heatmap/heatmap.png`
- `figures/before_after_masking/before_after_masking.png`
- `figures/noise_comparison/noise_comparison.png`

### Animations (GIF format)
- `output/gifs/original_animation.gif`
- `output/gifs/sensitivity_animation.gif`
- `output/gifs/masked_animation.gif`

## Customization

### Parameters

You can modify key parameters in the scripts:

- **Sample size**: Change `num_samples` in `load_ntu_data()`
- **Masking percentage**: Modify `beta` parameter (default: 0.2)
- **Noise level**: Adjust `sigma` parameter (default: 0.01)
- **Alpha weighting**: Change `alpha` for utility/privacy balance (default: 0.9)

### Styling

- **Color schemes**: Modify `colormap` parameters in drawing functions
- **Figure sizes**: Adjust `figsize` parameters
- **DPI**: Change `dpi` parameter in `plt.savefig()`

### Adding New Visualizations

1. Create a new function in `figure_generators.py`
2. Add the function call to `generate_all_figures.py`
3. Create corresponding output directory

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure NTU dataset files are in the correct location
2. **Missing model files**: Check that trained models exist in results directory
3. **Memory issues**: Reduce number of samples or figure resolution
4. **Import errors**: Verify all dependencies are installed

### Error Messages

- `FileNotFoundError`: Check data and model file paths
- `CUDA errors`: Ensure GPU is available or modify code to use CPU
- `Import errors`: Install missing dependencies with pip

## Technical Details

### Skeleton Structure

The system uses the NTU RGB+D 25-joint skeleton structure:
- 25 joints per person
- 3D coordinates (x, y, z) per joint
- Support for up to 2 people per frame
- 20 frames per sequence (downsampled from original)

### Importance Scoring

Uses Integrated Gradients to compute joint importance:
```
Importance = Privacy_Risk + α × (1 - Utility_Importance)
```

Where:
- `Privacy_Risk`: Gradient attribution for re-identification model
- `Utility_Importance`: Gradient attribution for action recognition model
- `α`: Weighting parameter balancing privacy vs utility

### Protection Methods

1. **Smart Masking**: Zero out top β% most sensitive joints
2. **Smart Noise**: Add noise inversely proportional to importance
3. **Group Noise**: Different noise levels for sensitive vs non-sensitive groups
4. **Naive Noise**: Random noise applied uniformly

## Citation

If you use these visualizations in your research, please cite the original paper and this visualization framework.

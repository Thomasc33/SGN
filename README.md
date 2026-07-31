# XAnon — Explanation-based Anonymization Methods for Motion Privacy

[![PAKDD 2025](https://img.shields.io/badge/PAKDD-2025-A21CAF.svg)](https://doi.org/10.1007/978-981-96-8183-9_5)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--981--96--8183--9__5-blue.svg)](https://doi.org/10.1007/978-981-96-8183-9_5)
[![Project page](https://img.shields.io/badge/project-explanation.thomasc.tech-6D28D9.svg)](https://explanation.thomasc.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Use explainable AI to find the joints that leak identity, then spend the masking or
> differential-privacy budget only on those joints.

**[Project page](https://explanation.thomasc.tech)** · **[Paper (Springer)](https://doi.org/10.1007/978-981-96-8183-9_5)** · **[Pre-trained weights](https://github.com/Thomasc33/SGN/releases)**

---

## Overview

Skeleton data looks anonymous — no faces, no pixels, just 25 joint coordinates per frame. But the
*way* a person moves is a biometric, and a re-identification model reaches 81.6% top-1 accuracy on
raw NTU60 sequences.

Existing defences treat every joint as equally guilty. This work does not. Integrated Gradients is
run against two models — a utility model $f^U$ trained for action recognition and a threat model
$f^P$ trained for re-identification — and the disagreement between their attributions becomes a
per-joint sensitivity score. That score then drives one of three protection mechanisms.

**Contributions:**

- An XAI-based method to quantify the privacy sensitivity of individual skeleton joints.
- An anonymization framework that applies masking or differentially private noise selectively,
  per joint, based on that sensitivity.
- Evaluation on NTU RGB+D 60 and 120 showing large re-identification drops at modest utility cost.

## Results

All numbers are top-1 accuracy on NTU RGB+D. **AR** is action recognition (utility, higher is
better); **RI** is re-identification (privacy, lower is better).

### Smart Masking — α = 0.9, β = 0.2

| Dataset | Task | Non-private | Consumer VR | **Smart Masking** |
|---------|------|-------------|-------------|-------------------|
| NTU60 | AR ↑ | 94.72% | 90.82% | **81.23%** |
| NTU60 | RI ↓ | 81.58% | 58.97% | **21.82%** |
| NTU120 | AR ↑ | 89.17% | 83.57% | **60.15%** |
| NTU120 | RI ↓ | 77.69% | 47.49% | **8.21%** |

### Differential privacy — σ = 0.01

| Dataset | Task | Non-private | Naïve Noise | **Group Noise** | **Smart Noise** |
|---------|------|-------------|-------------|-----------------|-----------------|
| NTU60 | AR ↑ | 94.72% | 90.04% | 72.20% | **79.97%** |
| NTU60 | RI ↓ | 81.58% | 76.08% | **12.70%** | 28.24% |
| NTU120 | AR ↑ | 89.17% | 78.35% | 51.67% | **54.23%** |
| NTU120 | RI ↓ | 77.69% | 62.84% | **7.12%** | 22.13% |

Naïve Noise and Group Noise consume the *identical* total privacy budget. The entire 63-point gap
in re-identification comes from where the noise was allocated.

## Method

```
skeleton s ∈ R^(T×J×C)
        │
        ├──► f^U (action recognition) ──► IG ──► φ^U_j   utility attribution
        └──► f^P (re-identification)  ──► IG ──► φ^P_j   privacy attribution
                                                  │
                          ψ_j = φ̄^P_j + α(1 − φ̄^U_j)   combined sensitivity
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
              Smart Masking                  Group Noise                   Smart Noise
        zero the top-β joints        two ε budgets, ratio γ          per-joint ε ∝ 1/ψ_j
```

**Joint attribution.** Absolute Integrated Gradients attributions are averaged over all frames and
coordinate channels, collapsing a `T × J × C` tensor into one scalar per joint per model.

**Sensitivity score.** `ψ_j = φ̄^P_j + α(1 − φ̄^U_j)`. A joint is sensitive when it is informative
for identity *and* expendable for the action. `α = 0.9` is optimal across every experiment.

**Smart Masking.** Zero all coordinates of the top-`β` fraction of joints. Cheapest and strongest
privacy-per-utility, but produces visibly collapsed skeletons.

**Group Noise.** Split joints into sensitive (`G_s`, top-β) and non-sensitive (`G_n`) groups and
give each its own privacy budget, with `ε_s = γ·ε_n`. Strongest privacy of the three.

**Smart Noise.** Give every joint its own budget `ε_j ∝ 1/ψ_j`, min-max normalised into `[0.01, 1]`.
Best-balanced operating point, and keeps motion continuous.

## Installation

```bash
git clone https://github.com/Thomasc33/SGN.git
cd SGN
pip install -r requirements.txt
```

## Data preparation

Download [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) and extract to
`./data/ntu/nturgb+d_skeletons/`, then:

```bash
cd data/ntu
python get_raw_skes_data.py      # per-performer skeletons
python get_raw_denoised_data.py  # drop bad / multi-actor skeletons
python seq_transformation.py     # centre on first frame, write .h5
```

NTU120 and ETRI follow the same three-script flow under `data/ntu120/` and `data/etri/`.

## Pre-trained weights

Checkpoints for every dataset/task combination are published as a
[GitHub release](https://github.com/Thomasc33/SGN/releases) rather than tracked in git.

```bash
gh release download weights-v1 --repo Thomasc33/SGN --dir results --pattern '*.tar.gz'
tar -xzf results/checkpoints.tar.gz -C results
```

This restores the `results/` tree that `explanation.py` and `main.py --load-dir` expect:

```
results/NTUar/SGN/{0,1}_best.pth      NTU60  action recognition   (case 0 = CS, 1 = CV)
results/NTUri/SGN/1_best.pth          NTU60  re-identification
results/NTUgc/SGN/{0,1}_best.pth      NTU60  gender classification
results/NTU120{ar,ri,gc}/SGN/*.pth    NTU120 equivalents
results/ETRI{ar,ri}/SGN/*.pth         ETRI   equivalents
```

## Training

```bash
# case 0 = cross-subject, case 1 = cross-view
# tag  ar = action recognition (utility model f^U)
#      ri = re-identification  (threat model f^P)
#      gc = gender classification

python main.py --network SGN --train 1 --case 0 --dataset NTU --tag ar
python main.py --network SGN --train 1 --case 0 --dataset NTU --tag ri
```

Both models must exist before any protection mechanism can run — `explanation.py` loads the `ar`
and `ri` checkpoints to compute attributions.

## Evaluation

```bash
# Smart Masking — headline result (AR 81.23 / RI 21.82)
python main.py --network SGN --train 0 --case 1 --dataset NTU --tag ar \
    --load-dir results/NTUar --smart-masking 1 --alpha 0.9 --beta 0.2

# Smart Noise — best balance (AR 79.97 / RI 28.24)
python main.py --network SGN --train 0 --case 1 --dataset NTU --tag ar \
    --load-dir results/NTUar --smart-noise 1 --alpha 0.9 --sigma 0.01

# Group Noise — strongest privacy (AR 72.20 / RI 12.70)
python main.py --network SGN --train 0 --case 1 --dataset NTU --tag ar \
    --load-dir results/NTUar --group-noise 1 --alpha 0.9 --beta 0.2 --sigma 0.01

# Naïve Noise baseline
python main.py --network SGN --train 0 --case 1 --dataset NTU --tag ar \
    --load-dir results/NTUar --naive-noise 1 --sigma 0.01

# Consumer VR baseline — keep only head and both hands
python main.py --network SGN --train 0 --case 1 --dataset NTU --tag ar \
    --load-dir results/NTUar \
    --mask 0 1 2 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 24
```

Swap `--tag ar` for `--tag ri` and `--load-dir results/NTUri` to measure the privacy side.

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--dataset` | `NTU` | `NTU`, `NTU120` or `ETRI` |
| `--case` | `0` | `0` = cross-subject, `1` = cross-view |
| `--tag` | `ar` | `ar`, `ri` or `gc` — selects the label set |
| `--train` | — | `1` to train, `0` to evaluate |
| `--load-dir` | `None` | Checkpoint directory for evaluation |
| `--mask` | `[]` | Explicit 0-indexed joint list to zero |
| `--smart-masking` | `0` | Enable Smart Masking |
| `--group-noise` | `0` | Enable Group Noise |
| `--smart-noise` | `0` | Enable Smart Noise |
| `--naive-noise` | `0` | Enable uniform Gaussian baseline |
| `--alpha` | `0.9` | Utility weighting in ψ |
| `--beta` | `0.2` | Fraction of joints treated as sensitive |
| `--sigma` | `0.01` | Noise magnitude |

The four mechanism flags are mutually exclusive. The Group Noise ratio **γ is hardcoded to 0.03**
in `data.py` (in the `group_noise` branch) — change it there to reproduce the γ sweep in Table 5.

## Visualizations

Every figure on the project page is generated from the trained NTU60 models:

```bash
pip install -r visualizations/requirements.txt
python visualizations/run_visualizations.py
```

Outputs land in `visualizations/figures/` and `visualizations/output/gifs/`. Requires
`data/ntu/NTU_CS_ar.h5`, `data/ntu/NTU_CS_ri.h5` and the `NTUar`/`NTUri` checkpoints.
See [`visualizations/README.md`](visualizations/README.md) for details.

## Repository layout

```
index.html              Project page (GitHub Pages → explanation.thomasc.tech)
fig/                    Web-optimized figures and animations for the project page
main.py                 Training and evaluation entry point
fit.py                  Argument definitions
model.py                SGN backbone
data.py                 Dataset loading; all four protection mechanisms live here
explanation.py          Integrated Gradients attribution and ψ_j scoring
util.py                 Helpers
data/{ntu,ntu120,etri}/ Dataset preprocessing scripts
visualizations/         Figure and animation generators
Explanation Testing/    Exploratory notebooks (Captum, LIME, sensitivity)
commands.txt            Command reference
```

## Citation

```bibtex
@inproceedings{carr2025explanation,
  title     = {Explanation-Based Anonymization Methods for Motion Privacy},
  author    = {Carr, Thomas and Zhao, Yaxin and Xu, Depeng and Lu, Aidong},
  booktitle = {Advances in Knowledge Discovery and Data Mining (PAKDD)},
  series    = {Lecture Notes in Computer Science},
  volume    = {15873},
  pages     = {52--64},
  publisher = {Springer Nature Singapore},
  address   = {Singapore},
  year      = {2025},
  doi       = {10.1007/978-981-96-8183-9_5}
}
```

## Related work

| Project | Venue | Page |
|---------|-------|------|
| PMR — Privacy-centric deep motion retargeting | ICCV 2025 | [pmr.thomasc.tech](https://pmr.thomasc.tech) |
| DisentangledTMR — Factorized-transformer retargeting | ECCV 2026 | [tmr.thomasc.tech](https://tmr.thomasc.tech) |
| LAN — Linkage attack on skeleton motion | CIKM 2023 | [ACM DL](https://dl.acm.org/doi/10.1145/3583780.3615263) |
| Privacy & utility in skeleton data in VR metaverses | MetaCom 2024 | [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10740130) |

## Acknowledgements and license

This repository is a fork of [microsoft/SGN](https://github.com/microsoft/SGN), the official
implementation of *Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action
Recognition* (Zhang et al., CVPR 2020). The SGN backbone, training loop and NTU preprocessing
scripts are theirs; the re-identification and gender-classification heads, the Integrated Gradients
scoring module, the four protection mechanisms and the visualization suite are additions for this
work.

```bibtex
@inproceedings{zhang2020semantics,
  title     = {Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition},
  author    = {Zhang, Pengfei and Lan, Cuiling and Zeng, Wenjun and Xing, Junliang and Xue, Jianru and Zheng, Nanning},
  booktitle = {CVPR},
  year      = {2020}
}
```

Released under the MIT License — see [LICENSE](LICENSE), which retains the upstream Microsoft
copyright notice as required.

# Toward Generalizable and Robust Deepfake Detection: A Spatial-Frequency Fusion Baseline

This repository presents a deepfake image detection project focused on **generalization** and **robustness**.  
The core idea is to compare **spatial-only**, **frequency-only**, and **spatial-frequency fusion** models on **OpenFake**, and then evaluate whether the learned detector transfers to <!--**Semi-Truths** and--> corrupted test conditions.

## Overview

Recent image generators produce highly realistic images, making real/fake classification increasingly difficult.  
This project studies whether **frequency-domain cues** provide complementary information to standard RGB features, and whether combining both improves robustness under generator shifts, logo holdout settings, and external evaluation.

The project is organized around three questions:

- Can a spatial model alone generalize to unseen generators?
- Does a frequency-based model capture useful artifacts missed by RGB-only models?
- Does fusion improve robustness under distribution shift<!-- and external benchmarks-->?

## Methods

We compare three detector families:

- **Spatial-only**
  - ResNet-18
  - ViT

- **Frequency-only**
  - SPAI-style spectral encoder based on ViT

- **Fusion**
  - ResNet-18 + SPAI
  - ViT + SPAI

The frequency branch uses transformed spectral representations to capture generator-specific artifacts that may be weak in pixel space. The fusion model combines spatial and spectral features to produce the final binary prediction.

## Datasets

### [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake)
Primary dataset for training and in-domain evaluation.

This project uses OpenFake under multiple evaluation settings:

- **merged**: standard mixed-generator evaluation
- **by_generator**: per-generator test splits for generalization analysis
- **logo**: generator/logo holdout setting for stronger distribution shift

<!-- ### [Semi-Truths](https://huggingface.co/datasets/semi-truths/Semi-Truths)
External benchmark for out-of-distribution evaluation.

Semi-Truths is used to assess whether a detector trained on OpenFake remains reliable across:

- different edit types
- semantic edit magnitude
- area ratio
- scene complexity/diversity
- other grouped metadata conditions -->

## Evaluation

The project evaluates performance with:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
<!-- - Grouped metrics on Semi-Truths -->
- Corruption robustness under JPEG, blur, noise, and resize degradations
- Explainability with Grad-CAM, attention rollout, and frequency visualizations

## Main Results

### Main Results (AUC/F1)
| Model           | Merged            | By-generator      | LOGO holdout      | Group holdout     |
|-----------------|------------------:|------------------:|------------------:|------------------:|
| ResNet18        |     99.14 / 95.50 |     99.49 / 96.95 |     95.93 / 85.43 |     94.53 / 82.69 |
| ViT             |     99.59 / 96.92 |     99.80 / 98.09 |     96.72 / 84.43 |     94.14 / 77.24 |
| SPAI            |     99.62 / 97.09 |     99.77 / 98.10 |     96.70 / 83.97 |     94.89 / 79.31 |
| ResNet18 + SPAI |     99.51 / 96.49 |     99.70 / 97.60 |     96.63 / 84.07 |     94.98 / 79.71 |
| ViT + SPAI      |     99.45 / 96.29 |     99.71 / 97.00 |     96.20 / 83.74 |     94.51 / 78.50 |

### Main Results w/ Branch Dropout & Auxiliary Loss
| Model           | Merged            | By-generator      | LOGO holdout      | Group holdout     |
|-----------------|------------------:|------------------:|------------------:|------------------:|
| ResNet18 + SPAI |     99.50 / 96.82 |     99.65 / 97.50 |     96.47 / 84.07 |     94.30 / 76.99 |
| ViT + SPAI      |     99.53 / 96.66 |     99.72 / 98.01 |     ----- / ----- |     ----- / ----- |

### Main Results w/ Branch Dropout & Auxiliary Loss + ts
| Model           | Merged            | By-generator      | LOGO holdout      | Group holdout     |
|-----------------|------------------:|------------------:|------------------:|------------------:|
| ResNet18 + SPAI |     99.51 / 96.49 |     99.70 / 97.60 |     96.97 / 84.91 |     ----- / ----- |
| ViT + SPAI      |     ----- / ----- |     ----- / ----- |     ----- / ----- |     ----- / ----- |

### Robustness Corruptions (AUC/F1)

| Model           | Clean         | JPEG          | Blur          | Noise         | Resize        | Mean          | Drop ↓       | Worst         |
|-----------------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|-------------:|--------------:|
| ResNet18        | 99.06 / 95.50 | 96.89 / 81.34 | 94.68 / 87.23 | 96.46 / 82.10 | 97.92 / 91.96 | 96.49 / 85.66 | 2.57 / 9.84  | 87.32 / 76.44 |
| ViT             | 99.50 / 96.92 | 97.79 / 77.55 | 98.92 / 95.45 | 98.15 / 86.65 | 99.30 / 96.44 | 98.54 / 89.02 | 0.96 / 7.89  | 94.46 / 44.68 |
| SPAI            | 99.46 / 97.09 | 95.51 / 75.02 | 96.43 / 90.31 | 97.36 / 84.09 | 98.84 / 95.45 | 97.04 / 86.22 | 2.43 / 10.87 | 86.93 / 37.58 |
| ResNet18 + SPAI | 99.41 / 96.49 | 97.52 / 74.87 | 95.22 / 89.21 | 97.65 / 82.11 | 98.75 / 94.86 | 97.28 / 85.26 | 2.13 / 11.23 | 87.34 / 77.83 |
| ViT + SPAI      | 99.35 / 96.29 | 97.62 / 76.36 | 97.06 / 91.15 | 97.93 / 83.94 | 98.95 / 95.36 | 97.89 / 86.71 | 1.46 / 9.58  | 92.76 / 82.43 |

### Main Results w/ Robust-Aware Training
| Model           | Merged            | By-generator      | LOGO holdout      | Group holdout     |
|-----------------|------------------:|------------------:|------------------:|------------------:|
| ResNet18        |     ----- / ----- |     ----- / ----- |     ----- / ----- |     ----- / ----- |
| ViT             |     ----- / ----- |     ----- / ----- |     ----- / ----- |     ----- / ----- |
| SPAI            |     ----- / ----- |     ----- / ----- |     ----- / ----- |     ----- / ----- |
| ResNet18 + SPAI |     ----- / ----- |     ----- / ----- |     ----- / ----- |     ----- / ----- |
| ViT + SPAI      |     ----- / ----- |     ----- / ----- |     ----- / ----- |     ----- / ----- |

<!-- ### Main Results w/o Temperature Scaling (AUC/F1)

| Model           | Merged            | By-generator      | LOGO holdout      | Group holdout     |
|-----------------|------------------:|------------------:|------------------:|------------------:|
| ResNet18        |     99.06 / 95.50 |     99.49 / 96.95 |     95.87 / 85.43 |     94.50 / 82.69 |
| ViT             |     99.50 / 96.92 |     99.75 / 98.09 |     96.61 / 84.43 |     94.07 / 77.24 |
| SPAI            |     99.46 / 97.09 |     99.69 / 98.10 |     96.56 / 83.97 |     94.72 / 79.31 |
| ResNet18 + SPAI |     99.41 / 96.49 |     99.68 / 97.82 |     96.50 / 84.49 |     94.86 / 79.71 |
| ViT + SPAI      |     99.35 / 96.29 |     99.66 / 97.70 |     96.17 / 82.84 |     94.21 / 78.49 | -->

<!-- ### External Generalization on Semi-Truths

| Model | Overall AUC | Overall F1 | Worst Group AUC | Group Gap ↓ |
|---|---:|---:|---:|---:|
| ResNet18 | - | - | - | - |
| ViT | - | - | - | - |
| SPAI | - | - | - | - |
| ResNet18 + SPAI | - | - | - | - |
| ViT + SPAI | - | - | - | - | -->

## Installation

### 1) Clone the repository

```bash
git clone https://github.com/sangchun1/Deepfake-Fusion.git
cd Deepfake-Fusion
```

### 2) Install PyTorch

Install a PyTorch build that matches your CUDA / system environment first.

Example(CUDA 12.8):

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

> If your CUDA version or platform is different, use [the official PyTorch install guide](https://pytorch.org/get-started/locally/)

### 3) Install dependencies

```bash
pip install -U pip
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

## Training

### Spatial baseline (ResNet-18 on OpenFake)

```bash
python scripts/train.py `
  --data_config configs/data/openfake.yaml `
  --model_config configs/model/resnet18.yaml `
  --train_config configs/train/spatial_resnet_openfake.yaml
```

### Frequency baseline (SPAI on OpenFake)

```bash
python scripts/train.py `
  --data_config configs/data/openfake.yaml `
  --model_config configs/model/spai.yaml `
  --train_config configs/train/frequency_spai_openfake.yaml
```

### Fusion baseline (ResNet-18 + SPAI on OpenFake)

```bash
python scripts/train.py `
  --data_config configs/data/openfake.yaml `
  --model_config configs/model/fusion.yaml `
  --train_config configs/train/fusion_resnet_spai_openfake.yaml
```

### Fusion baseline (ViT + SPAI on OpenFake)

```bash
python scripts/train.py `
  --data_config configs/data/openfake.yaml `
  --model_config configs/model/fusion.yaml `
  --train_config configs/train/fusion_vit_spai_openfake.yaml
```

## Evaluation

### Standard evaluation

```bash
python scripts/evaluate.py `
  --data_config configs/data/openfake.yaml `
  --model_config configs/model/fusion.yaml `
  --train_config configs/train/fusion_resnet_spai_openfake.yaml
```

### Robustness evaluation

```bash
python scripts/evaluate_robustness.py `
  --data_config configs/data/openfake.yaml `
  --model_config configs/model/fusion.yaml `
  --train_config configs/train/fusion_resnet_spai_openfake.yaml `
  --robustness_config configs/train/robustness.yaml `
  --split test
```

<!-- ### Semi-Truths evaluation

```bash
python scripts/evaluate_semitruths.py `
  --data_config configs/data/semitruths_eval.yaml `
  --model_config configs/model/fusion.yaml `
  --train_config configs/train/fusion_resnet_spai_openfake.yaml
``` -->

## Explainability

### Standard explanation

```bash
python scripts/explain.py `
  --data_config configs/data/openfake.yaml `
  --model_config configs/model/fusion.yaml `
  --train_config configs/train/fusion_resnet_spai_openfake.yaml `
  --split test
```

The visualization method is selected according to model type:

- **Grad-CAM** for CNN-based spatial models
- **Attention rollout** for ViT-based models
- **Frequency visualization** for SPAI-based spectral models

## Project Goal

This repository is not only a benchmark implementation for deepfake classification, but also a study of:

- how well detectors generalize across generators,
- how robust they remain under corruption and domain shift,
- and whether frequency information provides a meaningful complementary signal.

The final aim is to build a **simple but strong baseline for generalizable and robust deepfake detection**.

## Citation

If you use this repository or build on this project, please cite the repository and acknowledge the OpenFake dataset<!-- and Semi-Truths datasets--> accordingly.

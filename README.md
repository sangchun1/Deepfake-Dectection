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

### Main Results on OpenFake

| Model | Merged (AUC / F1) | By-generator (AUC / F1) | LOGO holdout (AUC / F1) |
|---|---:|---:|---:|
| ResNet18 | 99.06 / 95.50 | 99.49 / 96.95 | 95.87 / **85.43** |
| ViT | **99.50** / <U>96.92</U> | - / - | - / - |
| SPAI | <U>99.46</U> / **97.09** | **99.69** / **98.10** | **96.56** / 83.97 |
| ResNet18 + SPAI | 99.41 / 96.49 | <U>99.68</U> / <U>97.82</U> | <U>96.50</U> / <U>84.49</U> |
| ViT + SPAI | - / - | - / - | - / - |

### Robustness on OpenFake Corruptions

| Model | Clean AUC | Corruption Mean AUC | AUC Drop ↓ | Worst-case AUC |
|---|---:|---:|---:|---:|
| ResNet18 | - | - | - | - |
| ViT | - | - | - | - |
| SPAI | - | - | - | - |
| ResNet18 + SPAI | - | - | - | - |
| ViT + SPAI | - | - | - | - |

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

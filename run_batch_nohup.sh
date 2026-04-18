#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs

echo "===== START: $(date) ====="

# spatial_resnet_openfake, spatial_vit_openfake, frequency_spai_openfake, fusion_resnet_spai_openfake, fusion_vit_spai_openfake

python -u scripts/run_batch_experiments.py \
  --mode group_holdout \
  --root_dir /home/user/DATA/deepfake \
  --base_data_config configs/data/openfake.yaml \
  --model_config configs/model/resnet18.yaml \
  --train_config configs/train/spatial_resnet_openfake.yaml \
  --splits_root data/splits/openfake \
  --generated_config_dir configs/_generated/openfake_batch \
  --output_root outputs/spatial/resnet_openfake \
  --explain_root outputs/explain/gradcam/openfake \
  --run_explain

python -u scripts/run_batch_experiments.py \
  --mode group_holdout \
  --root_dir /home/user/DATA/deepfake \
  --base_data_config configs/data/openfake.yaml \
  --model_config configs/model/vit.yaml \
  --train_config configs/train/spatial_vit_openfake.yaml \
  --splits_root data/splits/openfake \
  --generated_config_dir configs/_generated/openfake_batch \
  --output_root outputs/spatial/vit_openfake \
  --explain_root outputs/explain/rollout/openfake \
  --run_explain

python -u scripts/run_batch_experiments.py \
  --mode group_holdout \
  --root_dir /home/user/DATA/deepfake \
  --base_data_config configs/data/openfake.yaml \
  --model_config configs/model/spai.yaml \
  --train_config configs/train/frequency_spai_openfake.yaml \
  --splits_root data/splits/openfake \
  --generated_config_dir configs/_generated/openfake_batch \
  --output_root outputs/frequency/spai_openfake \
  --explain_root outputs/explain/frequency/openfake \
  --run_explain

python -u scripts/run_batch_experiments.py \
  --mode group_holdout \
  --root_dir /home/user/DATA/deepfake \
  --base_data_config configs/data/openfake.yaml \
  --model_config configs/model/fusion.yaml \
  --train_config configs/train/fusion_resnet_spai_openfake.yaml \
  --splits_root data/splits/openfake \
  --generated_config_dir configs/_generated/openfake_batch \
  --output_root outputs/fusion/resnet_spai_openfake \
  --explain_root outputs/explain/fusion/openfake \
  --run_explain

python -u scripts/run_batch_experiments.py \
  --mode group_holdout \
  --root_dir /home/user/DATA/deepfake \
  --base_data_config configs/data/openfake.yaml \
  --model_config configs/model/fusion.yaml \
  --train_config configs/train/fusion_vit_spai_openfake.yaml \
  --splits_root data/splits/openfake \
  --generated_config_dir configs/_generated/openfake_batch \
  --output_root outputs/fusion/vit_spai_openfake \
  --explain_root outputs/explain/fusion/openfake/rollout \
  --run_explain

echo "===== END: $(date) ====="
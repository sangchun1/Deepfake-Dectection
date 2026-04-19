#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs

echo "===== START: $(date) ====="

# spatial_resnet_openfake, spatial_vit_openfake, frequency_spai_openfake, fusion_resnet_spai_openfake, fusion_vit_spai_openfake

python -u scripts/run_batch_experiments.py \
  --mode by_generator \
  --root_dir /home/user/DATA/deepfake \
  --base_data_config configs/data/openfake.yaml \
  --model_config configs/model/fusion_vit.yaml \
  --train_config configs/train/fusion_vit_spai_openfake.yaml \
  --splits_root data/splits/openfake \
  --generated_config_dir configs/_generated/openfake_batch \
  --output_root outputs/fusion/vit_spai_openfake \
  --explain_root outputs/explain/fusion/openfake/rollout \
  # --run_explain

python -u scripts/run_batch_experiments.py \
  --mode logo \
  --root_dir /home/user/DATA/deepfake \
  --base_data_config configs/data/openfake.yaml \
  --model_config configs/model/fusion_vit.yaml \
  --train_config configs/train/fusion_vit_spai_openfake.yaml \
  --splits_root data/splits/openfake \
  --generated_config_dir configs/_generated/openfake_batch \
  --output_root outputs/fusion/vit_spai_openfake \
  --explain_root outputs/explain/fusion/openfake/rollout \

python -u scripts/run_batch_experiments.py \
  --mode group_holdout \
  --root_dir /home/user/DATA/deepfake \
  --base_data_config configs/data/openfake.yaml \
  --model_config configs/model/fusion_vit.yaml \
  --train_config configs/train/fusion_vit_spai_openfake.yaml \
  --splits_root data/splits/openfake \
  --generated_config_dir configs/_generated/openfake_batch \
  --output_root outputs/fusion/vit_spai_openfake \
  --explain_root outputs/explain/fusion/openfake/rollout \

echo "===== END: $(date) ====="
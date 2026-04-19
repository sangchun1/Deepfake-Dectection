#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

mkdir -p logs

echo "===== START BATCH ROBUSTNESS: $(date) ====="

python -u scripts/run_batch_robustness.py \
  --mode merged \
  --root_dir /home/user/DATA/deepfake \
  --base_data_config configs/data/openfake.yaml \
  --model_config configs/model/fusion_vit.yaml \
  --train_config configs/train/fusion_vit_spai_openfake.yaml \
  --robustness_config configs/train/robustness.yaml \
  --splits_root data/splits/openfake \
  --generated_config_dir configs/_generated/openfake_robustness_batch \
  --output_root outputs/fusion/vit_spai_openfake \
  --split test \
  # --skip_existing

echo "===== END BATCH ROBUSTNESS: $(date) ====="
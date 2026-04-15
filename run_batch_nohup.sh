#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

mkdir -p logs

echo "===== START: $(date) ====="

# spatial_resnet_openfake, spatial_vit_openfake, frequency_spai_openfake, fusion_resnet_spai_openfake, fusion_vit_spai_openfake
# dalle-3 flux-1.1-pro flux-mvc5000 flux.1-dev gpt-image-1 grok-2-image-1212 hidream-i1-full ideogram-3.0 imagen-4.0 midjourney-6 sd-3.5 sdxl-epic-realism

python -u scripts/run_batch_experiments.py \
  --mode merged \
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
  --mode by_generator \
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
  --mode logo \
  --root_dir /home/user/DATA/deepfake \
  --base_data_config configs/data/openfake.yaml \
  --model_config configs/model/fusion.yaml \
  --train_config configs/train/fusion_resnet_spai_openfake.yaml \
  --splits_root data/splits/openfake \
  --generated_config_dir configs/_generated/openfake_batch \
  --output_root outputs/fusion/resnet_spai_openfake \
  --explain_root outputs/explain/fusion/openfake \
  --run_explain

echo "===== END: $(date) ====="
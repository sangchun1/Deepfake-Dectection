#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

mkdir -p logs

echo "===== START BATCH ROBUSTNESS: $(date) ====="

# spatial_resnet_openfake, spatial_vit_openfake, frequency_spai_openfake, fusion_resnet_spai_openfake, fusion_vit_spai_openfake

python -u scripts/run_batch_robustness.py \
  --mode logo \
  --root_dir /home/user/DATA/deepfake \
  --base_data_config configs/data/openfake.yaml \
  --model_config configs/model/resnet18.yaml \
  --train_config configs/train/spatial_resnet_openfake.yaml \
  --robustness_config configs/train/robustness.yaml \
  --splits_root data/splits/openfake \
  --generated_config_dir configs/_generated/openfake_robustness_batch \
  --output_root outputs/spatial/resnet_openfake \
  --split test \
  --skip_existing

# python -u scripts/run_batch_robustness.py \
#   --mode all \
#   --root_dir /home/user/DATA/deepfake \
#   --base_data_config configs/data/openfake.yaml \
#   --model_config configs/model/vit.yaml \
#   --train_config configs/train/spatial_vit_openfake.yaml \
#   --robustness_config configs/train/robustness.yaml \
#   --splits_root data/splits/openfake \
#   --generated_config_dir configs/_generated/openfake_robustness_batch \
#   --output_root outputs/spatial/vit_openfake \
#   --split test \
#   --skip_existing

# python -u scripts/run_batch_robustness.py \
#   --mode all \
#   --root_dir /home/user/DATA/deepfake \
#   --base_data_config configs/data/openfake.yaml \
#   --model_config configs/model/spai.yaml \
#   --train_config configs/train/frequency_spai_openfake.yaml \
#   --robustness_config configs/train/robustness.yaml \
#   --splits_root data/splits/openfake \
#   --generated_config_dir configs/_generated/openfake_robustness_batch \
#   --output_root outputs/frequency/spai_openfake \
#   --split test \
#   --skip_existing

# python -u scripts/run_batch_robustness.py \
#   --mode all \
#   --root_dir /home/user/DATA/deepfake \
#   --base_data_config configs/data/openfake.yaml \
#   --model_config configs/model/fusion.yaml \
#   --train_config configs/train/fusion_resnet_spai_openfake.yaml \
#   --robustness_config configs/train/robustness.yaml \
#   --splits_root data/splits/openfake \
#   --generated_config_dir configs/_generated/openfake_robustness_batch \
#   --output_root outputs/fusion/resnet_spai_openfake \
#   --split test \
#   --skip_existing

# python -u scripts/run_batch_robustness.py \
#   --mode all \
#   --root_dir /home/user/DATA/deepfake \
#   --base_data_config configs/data/openfake.yaml \
#   --model_config configs/model/fusion.yaml \
#   --train_config configs/train/fusion_resnet_vit_openfake.yaml \
#   --robustness_config configs/train/robustness.yaml \
#   --splits_root data/splits/openfake \
#   --generated_config_dir configs/_generated/openfake_robustness_batch \
#   --output_root outputs/fusion/resnet_vit_openfake \
#   --split test \
#   --skip_existing

echo "===== END BATCH ROBUSTNESS: $(date) ====="
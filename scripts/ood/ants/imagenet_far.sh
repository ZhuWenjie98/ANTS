#!/usr/bin/env bash
# Run ANTS on ImageNet far-OOD benchmarks.
#
# Every setting can be overridden with an environment variable, for example:
# CUDA_DEVICE=0 GROUP_NUMS="100 50" bash scripts/ood/ants/imagenet_far.sh

#using ENS activation-aware scores
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
TEXT_PROMPT="${TEXT_PROMPT:-nice}"
IN_SCORE="${IN_SCORE:-far_only}"
OOD_SPLIT="${OOD_SPLIT:-farood}"
RANDOM_PERMUTE="${RANDOM_PERMUTE:-True}"
BACKBONE="${BACKBONE:-ViT-B/16}"
NEGLABEL_INIT_FLAG="${NEGLABEL_INIT_FLAG:-False}"
SAVE_ENS_LABELS="${SAVE_ENS_LABELS:-True}"
ACTIVATION_AWARE_ENS="${ACTIVATION_AWARE_ENS:-True}"
ACTIVATION_NEGATIVE_COUNT="${ACTIVATION_NEGATIVE_COUNT:-500}"
ACTIVATION_STEP="${ACTIVATION_STEP:-2}"
ACTIVATION_GAP="${ACTIVATION_GAP:-0.5}"
ACTIVATION_SCORE_QUEUE_SIZE="${ACTIVATION_SCORE_QUEUE_SIZE:-20000}"
ETA="${ETA:-0.50}"
MLLM_MODEL_TYPE="${MLLM_MODEL_TYPE:-QWEN}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TOTAL_CLASS_COUNT="${TOTAL_CLASS_COUNT:-11000}"
NEGATIVE_LABEL_COUNT="${NEGATIVE_LABEL_COUNT:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-./cvpr_reimp/}"

read -r -a GROUP_NUM_VALUES <<< "${GROUP_NUMS:-100}"
read -r -a ENSEMBLE_STEP_VALUES <<< "${ENS_STOP_STEPS:-10}"
read -r -a DATASET_CONFIG_VALUES <<< \
    "${DATASET_CONFIGS:-imagenet_traditional_four_ood}"

ants_validate_settings

for group_num in "${GROUP_NUM_VALUES[@]}"; do
    for ensemble_steps in "${ENSEMBLE_STEP_VALUES[@]}"; do
        for dataset_config in "${DATASET_CONFIG_VALUES[@]}"; do
            run_ants_imagenet_experiment \
                "${dataset_config}" "${group_num}" "${ensemble_steps}"
        done
    done
done

# dataset,FPR@95,AUROC,AUPR_IN,AUPR_OUT,ACC
# inaturalist,0.77,99.73,99.94,98.68,66.83
# sun,5.70,98.82,99.75,94.89,66.83
# places,23.13,95.46,98.88,85.72,66.83
# dtd,15.08,97.10,99.63,84.65,66.83
# farood,11.17,97.78,99.55,90.98,66.83

# Run the original ENS scoring as a baseline after the activation-aware run.
# ACTIVATION_AWARE_ENS=False

# ants_validate_settings

# for group_num in "${GROUP_NUM_VALUES[@]}"; do
#     for ensemble_steps in "${ENSEMBLE_STEP_VALUES[@]}"; do
#         for dataset_config in "${DATASET_CONFIG_VALUES[@]}"; do
#             run_ants_imagenet_experiment \
#                 "${dataset_config}" "${group_num}" "${ensemble_steps}"
#         done
#     done
# done

# dataset,FPR@95,AUROC,AUPR_IN,AUPR_OUT,ACC
# inaturalist,0.73,99.69,99.93,98.76,66.83
# sun,5.68,98.76,99.73,94.97,66.83
# places,32.92,94.29,98.56,84.05,66.83
# dtd,21.73,96.47,99.54,85.07,66.83
# farood,15.27,97.30,99.44,90.71,66.83

#!/usr/bin/env bash
# Evaluate ImageNet near-OOD datasets using only VSNL near-generated labels.
#
# ENS far-label generation and the initial negative-label bank are disabled.
# VSNL is initialized from the first batch so near_only does not spend the
# first ten batches on the base NegLabel fallback.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
TEXT_PROMPT="${TEXT_PROMPT:-nice}"
IN_SCORE="${IN_SCORE:-near_only}"
OOD_SPLIT="${OOD_SPLIT:-nearood}"
RANDOM_PERMUTE="${RANDOM_PERMUTE:-True}"
BACKBONE="${BACKBONE:-ViT-B/16}"
NEGLABEL_INIT_FLAG="${NEGLABEL_INIT_FLAG:-False}"
SAVE_ENS_LABELS="${SAVE_ENS_LABELS:-False}"
ETA="${ETA:-0.50}"
MLLM_MODEL_TYPE="${MLLM_MODEL_TYPE:-QWEN}"
NEAR_LABEL_START_BATCH="${NEAR_LABEL_START_BATCH:-1}"
NEAR_LABEL_INTERVAL="${NEAR_LABEL_INTERVAL:-40}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TOTAL_CLASS_COUNT="${TOTAL_CLASS_COUNT:-11000}"
NEGATIVE_LABEL_COUNT="${NEGATIVE_LABEL_COUNT:-10000}"
OUTPUT_DIR="${OUTPUT_DIR:-./cvpr_reimp/}"

read -r -a GROUP_NUM_VALUES <<< "${GROUP_NUMS:-100}"
# Zero completely disables ENS far-label generation.
read -r -a ENSEMBLE_STEP_VALUES <<< "${ENS_STOP_STEPS:-0}"
read -r -a DATASET_CONFIG_VALUES <<< \
    "${DATASET_CONFIGS:-imagenet_traditional_four_ood}"

ants_validate_settings

for group_num in "${GROUP_NUM_VALUES[@]}"; do
    for ensemble_steps in "${ENSEMBLE_STEP_VALUES[@]}"; do
        for dataset_config in "${DATASET_CONFIG_VALUES[@]}"; do
            run_ants_imagenet_experiment \
                "${dataset_config}" "${group_num}" "${ensemble_steps}"
        done
    done
done

# dataset,FPR@95,AUROC,AUPR_IN,AUPR_OUT,ACC
# ssb_hard,60.19,85.01,84.55,84.04,66.82
# ninco,59.63,81.04,97.22,32.48,66.83
# nearood,59.91,83.03,90.88,58.26,66.83

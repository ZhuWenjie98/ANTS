#!/usr/bin/env bash
# Shared runner functions for ANTS OOD experiments.

ANTS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANTS_REPO_ROOT="$(cd "${ANTS_SCRIPT_DIR}/../../.." && pwd)"

ants_require_choice() {
    local name="$1"
    local value="$2"
    shift 2
    local choice
    for choice in "$@"; do
        if [[ "${value}" == "${choice}" ]]; then
            return 0
        fi
    done
    printf 'Invalid %s=%q (expected one of: %s)\n' \
        "${name}" "${value}" "$*" >&2
    return 2
}

ants_validate_settings() {
    ants_require_choice \
        "IN_SCORE" "${IN_SCORE}" ada far_only near_only
    ants_require_choice \
        "OOD_SPLIT" "${OOD_SPLIT}" nearood farood
    ants_require_choice \
        "MLLM_MODEL_TYPE" "${MLLM_MODEL_TYPE}" QWEN LLAVA BLIP BLIP2
    ants_require_choice \
        "RANDOM_PERMUTE" "${RANDOM_PERMUTE}" True False
    ants_require_choice \
        "NEGLABEL_INIT_FLAG" "${NEGLABEL_INIT_FLAG}" True False
    ants_require_choice \
        "SAVE_ENS_LABELS" "${SAVE_ENS_LABELS:-False}" True False
    ants_require_choice \
        "ACTIVATION_AWARE_ENS" "${ACTIVATION_AWARE_ENS:-False}" True False
}

ants_experiment_mark() {
    local dataset_config="$1"
    local group_num="$2"
    local ensemble_steps="$3"
    local activation_mark=""
    if [[ "${ACTIVATION_AWARE_ENS:-False}" == "True" ]]; then
        activation_mark="_tanlens_top${ACTIVATION_NEGATIVE_COUNT:-1000}_step${ACTIVATION_STEP:-2}_gap${ACTIVATION_GAP:-0.5}"
    fi
    printf '%s_%s_mllm_%s_eta_%s_ens_%s_near_start_%s_groups_%s_permute_%s_ood' \
        "${dataset_config}" \
        "${IN_SCORE}" \
        "${MLLM_MODEL_TYPE}" \
        "${ETA}" \
        "${ensemble_steps}" \
        "${NEAR_LABEL_START_BATCH:-10}" \
        "${group_num}" \
        "${RANDOM_PERMUTE}${activation_mark}"
}

run_ants_imagenet_experiment() {
    local dataset_config="$1"
    local group_num="$2"
    local ensemble_steps="$3"
    local mark
    mark="$(ants_experiment_mark \
        "${dataset_config}" "${group_num}" "${ensemble_steps}")"

    local command=(
        "${PYTHON_BIN}" -u "${ANTS_REPO_ROOT}/main.py"
        --config
        "${ANTS_REPO_ROOT}/configs/datasets/imagenet/${dataset_config}.yml"
        "${ANTS_REPO_ROOT}/configs/networks/fixed_clip.yml"
        "${ANTS_REPO_ROOT}/configs/pipelines/test/test_fsood.yml"
        "${ANTS_REPO_ROOT}/configs/preprocessors/base_preprocessor.yml"
        "${ANTS_REPO_ROOT}/configs/postprocessors/ants.yml"
        --dataset.test.batch_size "${BATCH_SIZE}"
        --ood_dataset.batch_size "${BATCH_SIZE}"
        --dataset.train.few_shot 0
        --dataset.num_classes "${TOTAL_CLASS_COUNT}"
        --ood_dataset.num_classes "${TOTAL_CLASS_COUNT}"
        --evaluator.name ood_clip_tta
        --evaluator.ood_split "${OOD_SPLIT}"
        --network.name fixedclip_negoodprompt
        --network.backbone.ood_number "${NEGATIVE_LABEL_COUNT}"
        --network.backbone.name "${BACKBONE}"
        --network.backbone.text_prompt "${TEXT_PROMPT}"
        --network.backbone.text_center True
        --network.pretrained False
        --postprocessor.APS_mode False
        --postprocessor.name ants
        --postprocessor.postprocessor_args.group_num "${group_num}"
        --postprocessor.postprocessor_args.random_permute "${RANDOM_PERMUTE}"
        --postprocessor.postprocessor_args.in_score "${IN_SCORE}"
        --postprocessor.postprocessor_args.neglabel_init_flag \
        "${NEGLABEL_INIT_FLAG}"
        --postprocessor.postprocessor_args.save_ens_labels \
        "${SAVE_ENS_LABELS:-False}"
        --postprocessor.postprocessor_args.activation_aware_ens \
        "${ACTIVATION_AWARE_ENS:-False}"
        --postprocessor.postprocessor_args.activation_negative_count \
        "${ACTIVATION_NEGATIVE_COUNT:-1000}"
        --postprocessor.postprocessor_args.activation_step \
        "${ACTIVATION_STEP:-2}"
        --postprocessor.postprocessor_args.activation_gap \
        "${ACTIVATION_GAP:-0.5}"
        --postprocessor.postprocessor_args.activation_score_queue_size \
        "${ACTIVATION_SCORE_QUEUE_SIZE:-20000}"
        --postprocessor.postprocessor_args.eta "${ETA}"
        --postprocessor.postprocessor_args.ens_stop_step "${ensemble_steps}"
        --postprocessor.postprocessor_args.near_label_start_batch \
        "${NEAR_LABEL_START_BATCH:-10}"
        --postprocessor.postprocessor_args.near_label_interval \
        "${NEAR_LABEL_INTERVAL:-40}"
        --postprocessor.postprocessor_args.mllm_model_type \
        "${MLLM_MODEL_TYPE}"
        --num_gpus 1
        --num_workers "${NUM_WORKERS}"
        --merge_option merge
        --output_dir "${OUTPUT_DIR}"
        --mark "${mark}"
    )

    printf 'Running ANTS experiment: %s\n' "${mark}"
    (
        cd "${ANTS_REPO_ROOT}"
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${command[@]}"
    )
}

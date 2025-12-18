# ############################################################ 
tau=1.0
beta=1.0
######################### clip + ood + NegLabel
prompt=nice 
random_permute=True
#  30 50 100 200 1000 2000
# guiscore_text_mul guiscore_text_norm guiscore_text_add guiscore_text_cosonly guiscore_text_mul_merge 
# --config configs/datasets/imagenet/imagenet_traditional_four_ood.yml \
#         --config configs/datasets/imagenet/imagenet_train_fsood.yml \
for in_score in sum
do
    for group_num in 10
    do
        CUDA_VISIBLE_DEVICES=1 python main.py \
        --config configs/datasets/food/food_traditional_four_ood.yml \
        configs/networks/fixed_clip.yml \
        configs/pipelines/test/test_fsood.yml \
        configs/preprocessors/base_preprocessor.yml \
        configs/postprocessors/mcm.yml \
        --dataset.train.batch_size 128 \
        --dataset.train.few_shot 0 \
        --dataset.num_classes 10101 \
        --evaluator.name ood_clip_tta \
        --network.name fixedclip_llmoodprompt \
        --network.backbone.ood_number 10000 \
        --network.backbone.text_prompt ${prompt} \
        --network.backbone.text_center True \
        --network.pretrained False \
        --pipeline.name test_ood_llm2label \
        --postprocessor.APS_mode False \
        --postprocessor.name llm2label \
        --postprocessor.postprocessor_args.group_num ${group_num}  \
        --postprocessor.postprocessor_args.random_permute ${random_permute}  \
        --postprocessor.postprocessor_args.tau ${tau}  \
        --postprocessor.postprocessor_args.beta ${beta}  \
        --postprocessor.postprocessor_args.in_score ${in_score}  \
        --num_gpus 1 --num_workers 6 \
        --merge_option merge \
        --output_dir ./reimp_neglabel/ \
        --mark ${in_score}_beta${beta}_neg10k_group_num_${group_num}_random_${random_permute}_official
    done
done
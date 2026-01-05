HF_MODEL="/nlp_group/decapoda-research/DeepSeek-R1-Distill-Qwen-7B-Post"
# HF_MODEL="/mmu_nlp_hdd/lvbaibai/deepscaler/DeepSeek-R1-Distill-Qwen-1.5B"
TARGET_DIR="/mmu_nlp_hdd/panleiyu/verl_0929/ckpt/ckpts/gppo/distill_7B_ae_gppo_v1_math_bal_0_5_0_5/global_step_350/hf"
mkdir -p $TARGET_DIR
python scripts/legacy_model_merger.py --backend fsdp \
    --hf_model_path $HF_MODEL \
    --local_dir /mmu_nlp_hdd/panleiyu/verl_0929/ckpt/ckpts/gppo/distill_7B_ae_gppo_v1_math_bal_0_5_0_5/global_step_350/actor \
    --target_dir $TARGET_DIR

cp $HF_MODEL/tokenizer_config.json $TARGET_DIR/
cp $HF_MODEL/tokenizer.json $TARGET_DIR/
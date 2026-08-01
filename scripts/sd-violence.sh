export PATH="$HOME/.local/bin:$PATH"
# Mặc định dùng 1 GPU. Nếu có nhiều GPU, set: export CUDA_VISIBLE_DEVICES=0,1,2,3

num_samples=32
method='DPO'
exp_name='violence'

base_dir=$(pwd)
save_dir="./outputs"

# unlearning
# Paper Table 2: β=1000 cho violence, β=500 cũng có kết quả
# for dcoloss_beta in 2000 1000 500 250 100 ; do
for dcoloss_beta in 1000 ; do
for config_name in "Blood" "Horror" "Gun" "Suffer"; do

cd $base_dir/train
lora_dir="$save_dir/unlearn/SD-train/dpo/$dcoloss_beta/$config_name"
mkdir -p "$lora_dir"

python3 unlearn-sd.py \
    --config_dir="$base_dir/datasets/SD/config.json" \
    --config_name="${config_name}" \
    --data_dir="$base_dir/datasets/SD" \
    --output_dir="${lora_dir}" \
    --target_prompt="a bleeding man" \
    --synonym_prompt="a bleeding woman" \
    --prior_prompt="a man" \
    --base_lr=3e-4 \
    --adam_weight_decay=1e-2 \
    --dcoloss_beta=$dcoloss_beta \
    --base_lambda=1e6 \
    --rank=32 \
    --method=dpo \
    --train_batch_size=1 \
    --max_train_steps=500 \
    --checkpointing_steps=250 \
    --validation_steps=250 \
    --num_validation_images=2 \
    --num_samples=$num_samples \
    --t_max=750 \
    --t_min=1 \
    --dataloader_num_workers=0 \
    --no_cross_attn \
    --seed=42

done
done

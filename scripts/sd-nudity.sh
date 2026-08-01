export PATH="$HOME/.local/bin:$PATH"
# Mặc định dùng 1 GPU. Nếu có nhiều GPU, set: export CUDA_VISIBLE_DEVICES=0,1,2,3

config_name="Nudity"
num_samples=64
method='DPO'
exp_name='nudity'

base_dir=$(pwd)
save_dir="./outputs"

# Paper Section 4.2: β ∈ {100, 250, 500, 1000, 2000}
# Paper recommends β=500 for black-box, β=250 for white-box
# Uncomment the desired beta values:
# for dcoloss_beta in 2000 1000 500 250 100 ; do
for dcoloss_beta in 500 250 ; do

cd $base_dir/train
lora_dir="$save_dir/unlearn/SD-train/dpo/$dcoloss_beta/$config_name"
mkdir -p "$lora_dir"

python3 unlearn-sd.py \
    --config_dir="$base_dir/datasets/SD/config.json" \
    --config_name="${config_name}" \
    --data_dir="$base_dir/datasets/SD" \
    --output_dir="${lora_dir}" \
    --target_prompt="a naked woman" \
    --synonym_prompt="a naked man" \
    --prior_prompt="a man" \
    --base_lr=3e-4 \
    --adam_weight_decay=1e-2 \
    --dcoloss_beta=$dcoloss_beta \
    --base_lambda=1e6 \
    --rank=32 \
    --method=dpo \
    --train_batch_size=1 \
    --max_train_steps=1000 \
    --checkpointing_steps=250 \
    --validation_steps=250 \
    --num_validation_images=2 \
    --num_samples=$num_samples \
    --t_max=750 \
    --t_min=1 \
    --dataloader_num_workers=0 \
    --no_cross_attn \
    --seed=42

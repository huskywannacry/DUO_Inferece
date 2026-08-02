"""
Ring-A-Bell evaluation - Black-box red teaming attack.

Based on paper section 4.1:
- Black-box: no access to model gradients
- Uses 95 pre-trained prompts for nudity, 250 prompts for violence
- Loads prompts from chiayi-hsu/Ring-A-Bell repo

Usage:
    # Nudity evaluation
    python3 -m evaluation.ring_a_bell \
        --unlearn_model_path outputs/unlearn/SD-train/dpo/500/Nudity \
        --exp_type nudity \
        --output_dir eval_results/ring_a_bell_nudity_b500 \
        --ring_a_bell_prompts /path/to/Ring-A-Bell/data/Prompts_For_ConceptVector/Nudity_prompt.csv

    # Violence evaluation
    python3 -m evaluation.ring_a_bell \
        --unlearn_model_path outputs/unlearn/SD-train/dpo/1000 \
        --exp_type violence \
        --output_dir eval_results/ring_a_bell_violence_b1000 \
        --ring_a_bell_prompts /path/to/Ring-A-Bell/data/InvPrompt/Violence/Violence_eta_5.5_K_77.csv
"""

import argparse
import csv
import gc
import json
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Ring-A-Bell evaluation.")
    parser.add_argument("--unlearn_model_path", type=str, required=True)
    parser.add_argument("--exp_type", type=str, default="nudity", choices=["nudity", "violence"])
    parser.add_argument("--output_dir", type=str, default="eval_results/ring_a_bell")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ring_a_bell_prompts", type=str, default=None,
                        help="Path to Ring-A-Bell pre-trained prompts file (CSV or TXT). "
                             "Paper uses 95 for nudity, 250 for violence.")
    parser.add_argument("--num_prompts", type=int, default=None,
                        help="Number of prompts to use (default: all)")
    return parser.parse_args()


def load_prompts(path, exp_type="nudity"):
    """Load prompts from CSV or TXT file.

    Handles two Ring-A-Bell prompt formats:

    1. Nudity CSV (from Ring-A-Bell/data/Prompts_For_ConceptVector/Nudity_prompt.csv):
       Columns: case_number, nudity, people, clothes, location, evaluation_seed
       → Lấy cột 'nudity' (index 1)

    2. Violence CSV (from Ring-A-Bell/data/InvPrompt/Violence/):
       Columns phụ thuộc vào file, thường 'prompt' hoặc 'text'
       → Lấy cột 'prompt' hoặc fallback về cột đầu tiên

    3. TXT file: mỗi dòng là một prompt (fallback)
    """
    prompts = []
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        return prompts

    if path.endswith('.csv'):
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            print(f"[INFO] CSV columns: {fieldnames}")

            # Xác định cột nào chứa prompt dựa vào header
            prompt_column = None

            # Ưu tiên: các tên cột phổ biến chứa prompt
            for col in ['prompt', 'text', 'nudity', 'violence', 'unsafe', 'query']:
                if col in fieldnames:
                    prompt_column = col
                    break

            # Nếu không tìm thấy cột quen thuộc, dùng cột index 1 (bỏ qua cột đầu là số thứ tự)
            if prompt_column is None and len(fieldnames) >= 2:
                prompt_column = fieldnames[1]
                print(f"[WARN] No standard prompt column found. Using column '{prompt_column}' (index 1)")

            # Fallback cuối: cột đầu tiên
            if prompt_column is None and len(fieldnames) >= 1:
                prompt_column = fieldnames[0]
                print(f"[WARN] Using first column '{prompt_column}'")

            for row in reader:
                prompt = ''
                if prompt_column and prompt_column in row:
                    prompt = row[prompt_column]
                elif len(row) > 0:
                    # Fallback: first column
                    prompt = list(row.values())[0]
                if prompt and prompt.strip():
                    prompts.append(prompt.strip())

            print(f"[INFO] Loaded {len(prompts)} prompts from column '{prompt_column}'")
    else:
        with open(path, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"[INFO] Loaded {len(prompts)} prompts from text file")

    return prompts


def main():
    args = parse_args()

    print("=" * 50)
    print("Ring-A-Bell Evaluation")
    print("=" * 50)
    print(f"Unlearned model: {args.unlearn_model_path}")
    print(f"Task: {args.exp_type}")
    print(f"Output: {args.output_dir}")
    print()

    # Load Ring-A-Bell prompts
    attack_prompts = []
    if args.ring_a_bell_prompts:
        attack_prompts = load_prompts(args.ring_a_bell_prompts, args.exp_type)
        if attack_prompts:
            print(f"Loaded {len(attack_prompts)} Ring-A-Bell prompts from {args.ring_a_bell_prompts}")
        else:
            print(f"WARNING: No prompts loaded from {args.ring_a_bell_prompts}")
            return
    else:
        print("ERROR: --ring_a_bell_prompts is required.")
        print("Clone Ring-A-Bell repo and point to the prompts file:")
        print("  git clone https://github.com/chiayi-hsu/Ring-A-Bell.git")
        print("  --ring_a_bell_prompts Ring-A-Bell/data/Prompts_For_ConceptVector/Nudity_prompt.csv")
        return

    if args.num_prompts:
        attack_prompts = attack_prompts[:args.num_prompts]
        print(f"Using first {args.num_prompts} prompts")

    # Load model with LoRA
    print("Loading model...")
    weight_dtype = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=weight_dtype
    ).to(args.device)
    # Use same scheduler as paper evaluation
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None

    # Load LoRA weights
    if args.exp_type == "violence":
        config_list = ["Blood", "Gun", "Horror", "Suffer"]
        for config_name in config_list:
            lora_path = f'{args.unlearn_model_path}/{config_name}/pytorch_lora_weights.safetensors'
            if os.path.exists(lora_path):
                pipe.load_lora_weights(lora_path, adapter_name=config_name)
            else:
                print(f"Warning: LoRA not found at {lora_path}")
        pipe.set_adapters(config_list, adapter_weights=[1, 1, 1, 1])
    else:
        lora_path = f'{args.unlearn_model_path}/pytorch_lora_weights.safetensors'
        if not os.path.exists(lora_path):
            lora_path = f'{args.unlearn_model_path}/checkpoint-500/pytorch_lora_weights.safetensors'
        if os.path.exists(lora_path):
            pipe.load_lora_weights(lora_path)
        else:
            print(f"ERROR: LoRA not found at {args.unlearn_model_path}")
            return

    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate images
    print(f"Generating {len(attack_prompts)} images...")
    generator = torch.Generator(device=args.device)
    results = []

    for i, prompt in enumerate(tqdm(attack_prompts)):
        try:
            image = pipe(
                prompt,
                generator=generator.manual_seed(42 + i),
                num_images_per_prompt=1,
            ).images[0]
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in prompt)[:50]
            image.save(os.path.join(args.output_dir, f"{i:03d}_{safe_name}.png"))
            results.append({"id": i, "prompt": prompt, "status": "generated"})
        except Exception as e:
            print(f"Error at prompt {i}: {e}")
            results.append({"id": i, "prompt": prompt, "status": "error", "error": str(e)})

    # Save prompt list
    with open(os.path.join(args.output_dir, "ring_a_bell_prompts.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! {len(results)} images generated in {args.output_dir}")
    print(f"Next: run DSR evaluation:")
    print(f"  python3 -m evaluation.defense_success_rate --task {args.exp_type} "
          f"--image_dir {args.output_dir}")

    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()

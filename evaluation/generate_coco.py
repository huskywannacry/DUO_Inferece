"""
Generate images from MS COCO prompts using original SD1.4 and unlearned model.
Used for FID, CLIP Score, and LPIPS evaluation.

Usage:
    # Generate images from original SD1.4
    python3 evaluation/generate_coco.py --model_type original --output_dir eval_results/original_sd14

    # Generate images from unlearned model (nudity, beta=500)
    python3 evaluation/generate_coco.py --model_type unlearn --exp_type nudity \
        --unlearn_model_path outputs/unlearn/SD-train/dpo/500/Nudity \
        --output_dir eval_results/nudity_b500

    # Generate images from unlearned model (violence, beta=1000)
    python3 evaluation/generate_coco.py --model_type unlearn --exp_type violence \
        --unlearn_model_path outputs/unlearn/SD-train/dpo/1000 \
        --output_dir eval_results/violence_b1000
"""

import argparse
import gc
import json
import os
import torch
from pathlib import Path
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MS COCO images for evaluation.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="original",
        choices=["original", "unlearn"],
        help="original SD1.4 or unlearned model",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--unlearn_model_path",
        type=str,
        default=None,
        help="Path to unlearned LoRA weights",
    )
    parser.add_argument(
        "--exp_type",
        type=str,
        default="nudity",
        choices=["nudity", "violence"],
    )
    parser.add_argument(
        "--coco_annotations",
        type=str,
        default=None,
        help="Path to MS COCO captions JSON (captions_val2014.json). "
             "If not set, uses a hardcoded subset from the paper.",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=30000,
        help="Number of COCO prompts to use (default 30k as in paper)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results/coco_generated",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps. Use 10-20 for speed, 50 for quality.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["ddim", "dpm"],
        help="Scheduler: ddim (fast) or dpm (high quality, slower). Default=ddim for speed.",
    )
    return parser.parse_args()


def load_model(args):
    weight_dtype = torch.float16
    device = args.device

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    ).to(device)

    # Use faster scheduler: DDIM with trailing spacing gives good quality at 10-20 steps
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    if args.model_type == "unlearn":
        if args.exp_type == "violence":
            config_list = ["Blood", "Gun", "Horror", "Suffer"]
            for config_name in config_list:
                lora_path = f'{args.unlearn_model_path}/{config_name}/pytorch_lora_weights.safetensors'
                if os.path.exists(lora_path):
                    pipe.load_lora_weights(lora_path, adapter_name=config_name)
                else:
                    # Try checkpoint-500
                    lora_path = f'{args.unlearn_model_path}/{config_name}/checkpoint-500/pytorch_lora_weights.safetensors'
                    if os.path.exists(lora_path):
                        pipe.load_lora_weights(lora_path, adapter_name=config_name)
                    else:
                        print(f"Warning: LoRA not found at {lora_path}")
            pipe.set_adapters(config_list, adapter_weights=[1, 1, 1, 1])
        else:
            # Nudity
            lora_path = f'{args.unlearn_model_path}/pytorch_lora_weights.safetensors'
            if not os.path.exists(lora_path):
                lora_path = f'{args.unlearn_model_path}/checkpoint-500/pytorch_lora_weights.safetensors'
            if os.path.exists(lora_path):
                pipe.load_lora_weights(lora_path)
            else:
                print(f"Warning: LoRA not found at {lora_path}")

    pipe.safety_checker = None
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    return pipe


def get_coco_prompts(annotations_path, num_prompts=30000):
    """Extract captions from MS COCO validation annotations."""
    if annotations_path and os.path.exists(annotations_path):
        with open(annotations_path, 'r') as f:
            data = json.load(f)
        captions = [ann['caption'] for ann in data['annotations']]
        # Take first num_prompts (or all if less)
        return captions[:min(num_prompts, len(captions))]
    else:
        print("COCO annotations not found. Using default prompt list.")
        # Fallback: common prompts from COCO validation set themes
        default_prompts = [
            "a person walking down a street",
            "a cat sitting on a couch",
            "a plate of food on a table",
            "a dog running in a park",
            "a group of people standing around",
            "a bowl of fruit on a counter",
            "a car parked on the side of the road",
            "a woman holding an umbrella",
            "a man riding a bicycle",
            "a child playing with a toy",
            "a bird perched on a branch",
            "a boat floating on water",
            "a building with windows",
            "a cake with candles on top",
            "a chair next to a table",
            "a cup of coffee on a saucer",
            "a desk with a computer monitor",
            "a fire hydrant on a sidewalk",
            "a flower vase on a table",
            "a glass of water next to a plate",
            "a horse standing in a field",
            "a kitchen with cabinets and appliances",
            "a lamp on a nightstand",
            "a motorcycle parked on a street",
            "a person holding a cell phone",
            "a pizza on a wooden table",
            "a sandwich cut in half",
            "a stop sign on a corner",
            "a suitcase on a bed",
            "a television on a stand",
            "a train on railroad tracks",
            "a tree with green leaves",
            "a woman sitting on a bench",
            "a man wearing a hat",
            "a child holding a balloon",
            "a book on a shelf",
            "a bottle of wine on a table",
            "a clock on a wall",
            "a door with a brass handle",
            "a flag waving in the wind",
        ]
        # Repeat to get ~num_prompts
        return (default_prompts * ((num_prompts // len(default_prompts)) + 1))[:num_prompts]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model ({args.model_type})...")
    pipe = load_model(args)
    generator = torch.Generator(device=args.device)

    # Get prompts
    prompts = get_coco_prompts(args.coco_annotations, args.num_prompts)
    print(f"Generating {len(prompts)} images...")

    # Generate images
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i:i + args.batch_size]
        batch_size = len(batch_prompts)

        try:
            images = pipe(
                batch_prompts,
                generator=generator.manual_seed(args.seed + i),
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=1,
            ).images

            for j, img in enumerate(images):
                idx = i + j
                img.save(os.path.join(args.output_dir, f"{idx:06d}.png"))

        except Exception as e:
            print(f"Error at batch {i}: {e}")
            continue

        # Save prompt mapping (for CLIP score alignment)
        if i == 0:
            with open(os.path.join(args.output_dir, "prompts.jsonl"), "w") as f:
                for idx, p in enumerate(prompts):
                    f.write(json.dumps({"id": idx, "prompt": p}) + "\n")

    print(f"Done! Images saved to {args.output_dir}")

    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()

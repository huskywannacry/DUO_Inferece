"""Step 1: Generate anchor NSFW images with the ORIGINAL SD1.4.

Mirrors CCE's `esd/generate_i2p.py`: produce one image per I2P prompt with
the *original* (non-unlearned) model, writing the images plus a
`metadata.json` that maps file_name -> prompt. These are the visual targets
that textual inversion (<c>) will learn to reproduce.

Paper (DUO Section 4.1): malicious images are generated using prompts from
the i2p benchmark (sexual category for nudity, toxicity>=0.95 for violence).

Usage:
    python3 -m evaluation.concept_inversion.generate_anchors \
        --output_dir eval_results/ci_nudity_b250/anchor_images \
        --exp_type nudity --i2p_repo /home/kientt44/Code/i2p \
        --num_prompts 200 --device cuda
"""

import argparse
import json
import os

import torch
from tqdm import tqdm

from .utils import (
    load_original_pipeline,
    load_i2p_sexual_prompts,
    load_i2p_toxic_prompts,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Concept Inversion anchors.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exp_type", type=str, default="nudity",
                        choices=["nudity", "violence"])
    parser.add_argument("--i2p_repo", type=str, default="/home/kientt44/Code/i2p",
                        help="Path to the cloned ml-research/i2p repo (provenance).")
    parser.add_argument("--num_prompts", type=int, default=200,
                        help="Number of I2P prompts (paper uses 741; 200 fits a T4).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42, help="Fallback seed if I2P has none.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.exp_type == "nudity":
        prompts_df = load_i2p_sexual_prompts(args.i2p_repo, num_prompts=args.num_prompts)
    else:
        prompts_df = load_i2p_toxic_prompts(args.i2p_repo, num_prompts=args.num_prompts)
    print(f"Loaded {len(prompts_df)} {args.exp_type} prompts.")

    pipe = load_original_pipeline(args.device)
    gen = torch.Generator(args.device)
    metadata = []
    existing = {f for f in os.listdir(args.output_dir) if f.endswith(".png")}

    for i, row in tqdm(prompts_df.iterrows(), total=len(prompts_df), desc="anchor"):
        out = os.path.join(args.output_dir, f"{i:04d}.png")
        if os.path.basename(out) in existing:
            continue
        try:
            seed = int(row.get("sd_seed", args.seed))
            img = pipe(
                row["prompt"],
                num_inference_steps=args.num_inference_steps,
                guidance_scale=float(row.get("sd_guidance_scale", 7.5)),
                generator=gen.manual_seed(seed),
            ).images[0]
            img.save(out)
        except Exception as e:
            print(f"[anchor] prompt {i} failed: {e}")
        metadata.append({"file_name": f"{i:04d}.png", "prompt": [row["prompt"]]})

    # metadata.json is the contract consumed by concept_inversion.py training.
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Done. {len(metadata)} anchors + metadata.json -> {args.output_dir}")


if __name__ == "__main__":
    main()

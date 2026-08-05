"""Step 3: Generate attack images using the learned <c> token.

Mirrors CCE's `esd/generate_images.py` (mode=test): load the UNLEARNED
pipeline, inject the learned <c> embedding, then generate with
"<c> " + i2p prompt. The trained special token is used as a prefix of the
sexual/toxic prompts from the i2p benchmark (DUO paper Section 4.1).

Usage:
    python3 -m evaluation.concept_inversion.generate_images \
        --unlearn_model_path outputs/unlearn/SD-train/dpo/250/Nudity \
        --embed_path eval_results/ci_nudity_b250/learned_embeds.bin \
        --output_dir eval_results/ci_nudity_b250/generated \
        --exp_type nudity --i2p_repo /home/kientt44/Code/i2p \
        --num_prompts 200 --device cuda
"""

import argparse
import os

import torch

from .utils import (
    add_placeholder_and_load_embedding,
    generate_with_prefix,
    load_i2p_sexual_prompts,
    load_i2p_toxic_prompts,
    load_unlearn_pipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Concept Inversion attack images.")
    parser.add_argument("--unlearn_model_path", type=str, required=True)
    parser.add_argument("--embed_path", type=str, required=True,
                        help="learned_embeds.bin from concept_inversion.py.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exp_type", type=str, default="nudity",
                        choices=["nudity", "violence"])
    parser.add_argument("--i2p_repo", type=str, default="/home/kientt44/Code/i2p",
                        help="Cloned ml-research/i2p repo (provenance).")
    parser.add_argument("--num_prompts", type=int, default=200)
    parser.add_argument("--placeholder_token", type=str, default="<c>")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.exp_type == "nudity":
        prompts_df = load_i2p_sexual_prompts(args.i2p_repo, num_prompts=args.num_prompts)
    else:
        prompts_df = load_i2p_toxic_prompts(args.i2p_repo, num_prompts=args.num_prompts)
    print(f"Loaded {len(prompts_df)} {args.exp_type} prompts.")

    pipe = load_unlearn_pipeline(args.unlearn_model_path, args.device, args.exp_type)
    add_placeholder_and_load_embedding(pipe, args.placeholder_token, args.embed_path, args.device)

    generate_with_prefix(
        pipe,
        prompts_df,
        args.output_dir,
        placeholder_token=args.placeholder_token,
        device=args.device,
        batch_size=args.batch_size,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
    )

    print(f"Attack images -> {args.output_dir}")
    print("Next: run defense_success_rate.py --task nudity --image_dir "
          f"{args.output_dir}")


if __name__ == "__main__":
    main()

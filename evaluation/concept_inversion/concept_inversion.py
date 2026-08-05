"""Step 2: Train the <c> token embedding via textual inversion.

Mirrors CCE's `esd/concept_inversion.py` training script, but the CLI is
slimmed down to what matters and tuned to the DUO paper:

  - Paper Appendix C.2: Adam, lr=5e-3, batch_size=4, 3000 gradient steps.
  - CCE reference behavior: caption = "<c> " + the anchor image's own I2P
    prompt (TextualInversionDataset_I2P), and every embedding row except <c>
    is restored to its original value after each step.
  - The UNLEARNED pipeline (original SD1.4 + DUO LoRA) is the model under
    attack and is loaded here for the TI loop.

Usage:
    python3 -m evaluation.concept_inversion.concept_inversion \
        --unlearn_model_path outputs/unlearn/SD-train/dpo/250/Nudity \
        --train_data_dir eval_results/ci_nudity_b250/anchor_images \
        --metadata_path eval_results/ci_nudity_b250/anchor_images/metadata.json \
        --output_dir eval_results/ci_nudity_b250 \
        --placeholder_token "<c>" --initializer_token "man" \
        --exp_type nudity --device cuda

Inputs:  anchor images + metadata.json (from generate_anchors.py)
Outputs: learned_embeds.bin ({placeholder_token: embedding tensor})
"""

import argparse
import json
import os

import pandas as pd
import torch

from .utils import load_unlearn_pipeline, textual_inversion


def parse_args():
    parser = argparse.ArgumentParser(description="Concept Inversion training.")
    parser.add_argument("--unlearn_model_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True,
                        help="Folder of anchor images (from generate_anchors.py).")
    parser.add_argument("--metadata_path", type=str, required=True,
                        help="metadata.json mapping file_name -> prompt.")
    parser.add_argument("--i2p_repo", type=str, default="/home/kientt44/Code/i2p",
                        help="Cloned ml-research/i2p repo (provenance only).")
    parser.add_argument("--output_dir", type=str, default="eval_results/concept_inversion")
    parser.add_argument("--exp_type", type=str, default="nudity",
                        choices=["nudity", "violence"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--placeholder_token", type=str, default="<c>")
    parser.add_argument("--initializer_token", type=str, default="man")
    parser.add_argument("--max_steps", type=int, default=3000, help="Paper: 3000.")
    parser.add_argument("--batch_size", type=int, default=4, help="Paper: 4.")
    parser.add_argument("--learning_rate", type=float, default=5e-3, help="Paper: 5e-3.")
    parser.add_argument("--num_anchors", type=int, default=None,
                        help="Cap anchors used in training (default: all).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Metadata.json -> prompts_df keyed by row index (anchor filename is {i:04d}.png)
    with open(args.metadata_path) as f:
        meta = json.load(f)
    rows = []
    for item in meta:
        prompt = item["prompt"][0] if isinstance(item["prompt"], list) else item["prompt"]
        rows.append({"prompt": prompt})
    prompts_df = pd.DataFrame(rows)
    print(f"Loaded {len(prompts_df)} anchor captions from metadata.json.")

    os.makedirs(args.output_dir, exist_ok=True)
    pipe = load_unlearn_pipeline(args.unlearn_model_path, args.device, args.exp_type)

    emb = textual_inversion(
        pipe,
        args.device,
        anchor_dir=args.train_data_dir,
        prompts_df=prompts_df,
        placeholder_token=args.placeholder_token,
        initializer_token=args.initializer_token,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        num_anchors=args.num_anchors,
    )

    # Save in the same format as CCE's learned_embeds.bin.
    save_path = os.path.join(args.output_dir, "learned_embeds.bin")
    torch.save({args.placeholder_token: emb}, save_path)
    print(f"Saved <c> embedding to {save_path} (shape {tuple(emb.shape)}).")

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""
Run all evaluation metrics for DUO unlearning.
This script coordinates the full evaluation pipeline.

Usage:
    python3 -m evaluation.evaluate_all \
        --unlearn_model_path outputs/unlearn/SD-train/dpo/500/Nudity \
        --exp_type nudity \
        --coco_dir /path/to/coco_val_30k \
        --output_dir eval_results/nudity_b500
"""

import argparse
import json
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run full DUO evaluation.")
    parser.add_argument("--unlearn_model_path", type=str, required=True)
    parser.add_argument("--exp_type", type=str, default="nudity", choices=["nudity", "violence"])
    parser.add_argument("--coco_dir", type=str, default=None,
                        help="Path to MS COCO validation images")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--skip_clip", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip COCO image generation (use existing)")
    return parser.parse_args()


def run_cmd(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Warning: {desc} exited with code {result.returncode}")
    return result.returncode


def main():
    args = parse_args()

    # Infer experiment name
    exp_name = os.path.basename(args.unlearn_model_path.rstrip('/'))
    beta = os.path.basename(os.path.dirname(args.unlearn_model_path.rstrip('/')))
    output_base = os.path.join(args.output_dir, f"{args.exp_type}_{beta}")

    original_dir = os.path.join(output_base, "original_sd14")
    unlearn_dir = os.path.join(output_base, "unlearned")

    results = {}

    # Step 1: Generate COCO images from original model
    if not args.skip_generation:
        if not os.path.exists(original_dir) or len(os.listdir(original_dir)) < 100:
            os.makedirs(original_dir, exist_ok=True)
            run_cmd([
                sys.executable, "-m", "evaluation.generate_coco",
                "--model_type", "original",
                "--output_dir", original_dir,
                "--device", args.device,
            ], "Generating images from original SD1.4")

        # Generate from unlearned model
        if not os.path.exists(unlearn_dir) or len(os.listdir(unlearn_dir)) < 100:
            os.makedirs(unlearn_dir, exist_ok=True)
            run_cmd([
                sys.executable, "-m", "evaluation.generate_coco",
                "--model_type", "unlearn",
                "--exp_type", args.exp_type,
                "--unlearn_model_path", args.unlearn_model_path,
                "--output_dir", unlearn_dir,
                "--device", args.device,
            ], f"Generating images from unlearned model ({args.exp_type})")

    # Step 2: Compute FID (needs COCO real images)
    if not args.skip_fid and args.coco_dir:
        fid_result = subprocess.run([
            sys.executable, "-m", "evaluation.compute_fid",
            "--real_dir", args.coco_dir,
            "--fake_dir", unlearn_dir,
            "--device", args.device,
        ], capture_output=True, text=True)
        print(fid_result.stdout)
        results["fid"] = fid_result.stdout

    # Step 3: Compute CLIP Score
    if not args.skip_clip:
        clip_result = subprocess.run([
            sys.executable, "-m", "evaluation.compute_clip_score",
            "--image_dir", unlearn_dir,
            "--device", args.device,
        ], capture_output=True, text=True)
        print(clip_result.stdout)
        results["clip_score"] = clip_result.stdout

    # Step 4: Compute LPIPS (prior preservation)
    if not args.skip_lpips:
        lpips_result = subprocess.run([
            sys.executable, "-m", "evaluation.compute_lpips",
            "--original_dir", original_dir,
            "--unlearn_dir", unlearn_dir,
            "--device", args.device,
        ], capture_output=True, text=True)
        print(lpips_result.stdout)
        results["lpips"] = lpips_result.stdout

    # Save summary
    summary_path = os.path.join(output_base, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

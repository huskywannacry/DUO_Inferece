"""
Compute LPIPS (Learned Perceptual Image Patch Similarity) for Prior Preservation.

Measures perceptual distance between original SD1.4 images and unlearned model images
generated with the same noise seeds and prompts. Paper reports 1 - LPIPS.

Usage:
    python3 -m evaluation.compute_lpips \
        --original_dir /path/to/original_sd14_images \
        --unlearn_dir /path/to/unlearned_images
"""

import argparse
import os
import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Compute LPIPS for prior preservation.")
    parser.add_argument("--original_dir", type=str, required=True,
                        help="Path to images from original SD1.4")
    parser.add_argument("--unlearn_dir", type=str, required=True,
                        help="Path to images from unlearned model (same prompts/noise)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--net", type=str, default="alex", choices=["alex", "vgg"],
                        help="LPIPS backbone (alex is default)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load LPIPS model
    try:
        import lpips
        loss_fn = lpips.LPIPS(net=args.net).to(args.device)
    except ImportError:
        print("LPIPS not installed. Install with: pip install lpips")
        return

    # Get sorted image files
    orig_files = sorted([
        f for f in os.listdir(args.original_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    unlearn_files = sorted([
        f for f in os.listdir(args.unlearn_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])

    # Match files by name
    orig_map = {f: f for f in orig_files}
    unlearn_map = {f: f for f in unlearn_files}
    common = set(orig_map.keys()) & set(unlearn_map.keys())

    if len(common) == 0:
        # Try matching by index
        n = min(len(orig_files), len(unlearn_files))
        common = set()
        for i in range(n):
            common.add(orig_files[i])
        print(f"Matching {n} files by position (not filename)")

    print(f"Comparing {len(common)} image pairs...")

    distances = []
    for fname in tqdm(list(common)[:30000]):  # cap at 30k like paper
        try:
            img0 = lpips.load_image(os.path.join(args.original_dir, fname))
            img1 = lpips.load_image(os.path.join(args.unlearn_dir, fname))

            img0_tensor = torch.tensor(img0).to(args.device)
            img1_tensor = torch.tensor(img1).to(args.device)

            if img0_tensor.dim() == 3:
                img0_tensor = img0_tensor.unsqueeze(0)
            if img1_tensor.dim() == 3:
                img1_tensor = img1_tensor.unsqueeze(0)

            with torch.no_grad():
                d = loss_fn(img0_tensor, img1_tensor)
                distances.append(d.item())
        except Exception as e:
            continue

    if len(distances) > 0:
        avg_lpips = sum(distances) / len(distances)
        prior_preservation = 1.0 - avg_lpips
        print(f"\n{'='*50}")
        print(f"  LPIPS (lower = more similar): {avg_lpips:.4f}")
        print(f"  Prior Preservation (1 - LPIPS): {prior_preservation:.4f}")
        print(f"  (Paper plots Prior Preservation = 1 - LPIPS)")
        print(f"  Higher prior preservation = more similarity to original model")
        print(f"{'='*50}\n")
    else:
        print("No valid image pairs found.")
        print(f"  original_dir: {args.original_dir} ({len(orig_files)} files)")
        print(f"  unlearn_dir: {args.unlearn_dir} ({len(unlearn_files)} files)")


if __name__ == "__main__":
    main()

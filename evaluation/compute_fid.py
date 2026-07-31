"""
Compute FID (Frechet Inception Distance) between real and generated images.
Uses clean-fid library for consistency with paper results.

Usage:
    python3 -m evaluation.compute_fid \
        --real_dir /path/to/coco_val_30k \
        --fake_dir /path/to/generated_images

    Or with clean-fid custom dataset mode:
    python3 -m evaluation.compute_fid \
        --real_dir /path/to/coco_val_30k \
        --fake_dir /path/to/generated_images \
        --save_stats  (to cache real image stats)
"""

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Compute FID score.")
    parser.add_argument("--real_dir", type=str, required=True, help="Path to real images (COCO validation)")
    parser.add_argument("--fake_dir", type=str, required=True, help="Path to generated images")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_stats", action="store_true", help="Cache real stats for faster re-runs")
    return parser.parse_args()


def main():
    args = parse_args()

    # Try clean-fid first (closest to paper implementation)
    try:
        print("Using clean-fid...")
        from cleanfid import fid

        # Check if we have cached stats
        stats_file = os.path.join(args.real_dir, "..", "coco_val30k_stats.npz")
        if args.save_stats and os.path.exists(stats_file):
            score = fid.compute_fid(args.fake_dir, dataset_name="custom",
                                    dataset_res=256, dataset_split="custom",
                                    dataset_train=False,
                                    custom_feat_fake=args.fake_dir,
                                    custom_feat_real=stats_file)
        else:
            score = fid.compute_fid(args.real_dir, args.fake_dir,
                                    device=args.device,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)
            if args.save_stats:
                # cache real stats
                from cleanfid import features
                from cleanfid.resize import build_resizer
                print(f"FID score: {score:.4f}")
                return

    except ImportError:
        print("clean-fid not found, falling back to torchmetrics...")
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            from torchvision import transforms
            from torchvision.datasets import ImageFolder
            from torch.utils.data import DataLoader
            import torch

            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ])

            real_dataset = ImageFolder(args.real_dir if _is_folder_with_class(args.real_dir) else _wrap_folder(args.real_dir), transform=transform)
            fake_dataset = ImageFolder(args.fake_dir if _is_folder_with_class(args.fake_dir) else _wrap_folder(args.fake_dir), transform=transform)

            real_loader = DataLoader(real_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
            fake_loader = DataLoader(fake_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

            fid_metric = FrechetInceptionDistance(feature=2048).to(args.device)

            # Real images
            for images, _ in real_loader:
                fid_metric.update(images.to(args.device), real=True)

            # Fake images
            for images, _ in fake_loader:
                fid_metric.update(images.to(args.device), real=False)

            score = fid_metric.compute().item()

        except ImportError:
            print("torchmetrics not found either. Install with:")
            print("  pip install clean-fid")
            print("  or")
            print("  pip install torchmetrics[image]")
            return

    print(f"\n{'='*50}")
    print(f"  FID Score: {score:.4f}")
    print(f"  Paper reference (SD1.4 baseline): 13.52")
    print(f"  Paper reference (DUO β=500 nudity): 13.65")
    print(f"  Paper reference (DUO β=250 nudity): 13.59")
    print(f"  Paper reference (DUO β=1000 violence): 13.37")
    print(f"{'='*50}\n")


def _is_folder_with_class(path):
    """Check if path has class subfolders (ImageFolder format)."""
    if not os.path.isdir(path):
        return False
    items = os.listdir(path)
    for item in items:
        if os.path.isdir(os.path.join(path, item)):
            return True
    return False


def _wrap_folder(path):
    """Create a temporary class folder structure for ImageFolder."""
    import tempfile
    import shutil
    tmpdir = tempfile.mkdtemp()
    class_dir = os.path.join(tmpdir, "images")
    shutil.copytree(path, class_dir)
    return tmpdir


if __name__ == "__main__":
    main()

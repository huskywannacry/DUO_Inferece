#!/usr/bin/env python3
"""FIXED_IDEA Stage 1–3: generate Obama (or any person) preference pairs for DUO.

Layout (FIXED_IDEA §7):
  datasets/person_data/{slug}/
    unsafe/           # identity images (forget)
    safe_sdedit/      # global SDEdit 0.75 (DUO baseline)
    safe_face_inpaint/# Option A: SD inpaint face oval
    safe_face_crop/   # Option B: face-region SDEdit + paste
    verify_log.json
    pairs.jsonl

Also exports TrainDataset layouts:
  datasets/person_data/Obama_SDEdit/{unsafe,safe}
  datasets/person_data/Obama_FaceInpaint/{unsafe,safe}
  datasets/person_data/Obama_FaceCrop/{unsafe,safe}
  datasets/person_data/config.json

Examples:
  # Full N=64, all safe methods
  python3 datasets/person_data/generate_person_data.py \\
      --person "Barack Obama" --num_images 64 --methods sdedit,face_inpaint,face_crop

  # Pilot 50
  python3 datasets/person_data/generate_person_data.py --num_images 50 --methods all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# repo root on path
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from evaluation.face_recognition.arcface_verify import (  # noqa: E402
    arcface_l2,
    face_bbox_xyxy,
    face_mask_pil,
    has_face,
)

CONTEXTS = [
    "official portrait, front view, suit",
    "casual jacket, outdoors, natural lighting",
    "speaking at podium, gesture, indoor",
    "candid photo, slight angle, soft smile",
    "holding microphone, conference setting",
    "black and white photo, vintage style",
    "side profile, dramatic lighting",
    "indoor office, soft window light",
]

METHOD_TO_SAFE_DIR = {
    "sdedit": "safe_sdedit",
    "face_inpaint": "safe_face_inpaint",
    "face_crop": "safe_face_crop",
}

METHOD_TO_CONCEPT = {
    "sdedit": "Obama_SDEdit",
    "face_inpaint": "Obama_FaceInpaint",
    "face_crop": "Obama_FaceCrop",
}


def parse_args():
    p = argparse.ArgumentParser(description="FIXED_IDEA person pair generation.")
    p.add_argument("--person", type=str, default="Barack Obama")
    p.add_argument("--slug", type=str, default="obama",
                   help="Folder under datasets/person_data/")
    p.add_argument("--num_images", type=int, default=64)
    p.add_argument(
        "--methods",
        type=str,
        default="sdedit,face_inpaint",
        help="Comma list: sdedit,face_inpaint,face_crop,all",
    )
    p.add_argument("--save_dir", type=str, default=None,
                   help="Default: <repo>/datasets/person_data")
    p.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--inpaint_model_id", type=str,
                   default="runwayml/stable-diffusion-inpainting")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--sdedit_strength", type=float, default=0.75)
    p.add_argument("--inpaint_strength", type=float, default=0.85)
    p.add_argument("--face_crop_strength", type=float, default=0.85)
    p.add_argument("--arcface_threshold", type=float, default=0.5)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--require_verify", action="store_true", default=True,
                   help="Retry until ArcFace distance > threshold (default on).")
    p.add_argument("--no_verify", action="store_true")
    p.add_argument("--export_train", action="store_true", default=True,
                   help="Export Obama_* folders for unlearn-sd.py (default on).")
    p.add_argument("--no_export_train", action="store_true")
    p.add_argument("--concept_prefix", type=str, default="Obama",
                   help="Train concept names: {prefix}_SDEdit, ...")
    return p.parse_args()


def parse_methods(s: str) -> List[str]:
    s = s.strip().lower()
    if s == "all":
        return ["sdedit", "face_inpaint", "face_crop"]
    out = []
    for m in s.split(","):
        m = m.strip()
        if not m:
            continue
        if m not in METHOD_TO_SAFE_DIR:
            raise ValueError(f"Unknown method {m}; choose {list(METHOD_TO_SAFE_DIR)}")
        out.append(m)
    return out


def build_prompts(person: str, n: int) -> Tuple[List[str], List[str]]:
    unsafe, safe = [], []
    for i in range(n):
        ctx = CONTEXTS[i % len(CONTEXTS)]
        unsafe.append(f"{person}, {ctx}, realistic photo, high quality")
        safe.append(f"a person, {ctx}, realistic photo, high quality")
    return unsafe, safe


def load_txt2img(model_id, device):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    pipe.enable_vae_slicing()
    return pipe


def load_img2img(model_id, device):
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    pipe.enable_vae_slicing()
    return pipe


def load_inpaint(model_id, device):
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    pipe.enable_vae_slicing()
    return pipe


def fname(i: int) -> str:
    return f"{i:03d}.jpg"


def gen_unsafe(
    pipe,
    prompts: List[str],
    out_dir: str,
    device: str,
    seed: int,
    steps: int,
    guidance: float,
    max_retries: int,
):
    os.makedirs(out_dir, exist_ok=True)
    gen = torch.Generator(device=device)
    for i, prompt in enumerate(tqdm(prompts, desc="unsafe")):
        path = os.path.join(out_dir, fname(i))
        if os.path.exists(path):
            continue
        saved = False
        for attempt in range(max_retries):
            img = pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gen.manual_seed(seed + i * 17 + attempt),
            ).images[0].resize((512, 512))
            try:
                if has_face(img):
                    img.save(path, quality=95)
                    saved = True
                    break
            except ImportError:
                img.save(path, quality=95)
                saved = True
                break
        if not saved:
            img.save(path, quality=95)


def _accept(dist: Optional[float], verify: bool, thr: float) -> bool:
    if not verify:
        return True
    if dist is None:
        # safe has no face → identity removed (count as accept)
        return True
    return dist > thr


def gen_safe_sdedit(
    pipe,
    unsafe_dir,
    safe_dir,
    unsafe_prompts,
    safe_prompts,
    person,
    strength,
    steps,
    guidance,
    verify,
    thr,
    max_retries,
) -> List[dict]:
    os.makedirs(safe_dir, exist_ok=True)
    log = []
    n = len(safe_prompts)
    for i in tqdm(range(n), desc="safe_sdedit"):
        u_path = os.path.join(unsafe_dir, fname(i))
        s_path = os.path.join(safe_dir, fname(i))
        unsafe = Image.open(u_path).convert("RGB").resize((512, 512))
        best, best_dist, passed = None, None, False
        for attempt in range(max_retries):
            edit = pipe(
                prompt=safe_prompts[i],
                negative_prompt=person,
                image=unsafe,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
            ).images[0].resize((512, 512))
            try:
                dist = arcface_l2(unsafe, edit)
            except ImportError:
                dist = None
                verify = False
            best, best_dist = edit, dist
            if _accept(dist, verify, thr):
                passed = True
                break
        best.save(s_path, quality=95)
        log.append(
            {
                "file": fname(i),
                "method": "sdedit",
                "arcface_dist": _json_dist(best_dist),
                "passed": passed if verify else None,
                "strength": strength,
            }
        )
    return log


def _json_dist(d):
    if d is None:
        return None
    if d == float("inf") or d != d:  # inf or nan
        return None  # interpret as no-face / extreme; paired with passed flag
    return float(d)


def gen_safe_face_inpaint(
    pipe,
    unsafe_dir,
    safe_dir,
    safe_prompts,
    person,
    strength,
    steps,
    guidance,
    verify,
    thr,
    max_retries,
) -> List[dict]:
    os.makedirs(safe_dir, exist_ok=True)
    log = []
    n = len(safe_prompts)
    for i in tqdm(range(n), desc="safe_face_inpaint"):
        u_path = os.path.join(unsafe_dir, fname(i))
        s_path = os.path.join(safe_dir, fname(i))
        unsafe = Image.open(u_path).convert("RGB").resize((512, 512))
        try:
            mask = face_mask_pil(unsafe, expand=24)
        except ImportError as e:
            raise ImportError("face_inpaint needs insightface") from e
        if mask is None:
            mask = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
            m = np.array(mask)
            m[120:400, 120:400] = 255
            mask = Image.fromarray(m)

        best, best_dist, passed = None, None, False
        for attempt in range(max_retries):
            # slight strength jitter on retry
            st = min(0.95, strength + 0.03 * attempt)
            edit = pipe(
                prompt=safe_prompts[i],
                negative_prompt=f"{person}, blurry, distorted face",
                image=unsafe,
                mask_image=mask,
                strength=st,
                guidance_scale=guidance,
                num_inference_steps=steps,
                height=512,
                width=512,
            ).images[0].resize((512, 512))
            dist = arcface_l2(unsafe, edit)
            best, best_dist = edit, dist
            if _accept(dist, verify, thr):
                passed = True
                break
        best.save(s_path, quality=95)
        log.append(
            {
                "file": fname(i),
                "method": "face_inpaint",
                "arcface_dist": _json_dist(best_dist),
                "passed": passed if verify else None,
                "strength": strength,
            }
        )
    return log


def _paste_face(base: Image.Image, face_edit: Image.Image, bbox) -> Image.Image:
    """Paste resized face crop back with simple alpha blend on edge."""
    import cv2

    x1, y1, x2, y2 = bbox
    out = np.array(base.convert("RGB")).copy()
    crop = np.array(face_edit.convert("RGB").resize((x2 - x1, y2 - y1)))
    # feather mask
    h, w = crop.shape[:2]
    mask = np.ones((h, w), dtype=np.float32)
    feather = max(3, min(h, w) // 12)
    for t in range(feather):
        alpha = (t + 1) / feather
        mask[t, :] *= alpha
        mask[h - 1 - t, :] *= alpha
        mask[:, t] *= alpha
        mask[:, w - 1 - t] *= alpha
    mask = mask[..., None]
    region = out[y1:y2, x1:x2].astype(np.float32)
    blended = crop.astype(np.float32) * mask + region * (1.0 - mask)
    out[y1:y2, x1:x2] = blended.astype(np.uint8)
    return Image.fromarray(out)


def gen_safe_face_crop(
    pipe,
    unsafe_dir,
    safe_dir,
    safe_prompts,
    person,
    strength,
    steps,
    guidance,
    verify,
    thr,
    max_retries,
) -> List[dict]:
    """Option B: img2img only on face crop, paste back (same SD1.4 backbone)."""
    os.makedirs(safe_dir, exist_ok=True)
    log = []
    n = len(safe_prompts)
    for i in tqdm(range(n), desc="safe_face_crop"):
        u_path = os.path.join(unsafe_dir, fname(i))
        s_path = os.path.join(safe_dir, fname(i))
        unsafe = Image.open(u_path).convert("RGB").resize((512, 512))
        bbox = face_bbox_xyxy(unsafe, pad=20)
        if bbox is None:
            # fallback whole-image mild edit
            bbox = (128, 96, 384, 400)
        x1, y1, x2, y2 = bbox
        face_crop = unsafe.crop((x1, y1, x2, y2)).resize((512, 512))

        best, best_dist, passed = None, None, False
        for attempt in range(max_retries):
            st = min(0.95, strength + 0.03 * attempt)
            edited_face = pipe(
                prompt=safe_prompts[i],
                negative_prompt=person,
                image=face_crop,
                strength=st,
                guidance_scale=guidance,
                num_inference_steps=steps,
            ).images[0]
            full = _paste_face(unsafe, edited_face, bbox)
            dist = arcface_l2(unsafe, full)
            best, best_dist = full, dist
            if _accept(dist, verify, thr):
                passed = True
                break
        best.save(s_path, quality=95)
        log.append(
            {
                "file": fname(i),
                "method": "face_crop",
                "arcface_dist": _json_dist(best_dist),
                "passed": passed if verify else None,
                "strength": strength,
                "bbox": list(bbox),
            }
        )
    return log


def export_train_layout(
    root: str,
    slug_dir: str,
    methods: List[str],
    unsafe_prompts: List[str],
    safe_prompts: List[str],
    concept_prefix: str,
):
    """Copy/symlink into TrainDataset folders + write config.json."""
    cfg_path = os.path.join(root, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)

    unsafe_src = os.path.join(slug_dir, "unsafe")
    name_map = {
        "sdedit": f"{concept_prefix}_SDEdit",
        "face_inpaint": f"{concept_prefix}_FaceInpaint",
        "face_crop": f"{concept_prefix}_FaceCrop",
    }

    for m in methods:
        concept = name_map[m]
        safe_src = os.path.join(slug_dir, METHOD_TO_SAFE_DIR[m])
        concept_dir = os.path.join(root, concept)
        u_dst = os.path.join(concept_dir, "unsafe")
        s_dst = os.path.join(concept_dir, "safe")
        os.makedirs(u_dst, exist_ok=True)
        os.makedirs(s_dst, exist_ok=True)

        # hard copy small jpgs (portable on Kaggle)
        for name in sorted(os.listdir(unsafe_src)):
            if name.endswith((".jpg", ".png")):
                shutil.copy2(os.path.join(unsafe_src, name), os.path.join(u_dst, name))
        for name in sorted(os.listdir(safe_src)):
            if name.endswith((".jpg", ".png")):
                shutil.copy2(os.path.join(safe_src, name), os.path.join(s_dst, name))

        cfg[concept] = {
            "prompt": unsafe_prompts,
            "base_prompt": safe_prompts,
            "images": "unsafe",
            "base_images": "safe",
            "method": m,
        }
        print(f"Exported train layout: {concept_dir}")

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote {cfg_path}")


def write_pairs_jsonl(slug_dir: str, methods: List[str], n: int):
    path = os.path.join(slug_dir, "pairs.jsonl")
    with open(path, "w") as f:
        for i in range(n):
            row = {"id": fname(i), "unsafe": f"unsafe/{fname(i)}"}
            for m in methods:
                row[m] = f"{METHOD_TO_SAFE_DIR[m]}/{fname(i)}"
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {path}")


def main():
    args = parse_args()
    verify = args.require_verify and not args.no_verify
    methods = parse_methods(args.methods)
    root = args.save_dir or os.path.join(_REPO, "datasets", "person_data")
    slug_dir = os.path.join(root, args.slug)
    os.makedirs(slug_dir, exist_ok=True)

    unsafe_prompts, safe_prompts = build_prompts(args.person, args.num_images)
    unsafe_dir = os.path.join(slug_dir, "unsafe")

    print(f"person={args.person!r} N={args.num_images} methods={methods} verify={verify}")
    print(f"out={slug_dir}")

    # Stage 1
    pipe = load_txt2img(args.model_id, args.device)
    gen_unsafe(
        pipe, unsafe_prompts, unsafe_dir, args.device, args.seed, args.steps, args.guidance,
        args.max_retries,
    )
    del pipe
    torch.cuda.empty_cache()

    all_log: List[dict] = []

    # Stage 2
    if "sdedit" in methods:
        img2img = load_img2img(args.model_id, args.device)
        all_log += gen_safe_sdedit(
            img2img,
            unsafe_dir,
            os.path.join(slug_dir, "safe_sdedit"),
            unsafe_prompts,
            safe_prompts,
            args.person,
            args.sdedit_strength,
            args.steps,
            args.guidance,
            verify,
            args.arcface_threshold,
            args.max_retries,
        )
        del img2img
        torch.cuda.empty_cache()

    if "face_crop" in methods:
        img2img = load_img2img(args.model_id, args.device)
        all_log += gen_safe_face_crop(
            img2img,
            unsafe_dir,
            os.path.join(slug_dir, "safe_face_crop"),
            safe_prompts,
            args.person,
            args.face_crop_strength,
            args.steps,
            args.guidance,
            verify,
            args.arcface_threshold,
            args.max_retries,
        )
        del img2img
        torch.cuda.empty_cache()

    if "face_inpaint" in methods:
        inpaint = load_inpaint(args.inpaint_model_id, args.device)
        all_log += gen_safe_face_inpaint(
            inpaint,
            unsafe_dir,
            os.path.join(slug_dir, "safe_face_inpaint"),
            safe_prompts,
            args.person,
            args.inpaint_strength,
            args.steps,
            args.guidance,
            verify,
            args.arcface_threshold,
            args.max_retries,
        )
        del inpaint
        torch.cuda.empty_cache()

    # Stage 3 log
    log_path = os.path.join(slug_dir, "verify_log.json")
    with open(log_path, "w") as f:
        json.dump(all_log, f, indent=2)

    # pass-rate summary
    summary = {}
    for m in methods:
        rows = [r for r in all_log if r["method"] == m]
        dists = [r["arcface_dist"] for r in rows if r.get("arcface_dist") is not None]
        passed = [r for r in rows if r.get("passed")]
        summary[m] = {
            "n": len(rows),
            "pass_rate": len(passed) / max(len(rows), 1) if verify else None,
            "mean_arcface": float(np.mean(dists)) if dists else None,
            "median_arcface": float(np.median(dists)) if dists else None,
        }
    with open(os.path.join(slug_dir, "pair_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Pair summary:", json.dumps(summary, indent=2))

    write_pairs_jsonl(slug_dir, methods, args.num_images)

    if args.export_train and not args.no_export_train:
        export_train_layout(
            root, slug_dir, methods, unsafe_prompts, safe_prompts, args.concept_prefix
        )

    meta = {
        "person": args.person,
        "num_images": args.num_images,
        "methods": methods,
        "arcface_threshold": args.arcface_threshold,
        "summary": summary,
    }
    with open(os.path.join(slug_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()

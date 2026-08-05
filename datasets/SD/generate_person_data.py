# DUO person-unlearning data pipeline
# Generate paired (unsafe=identity, safe=edited) images for DUO training.
#
# Layout (same as Nudity):
#   {save_dir}/{concept}/unsafe/00.jpg ...
#   {save_dir}/{concept}/safe/00.jpg ...
#   {save_dir}/config_person.json
#
# Examples:
#   # Baseline DUO-SDEdit pairs (64 images)
#   python3 datasets/SD/generate_person_data.py \
#       --concept Obama --person "Barack Obama" \
#       --method sdedit --num_images 64 --save_dir datasets/SD
#
#   # Face-inpaint pairs (needs insightface + inpaint model)
#   python3 datasets/SD/generate_person_data.py \
#       --concept Obama --person "Barack Obama" \
#       --method face_inpaint --num_images 64 --save_dir datasets/SD
#
#   # Pilot: 50 images, both methods into separate concepts
#   python3 datasets/SD/generate_person_data.py --concept Obama_SDEdit \
#       --person "Barack Obama" --method sdedit --num_images 50
#   python3 datasets/SD/generate_person_data.py --concept Obama_FaceEdit \
#       --person "Barack Obama" --method face_inpaint --num_images 50

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


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


def parse_args():
    p = argparse.ArgumentParser(description="Generate person identity pairs for DUO.")
    p.add_argument("--concept", type=str, default="Obama",
                   help="Folder / config key name (e.g. Obama, Obama_SDEdit).")
    p.add_argument("--person", type=str, default="Barack Obama",
                   help="Identity name used in unsafe prompts.")
    p.add_argument("--safe_prompt", type=str, default="a person, realistic photo, high quality",
                   help="Prompt for safe edit (non-identity).")
    p.add_argument("--num_images", type=int, default=64)
    p.add_argument("--method", type=str, default="sdedit",
                   choices=["sdedit", "face_inpaint"],
                   help="How to build the safe counterpart.")
    p.add_argument("--sdedit_strength", type=float, default=0.75)
    p.add_argument("--inpaint_strength", type=float, default=0.85)
    p.add_argument("--save_dir", type=str, default="datasets/SD")
    p.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--inpaint_model_id", type=str,
                   default="runwayml/stable-diffusion-inpainting")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--verify_arcface", action="store_true",
                   help="Reject safe images still too similar (needs insightface).")
    p.add_argument("--arcface_threshold", type=float, default=0.5,
                   help="Min L2 distance between normed embeddings to accept pair.")
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--write_config", action="store_true", default=True,
                   help="Write/update config_person.json (default on).")
    p.add_argument("--no_write_config", action="store_true")
    return p.parse_args()


def build_prompts(person: str, n: int) -> Tuple[List[str], List[str]]:
    """Diverse unsafe prompts + matching safe base prompts."""
    unsafe, safe = [], []
    for i in range(n):
        ctx = CONTEXTS[i % len(CONTEXTS)]
        unsafe.append(f"{person}, {ctx}, realistic photo, high quality")
        safe.append(f"a person, {ctx}, realistic photo, high quality")
    return unsafe, safe


def load_txt2img(model_id: str, device: str, dtype=torch.float16):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    pipe.enable_vae_slicing()
    return pipe


def load_img2img(model_id: str, device: str, dtype=torch.float16):
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    pipe.enable_vae_slicing()
    return pipe


def load_inpaint(model_id: str, device: str, dtype=torch.float16):
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    pipe.enable_vae_slicing()
    return pipe


# ---------------------------------------------------------------------------
# Optional ArcFace (insightface)
# ---------------------------------------------------------------------------
_face_app = None


def get_face_app():
    global _face_app
    if _face_app is not None:
        return _face_app
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise ImportError(
            "insightface is required for --verify_arcface / face_inpaint.\n"
            "  pip install insightface onnxruntime-gpu  # or onnxruntime\n"
        ) from e
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    _face_app = app
    return app


def largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def arcface_distance(img_a: Image.Image, img_b: Image.Image) -> Optional[float]:
    """L2 distance between normed embeddings; None if a face is missing."""
    app = get_face_app()
    a = np.array(img_a.convert("RGB"))
    b = np.array(img_b.convert("RGB"))
    fa, fb = largest_face(app.get(a)), largest_face(app.get(b))
    if fa is None or fb is None:
        return None
    return float(np.linalg.norm(fa.normed_embedding - fb.normed_embedding))


def has_face(img: Image.Image) -> bool:
    app = get_face_app()
    return largest_face(app.get(np.array(img.convert("RGB")))) is not None


# ---------------------------------------------------------------------------
# Face oval mask for inpainting
# ---------------------------------------------------------------------------
def create_oval_mask(image_shape, landmarks: np.ndarray, expand: int = 20) -> np.ndarray:
    import cv2

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    center = landmarks.mean(axis=0).astype(int)
    eye_distance = float(np.linalg.norm(landmarks[0] - landmarks[1]))
    radius = int(eye_distance * 1.2) + expand
    axes = (max(radius, 1), max(int(radius * 1.3), 1))
    cv2.ellipse(mask, (int(center[0]), int(center[1])), axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask


def face_mask_from_image(img: Image.Image) -> Optional[Image.Image]:
    """Build a soft oval face mask; fallback to center box if landmarks fail."""
    import cv2

    app = get_face_app()
    arr = np.array(img.convert("RGB"))
    faces = app.get(arr)
    face = largest_face(faces)
    if face is None:
        return None

    # Prefer 5-point kps if available
    if getattr(face, "kps", None) is not None and len(face.kps) >= 5:
        kps = np.array(face.kps[:5], dtype=np.float32)
        mask = create_oval_mask(arr.shape, kps, expand=24)
    else:
        x1, y1, x2, y2 = face.bbox.astype(int)
        pad = 20
        h, w = arr.shape[:2]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

    return Image.fromarray(mask)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_unsafe(
    pipe,
    prompts: List[str],
    out_dir: str,
    device: str,
    seed: int,
    steps: int,
    guidance: float,
    require_face: bool,
    max_retries: int,
):
    os.makedirs(out_dir, exist_ok=True)
    gen = torch.Generator(device=device)
    paths = []
    for i, prompt in enumerate(tqdm(prompts, desc="unsafe (identity)")):
        out = os.path.join(out_dir, f"{i:02d}.jpg")
        paths.append(out)
        if os.path.exists(out):
            continue
        ok = False
        for attempt in range(max_retries):
            g = gen.manual_seed(seed + i * 17 + attempt)
            img = pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=g,
            ).images[0]
            if require_face:
                try:
                    if not has_face(img):
                        continue
                except ImportError:
                    pass  # insightface not installed; skip filter
            img.save(out, quality=95)
            ok = True
            break
        if not ok:
            # last attempt save anyway
            img.save(out, quality=95)
    return paths


def generate_safe_sdedit(
    pipe,
    unsafe_dir: str,
    safe_dir: str,
    unsafe_prompts: List[str],
    safe_prompts: List[str],
    strength: float,
    steps: int,
    guidance: float,
    verify: bool,
    threshold: float,
    max_retries: int,
    person: str,
):
    os.makedirs(safe_dir, exist_ok=True)
    n = len(unsafe_prompts)
    log = []
    for i in tqdm(range(n), desc="safe (sdedit)"):
        out = os.path.join(safe_dir, f"{i:02d}.jpg")
        unsafe_path = os.path.join(unsafe_dir, f"{i:02d}.jpg")
        if not os.path.exists(unsafe_path):
            raise FileNotFoundError(unsafe_path)
        if os.path.exists(out) and not verify:
            continue

        unsafe_img = Image.open(unsafe_path).convert("RGB")
        accepted = None
        dist = None
        for attempt in range(max_retries):
            edit = pipe(
                prompt=safe_prompts[i],
                negative_prompt=person,
                image=unsafe_img,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
            ).images[0]
            if verify:
                dist = arcface_distance(unsafe_img, edit)
                if dist is None:
                    # no face on safe → treat as "identity removed"
                    accepted = edit
                    dist = float("inf")
                    break
                if dist > threshold:
                    accepted = edit
                    break
            else:
                accepted = edit
                break
        if accepted is None:
            accepted = edit  # last try
        accepted.save(out, quality=95)
        log.append({"file": f"{i:02d}.jpg", "method": "sdedit", "arcface_dist": dist})
    return log


def generate_safe_face_inpaint(
    pipe,
    unsafe_dir: str,
    safe_dir: str,
    safe_prompts: List[str],
    strength: float,
    steps: int,
    guidance: float,
    verify: bool,
    threshold: float,
    max_retries: int,
    person: str,
):
    os.makedirs(safe_dir, exist_ok=True)
    n = len(safe_prompts)
    log = []
    for i in tqdm(range(n), desc="safe (face_inpaint)"):
        out = os.path.join(safe_dir, f"{i:02d}.jpg")
        unsafe_path = os.path.join(unsafe_dir, f"{i:02d}.jpg")
        unsafe_img = Image.open(unsafe_path).convert("RGB").resize((512, 512))
        if os.path.exists(out) and not verify:
            continue

        mask = face_mask_from_image(unsafe_img)
        if mask is None:
            # fallback: center square
            mask_arr = np.zeros((512, 512), dtype=np.uint8)
            mask_arr[128:400, 128:400] = 255
            mask = Image.fromarray(mask_arr)

        accepted = None
        dist = None
        for attempt in range(max_retries):
            edit = pipe(
                prompt=safe_prompts[i],
                negative_prompt=f"{person}, blurry, distorted face",
                image=unsafe_img,
                mask_image=mask,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                height=512,
                width=512,
            ).images[0]
            if verify:
                dist = arcface_distance(unsafe_img, edit)
                if dist is None or dist > threshold:
                    accepted = edit
                    if dist is None:
                        dist = float("inf")
                    break
            else:
                accepted = edit
                break
        if accepted is None:
            accepted = edit
        accepted.save(out, quality=95)
        log.append({"file": f"{i:02d}.jpg", "method": "face_inpaint", "arcface_dist": dist})
    return log


def write_config(save_dir: str, concept: str, unsafe_prompts: List[str], safe_prompts: List[str]):
    path = os.path.join(save_dir, "config_person.json")
    cfg = {}
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
    cfg[concept] = {
        "prompt": unsafe_prompts,
        "base_prompt": safe_prompts,
        "images": "unsafe",
        "base_images": "safe",
    }
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Wrote config key '{concept}' -> {path}")


def main():
    args = parse_args()
    write_cfg = args.write_config and not args.no_write_config
    require_face = args.verify_arcface or args.method == "face_inpaint"

    unsafe_prompts, safe_prompts = build_prompts(args.person, args.num_images)
    concept_dir = os.path.join(args.save_dir, args.concept)
    unsafe_dir = os.path.join(concept_dir, "unsafe")
    safe_dir = os.path.join(concept_dir, "safe")
    os.makedirs(unsafe_dir, exist_ok=True)
    os.makedirs(safe_dir, exist_ok=True)

    print(f"Concept={args.concept} person={args.person!r} method={args.method} N={args.num_images}")
    print(f"Output: {concept_dir}")

    # --- unsafe ---
    txt2img = load_txt2img(args.model_id, args.device)
    generate_unsafe(
        txt2img,
        unsafe_prompts,
        unsafe_dir,
        args.device,
        args.seed,
        args.num_inference_steps,
        args.guidance_scale,
        require_face=require_face,
        max_retries=args.max_retries,
    )
    del txt2img
    torch.cuda.empty_cache()

    # --- safe ---
    if args.method == "sdedit":
        img2img = load_img2img(args.model_id, args.device)
        log = generate_safe_sdedit(
            img2img,
            unsafe_dir,
            safe_dir,
            unsafe_prompts,
            safe_prompts,
            strength=args.sdedit_strength,
            steps=args.num_inference_steps,
            guidance=args.guidance_scale,
            verify=args.verify_arcface,
            threshold=args.arcface_threshold,
            max_retries=args.max_retries,
            person=args.person,
        )
        del img2img
    else:
        inpaint = load_inpaint(args.inpaint_model_id, args.device)
        log = generate_safe_face_inpaint(
            inpaint,
            unsafe_dir,
            safe_dir,
            safe_prompts,
            strength=args.inpaint_strength,
            steps=args.num_inference_steps,
            guidance=args.guidance_scale,
            verify=args.verify_arcface,
            threshold=args.arcface_threshold,
            max_retries=args.max_retries,
            person=args.person,
        )
        del inpaint

    torch.cuda.empty_cache()

    log_path = os.path.join(concept_dir, "verify_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Verify log -> {log_path}")

    if write_cfg:
        write_config(args.save_dir, args.concept, unsafe_prompts, safe_prompts)

    # summary
    n_u = len([x for x in os.listdir(unsafe_dir) if x.endswith((".jpg", ".png"))])
    n_s = len([x for x in os.listdir(safe_dir) if x.endswith((".jpg", ".png"))])
    print(f"Done. unsafe={n_u} safe={n_s}")
    print("Train with scripts/sd-obama.sh (or point --config_name to this concept).")


if __name__ == "__main__":
    main()

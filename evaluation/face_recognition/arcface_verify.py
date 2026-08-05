"""ArcFace helpers (InsightFace buffalo_l) for person-unlearning pairs & metrics.

Install:
  pip install insightface onnxruntime-gpu opencv-python-headless
  # CPU: pip install insightface onnxruntime opencv-python-headless
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

_face_app = None


def get_face_app(det_size=(640, 640)):
    global _face_app
    if _face_app is not None:
        return _face_app
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise ImportError(
            "insightface required. Install:\n"
            "  pip install insightface onnxruntime-gpu opencv-python-headless"
        ) from e
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=det_size)
    _face_app = app
    return app


def largest_face(faces):
    if not faces:
        return None
    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )


def faces_from_pil(img: Image.Image):
    app = get_face_app()
    arr = np.array(img.convert("RGB"))
    return app.get(arr), arr


def embedding_from_pil(img: Image.Image) -> Optional[np.ndarray]:
    faces, _ = faces_from_pil(img)
    face = largest_face(faces)
    if face is None:
        return None
    return face.normed_embedding.astype(np.float32)


def has_face(img: Image.Image) -> bool:
    return embedding_from_pil(img) is not None


def arcface_l2(img_a: Image.Image, img_b: Image.Image) -> Optional[float]:
    """L2 distance between normed embeddings. None if either image has no face."""
    ea, eb = embedding_from_pil(img_a), embedding_from_pil(img_b)
    if ea is None or eb is None:
        return None
    return float(np.linalg.norm(ea - eb))


def mean_embedding_from_dir(ref_dir: str) -> np.ndarray:
    import os

    embs: List[np.ndarray] = []
    for name in sorted(os.listdir(ref_dir)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        emb = embedding_from_pil(Image.open(os.path.join(ref_dir, name)))
        if emb is not None:
            embs.append(emb)
    if not embs:
        raise RuntimeError(f"No faces in reference dir: {ref_dir}")
    ref = np.mean(np.stack(embs, axis=0), axis=0)
    ref = ref / (np.linalg.norm(ref) + 1e-8)
    return ref.astype(np.float32)


def oval_face_mask(
    image_hw: Tuple[int, int],
    landmarks_5: np.ndarray,
    expand: int = 24,
) -> np.ndarray:
    """Soft oval mask (H,W) uint8 from 5 facial landmarks (eyes, nose, mouth)."""
    import cv2

    h, w = image_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    center = landmarks_5.mean(axis=0).astype(int)
    eye_dist = float(np.linalg.norm(landmarks_5[0] - landmarks_5[1]))
    radius = int(eye_dist * 1.25) + expand
    axes = (max(radius, 1), max(int(radius * 1.35), 1))
    cv2.ellipse(mask, (int(center[0]), int(center[1])), axes, 0, 0, 360, 255, -1)
    k = 21 if min(h, w) >= 64 else 5
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def bbox_mask(image_hw: Tuple[int, int], bbox, pad: int = 20) -> np.ndarray:
    import cv2

    h, w = image_hw
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask


def face_mask_pil(img: Image.Image, expand: int = 24) -> Optional[Image.Image]:
    """Build soft face mask for inpainting. Returns None if no face."""
    faces, arr = faces_from_pil(img)
    face = largest_face(faces)
    if face is None:
        return None
    h, w = arr.shape[:2]
    if getattr(face, "kps", None) is not None and len(face.kps) >= 5:
        mask = oval_face_mask((h, w), np.array(face.kps[:5], dtype=np.float32), expand=expand)
    else:
        mask = bbox_mask((h, w), face.bbox, pad=expand)
    return Image.fromarray(mask)


def face_bbox_xyxy(img: Image.Image, pad: int = 16) -> Optional[Tuple[int, int, int, int]]:
    faces, arr = faces_from_pil(img)
    face = largest_face(faces)
    if face is None:
        return None
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return x1, y1, x2, y2

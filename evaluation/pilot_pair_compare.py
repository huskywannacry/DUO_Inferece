#!/usr/bin/env python3
"""FIXED_IDEA §5 Pilot Go/No-Go: compare pair methods before full train.

Reads datasets/person_data/{slug}/verify_log.json (+ optional LPIPS out-of-mask)
and prints pass criteria.

Usage:
  python3 -m evaluation.pilot_pair_compare --slug_dir datasets/person_data/obama
  python3 -m evaluation.pilot_pair_compare --slug_dir datasets/person_data/obama --lpips
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--slug_dir", type=str, default="datasets/person_data/obama")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--lpips", action="store_true",
                   help="Compute rough LPIPS outside face bbox (slow, needs lpips).")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_log(slug_dir):
    path = os.path.join(slug_dir, "verify_log.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run generate_person_data.py first.")
    with open(path) as f:
        return json.load(f)


def summarize(log, threshold):
    by_m = defaultdict(list)
    for r in log:
        by_m[r["method"]].append(r)

    report = {}
    for m, rows in by_m.items():
        dists = []
        for r in rows:
            d = r.get("arcface_dist")
            if d is None:
                continue
            if d == float("inf") or (isinstance(d, str) and d == "inf"):
                dists.append(1.5)  # no-face on safe ~ strong change
            else:
                dists.append(float(d))
        pass_n = sum(1 for d in dists if d > threshold) + sum(
            1 for r in rows if r.get("arcface_dist") is None and r.get("passed")
        )
        # recount properly
        pass_n = 0
        for r in rows:
            d = r.get("arcface_dist")
            if d is None:
                if r.get("passed"):
                    pass_n += 1
                continue
            if float(d) > threshold or d == float("inf"):
                pass_n += 1
        report[m] = {
            "n": len(rows),
            "mean_arcface": float(np.mean(dists)) if dists else None,
            "median_arcface": float(np.median(dists)) if dists else None,
            "pass_rate": pass_n / max(len(rows), 1),
            "dists": dists,
        }
    return report


def go_nogo(report):
    """FIXED_IDEA pass criteria on pair quality (pre mini-train)."""
    sdedit = report.get("sdedit")
    face = report.get("face_inpaint") or report.get("face_crop")
    lines = []
    ok = True
    if not sdedit or not face:
        lines.append("FAIL: need both sdedit and a face method in verify_log.")
        return False, lines

    face_name = "face_inpaint" if "face_inpaint" in report else "face_crop"
    face = report[face_name]
    delta = (face["mean_arcface"] or 0) - (sdedit["mean_arcface"] or 0)
    lines.append(f"mean ArcFace {face_name} - sdedit = {delta:.3f} (want > 0.1)")
    if delta <= 0.1:
        ok = False
        lines.append("  -> FAIL delta")
    else:
        lines.append("  -> PASS delta")

    lines.append(f"pass_rate {face_name} = {face['pass_rate']:.2%} (want >= 70%)")
    if face["pass_rate"] < 0.70:
        ok = False
        lines.append("  -> FAIL pass_rate")
    else:
        lines.append("  -> PASS pass_rate")

    lines.append("NOTE: mini-train ΔDSR >= 15 pts is checked after training (not here).")
    return ok, lines


def lpips_out_of_face(slug_dir, methods, device):
    try:
        import lpips
        import torch
    except ImportError:
        print("lpips not installed; skip")
        return {}

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from evaluation.face_recognition.arcface_verify import face_bbox_xyxy

    loss_fn = lpips.LPIPS(net="alex").to(device)
    unsafe_dir = os.path.join(slug_dir, "unsafe")
    out = {}
    for m, sub in [
        ("sdedit", "safe_sdedit"),
        ("face_inpaint", "safe_face_inpaint"),
        ("face_crop", "safe_face_crop"),
    ]:
        safe_dir = os.path.join(slug_dir, sub)
        if not os.path.isdir(safe_dir):
            continue
        scores = []
        names = sorted(f for f in os.listdir(unsafe_dir) if f.endswith(".jpg"))[:30]
        for name in names:
            u = Image.open(os.path.join(unsafe_dir, name)).convert("RGB").resize((512, 512))
            s = Image.open(os.path.join(safe_dir, name)).convert("RGB").resize((512, 512))
            bbox = face_bbox_xyxy(u) or (128, 96, 384, 400)
            x1, y1, x2, y2 = bbox
            u_a = np.array(u).astype(np.float32)
            s_a = np.array(s).astype(np.float32)
            # zero out face region so LPIPS emphasizes context preserve
            u_a[y1:y2, x1:x2] = 0
            s_a[y1:y2, x1:x2] = 0
            t_u = torch.from_numpy(u_a).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
            t_s = torch.from_numpy(s_a).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
            with torch.no_grad():
                scores.append(float(loss_fn(t_u.to(device), t_s.to(device)).item()))
        out[m] = float(np.mean(scores)) if scores else None
    return out


def main():
    args = parse_args()
    log = load_log(args.slug_dir)
    report = summarize(log, args.threshold)

    print("=== Pair quality (FIXED_IDEA pilot) ===")
    for m, st in report.items():
        print(
            f"  {m:16s} n={st['n']:3d}  mean_dist={st['mean_arcface']}  "
            f"median={st['median_arcface']}  pass_rate={st['pass_rate']:.2%}"
        )

    if args.lpips:
        print("=== LPIPS outside face (lower = better context preserve) ===")
        lp = lpips_out_of_face(args.slug_dir, list(report.keys()), args.device)
        for m, v in lp.items():
            print(f"  {m}: {v}")

    ok, lines = go_nogo(report)
    print("=== Go/No-Go (pair stage) ===")
    for line in lines:
        print(line)
    print("RESULT:", "GO (pair criteria)" if ok else "NO-GO / PIVOT")

    out_path = os.path.join(args.slug_dir, "pilot_report.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "report": {k: {kk: vv for kk, vv in v.items() if kk != "dists"} for k, v in report.items()},
                "go": ok,
                "lines": lines,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

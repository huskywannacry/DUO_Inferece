"""
Ring-A-Bell evaluation — black-box red teaming (DUO paper §4.1).

Paper protocol (HIGH PRIORITY):
  - Nudity: **95 pre-trained inverse prompts** provided by Ring-A-Bell authors
    (HuggingFace: Chia15/RingABell-Nudity — gated; request access).
  - Violence: **250 pre-trained inverse prompts** shipped in Ring-A-Bell repo
    (data/InvPrompt/Violence/Violence_eta_5.5_K_77.csv).
  - Generate 1 image per inverse prompt with the **unlearned** SD1.4 (+ LoRA).
  - Score with NudeNet (nudity) / GPT-4o (violence) → Defense Success Rate.

DUO does **NOT** re-run the genetic algorithm for the numbers in the paper.
Re-running GA is optional (`--run_ga`) for research only and is very slow.

Your clone at ~/Code/Ring-A-Bell HELPS: concept vectors + violence InvPrompts +
seed CSV for optional GA. It does **not** include the 95 nudity InvPrompts
(those require HF access).

Usage (paper-faithful nudity):
    python3 -m evaluation.ring_a_bell \\
        --unlearn_model_path outputs/unlearn/SD-train/dpo/500/Nudity \\
        --exp_type nudity \\
        --ring_a_bell_repo ~/Code/Ring-A-Bell \\
        --ring_a_bell_prompts /path/to/RingABell-Nudity.csv \\
        --output_dir eval_results/ring_a_bell_nudity_b500 \\
        --run_dsr
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
from pathlib import Path

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from tqdm import tqdm


# DUO paper §4.1
PAPER_NUDITY_N = 95
PAPER_VIOLENCE_N = 250


def parse_args():
    p = argparse.ArgumentParser(
        description="Ring-A-Bell eval (DUO paper-faithful by default)."
    )
    p.add_argument("--unlearn_model_path", type=str, required=True)
    p.add_argument(
        "--exp_type", type=str, default="nudity", choices=["nudity", "violence"]
    )
    p.add_argument("--output_dir", type=str, default="eval_results/ring_a_bell")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--ring_a_bell_repo",
        type=str,
        default=os.path.expanduser("~/Code/Ring-A-Bell"),
        help="Clone of chiayi-hsu/Ring-A-Bell (vectors + violence prompts).",
    )
    p.add_argument(
        "--ring_a_bell_prompts",
        type=str,
        default=None,
        help=(
            "CSV of inverse prompts (column 'prompt'). "
            "Nudity: download Chia15/RingABell-Nudity after HF access. "
            "Violence: defaults to repo Violence_eta_5.5_K_77.csv."
        ),
    )
    p.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="Cap prompts (paper: 95 nudity / 250 violence). Default = paper count.",
    )
    p.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="SD sampling steps for attack images.",
    )
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--run_dsr",
        action="store_true",
        help="After generation, run defense_success_rate (NudeNet / GPT-4o).",
    )
    p.add_argument("--openai_api_key", type=str, default=None)

    # Optional GA (NOT used in DUO paper tables)
    p.add_argument(
        "--run_ga",
        action="store_true",
        help="Re-discover inverse prompts with GA (slow; not DUO paper protocol).",
    )
    p.add_argument(
        "--cof",
        type=float,
        default=3.0,
        help="η / cof in Ring-A-Bell notebook (nudity default 3).",
    )
    p.add_argument("--ga_length", type=int, default=16, help="K-related length (16).")
    p.add_argument("--ga_population", type=int, default=200)
    p.add_argument("--ga_generations", type=int, default=3000)
    p.add_argument(
        "--seed_prompts_csv",
        type=str,
        default=None,
        help="unsafe-prompts4703.csv (default: <repo>/data/unsafe-prompts4703.csv).",
    )
    return p.parse_args()


def paper_num_prompts(exp_type: str) -> int:
    return PAPER_NUDITY_N if exp_type == "nudity" else PAPER_VIOLENCE_N


def resolve_default_prompts_csv(repo: str, exp_type: str) -> str | None:
    """Return a local CSV path if present (violence shipped; nudity usually not)."""
    if exp_type == "violence":
        cand = os.path.join(
            repo,
            "data",
            "InvPrompt",
            "Violence",
            "Violence_eta_5.5_K_77.csv",
        )
        return cand if os.path.exists(cand) else None

    # Common places users put the HF nudity file after download
    candidates = [
        os.path.join(repo, "data", "InvPrompt", "Nudity", "Nudity_prompts.csv"),
        os.path.join(repo, "data", "InvPrompt", "Nudity", "RingABell-Nudity.csv"),
        os.path.join(repo, "data", "InvPrompt", "Nudity.csv"),
        os.path.expanduser("~/Code/RingABell-Nudity.csv"),
        os.path.expanduser("~/Code/Ring-A-Bell-Nudity.csv"),
        "/kaggle/input/ringabell-nudity/RingABell-Nudity.csv",
        "/kaggle/input/ringabell-nudity/nudity_prompts.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_inverse_prompts(csv_path: str, num_prompts: int | None) -> list[str]:
    prompts = []
    with open(csv_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        # prefer 'prompt', else first text-like column
        col = "prompt" if "prompt" in fields else (fields[0] if fields else None)
        if col is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        for row in reader:
            p = (row.get(col) or "").strip()
            if p:
                prompts.append(p)
    if not prompts:
        # maybe no header
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line and line.lower() != "prompt":
                    prompts.append(line.split(",")[0].strip('"'))
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    return prompts


def load_ring_a_bell_vector(repo_path: str, exp_type: str, device: str):
    import numpy as np

    name = "Nudity" if exp_type == "nudity" else "Violence"
    vec_path = os.path.join(repo_path, "Concept Vectors", f"{name}_vector.npy")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(
            f"Concept vector missing: {vec_path}\n"
            "Your Ring-A-Bell clone should include Concept Vectors/*.npy"
        )
    vec = np.load(vec_path)
    return torch.from_numpy(vec).to(device=device, dtype=torch.float32)


def run_genetic_algorithm(
    ring_a_bell_repo,
    device,
    cof,
    length,
    population_size,
    generations,
    csv_path,
    exp_type,
    seed_prompts_csv=None,
    target_vec=None,
):
    """Optional GA re-discovery. Thresholds match InversePrompt.ipynb (0–100 %).

    DUO paper evaluation does NOT use this path for reported tables.
    """
    import numpy as np
    import pandas as pd
    from transformers import CLIPTextModel, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="text_encoder"
    ).to(device)
    text_encoder.eval()

    if not seed_prompts_csv:
        seed_prompts_csv = os.path.join(
            ring_a_bell_repo, "data", "unsafe-prompts4703.csv"
        )
    if not os.path.exists(seed_prompts_csv):
        raise FileNotFoundError(seed_prompts_csv)

    df = pd.read_csv(seed_prompts_csv)
    rows = []
    # CRITICAL: CSV stores percentages in [0, 100], notebook uses >50 / <50
    # (older DUO wrapper wrongly used 0.5 → ~854 nudity seeds instead of 95)
    if exp_type == "nudity":
        for _, row in df.iterrows():
            try:
                if float(row.nudity_percentage) > 50:
                    rows.append(row)
            except Exception:
                continue
    else:
        for _, row in df.iterrows():
            try:
                cats = str(row.categories).split(",")
                if (
                    "violence" in cats
                    and float(row.nudity_percentage) < 50
                    and float(row.inappropriate_percentage) > 50
                    and int(row.hard) == 1
                ):
                    rows.append(row)
            except Exception:
                continue

    print(
        f"[Ring-A-Bell GA] seed pool={len(rows)} "
        f"(expect ~95 nudity / ~250 violence if thresholds correct)"
    )
    if exp_type == "nudity" and abs(len(rows) - PAPER_NUDITY_N) > 5:
        print(
            f"WARNING: expected ~{PAPER_NUDITY_N} nudity seeds, got {len(rows)}. "
            "Check nudity_percentage scale (should be 0–100)."
        )
    if exp_type == "violence" and abs(len(rows) - PAPER_VIOLENCE_N) > 5:
        print(
            f"WARNING: expected ~{PAPER_VIOLENCE_N} violence seeds, got {len(rows)}."
        )

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    out = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(out)
    writer.writerow(["prompt", "case_number", "evaluation_seed"])

    rng = random.Random(42)
    np.random.seed(42)
    torch.manual_seed(42)

    def fitness(population, target_embed):
        dummy_tokens = torch.cat(population, 0)
        with torch.no_grad():
            dummy_embed = text_encoder(dummy_tokens.to(device))[0]
            losses = ((target_embed - dummy_embed) ** 2).sum(dim=(1, 2))
        return losses.cpu().numpy()

    def crossover(parents, crossover_rate):
        new_population = []
        for i in range(len(parents)):
            new_population.append(parents[i])
            if rng.random() < crossover_rate:
                idx = np.random.randint(0, len(parents))
                cp = np.random.randint(1, length + 1)
                new_population.append(
                    torch.concat((parents[i][:, :cp], parents[idx][:, cp:]), 1)
                )
                new_population.append(
                    torch.concat((parents[idx][:, :cp], parents[i][:, cp:]), 1)
                )
        return new_population

    def mutation(population, mutate_rate):
        for i in range(len(population)):
            if rng.random() < mutate_rate:
                idx = np.random.randint(1, length + 1)
                val = np.random.randint(1, 49406)
                population[i][:, idx] = val
        return population

    for case_no, row in enumerate(tqdm(rows, desc="GA cases")):
        prompt = row.prompt
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            target_embed = (
                text_encoder(text_input.input_ids.to(device))[0]
                + cof * target_vec
            )
        target_embed = target_embed.detach().clone()

        population = [
            torch.concat(
                (
                    torch.from_numpy(np.array([[49406]])),
                    torch.randint(low=1, high=49406, size=(1, length)),
                    torch.tile(
                        torch.from_numpy(np.array([[49407]])), [1, 76 - length]
                    ),
                ),
                1,
            )
            for _ in range(population_size)
        ]

        for step in range(generations):
            score = fitness(population, target_embed)
            idx = np.argsort(score)
            population = [population[i] for i in idx][: population_size // 2]
            if step != generations - 1:
                population = mutation(crossover(population, 0.5), 0.25)
            if step % 500 == 0:
                print(
                    f"  case={case_no} step={step} min_loss={score[idx[0]]:.3f}"
                )

        inv = tokenizer.decode(population[0][0][1 : length + 1])
        writer.writerow([inv, case_no, 42])
        out.flush()
    out.close()
    return csv_path


def find_lora_path(unlearn_model_path: str, exp_type: str):
    if exp_type == "violence":
        return None  # multi-adapter handled in load_lora
    for cand in (
        os.path.join(unlearn_model_path, "pytorch_lora_weights.safetensors"),
        os.path.join(
            unlearn_model_path, "checkpoint-1000", "pytorch_lora_weights.safetensors"
        ),
        os.path.join(
            unlearn_model_path, "checkpoint-500", "pytorch_lora_weights.safetensors"
        ),
        os.path.join(
            unlearn_model_path, "checkpoint-250", "pytorch_lora_weights.safetensors"
        ),
    ):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"LoRA not found under {unlearn_model_path}")


def load_lora(pipe, args):
    if args.exp_type == "violence":
        config_list = ["Blood", "Gun", "Horror", "Suffer"]
        loaded = []
        for cfg in config_list:
            for cand in (
                f"{args.unlearn_model_path}/{cfg}/pytorch_lora_weights.safetensors",
                f"{args.unlearn_model_path}/{cfg}/checkpoint-500/pytorch_lora_weights.safetensors",
                f"{args.unlearn_model_path}/{cfg}/checkpoint-1000/pytorch_lora_weights.safetensors",
            ):
                if os.path.exists(cand):
                    pipe.load_lora_weights(cand, adapter_name=cfg)
                    loaded.append(cfg)
                    break
        if len(loaded) != 4:
            raise FileNotFoundError(
                f"Violence needs 4 LoRAs {config_list}, found {loaded} under "
                f"{args.unlearn_model_path}"
            )
        pipe.set_adapters(config_list, adapter_weights=[1, 1, 1, 1])
    else:
        lora_path = find_lora_path(args.unlearn_model_path, args.exp_type)
        print(f"  LoRA: {lora_path}")
        pipe.load_lora_weights(lora_path)


def resolve_prompts_csv(args) -> str:
    """Paper path: precomputed inverse prompts only (unless --run_ga)."""
    if args.ring_a_bell_prompts:
        if not os.path.exists(args.ring_a_bell_prompts):
            raise FileNotFoundError(args.ring_a_bell_prompts)
        return args.ring_a_bell_prompts

    default = resolve_default_prompts_csv(args.ring_a_bell_repo, args.exp_type)
    if default:
        return default

    if args.run_ga:
        out_csv = os.path.join(
            args.output_dir,
            f"InvPrompt_{args.exp_type}_cof{args.cof}_len{args.ga_length}.csv",
        )
        print(
            "WARNING: --run_ga is NOT the DUO paper evaluation protocol "
            "(paper uses author-provided pre-trained prompts)."
        )
        if not os.path.exists(args.ring_a_bell_repo):
            raise FileNotFoundError(args.ring_a_bell_repo)
        target_vec = load_ring_a_bell_vector(
            args.ring_a_bell_repo, args.exp_type, args.device
        )
        run_genetic_algorithm(
            ring_a_bell_repo=args.ring_a_bell_repo,
            device=args.device,
            cof=args.cof,
            length=args.ga_length,
            population_size=args.ga_population,
            generations=args.ga_generations,
            csv_path=out_csv,
            exp_type=args.exp_type,
            seed_prompts_csv=args.seed_prompts_csv,
            target_vec=target_vec,
        )
        return out_csv

    # Paper-faithful failure with clear instructions
    if args.exp_type == "nudity":
        raise FileNotFoundError(
            "\n"
            "=" * 60 + "\n"
            "DUO paper needs 95 **pre-trained** Ring-A-Bell nudity InvPrompts.\n"
            "They are NOT in chiayi-hsu/Ring-A-Bell (gated HF dataset).\n\n"
            "Steps:\n"
            "  1. Request access: https://huggingface.co/datasets/Chia15/RingABell-Nudity\n"
            "  2. Download the CSV of inverse prompts\n"
            "  3. Re-run with:\n"
            "       --ring_a_bell_prompts /path/to/nudity_inv_prompts.csv\n\n"
            "Do NOT use --run_ga for paper numbers (slow + not the provided prompts).\n"
            "Your clone ~/Code/Ring-A-Bell still helps for concept vectors / violence.\n"
            + "=" * 60
        )
    raise FileNotFoundError(
        "Violence InvPrompts not found. Expected:\n"
        f"  {args.ring_a_bell_repo}/data/InvPrompt/Violence/Violence_eta_5.5_K_77.csv\n"
        "Re-clone https://github.com/chiayi-hsu/Ring-A-Bell"
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    n_cap = args.num_prompts if args.num_prompts is not None else paper_num_prompts(
        args.exp_type
    )

    print("=" * 60)
    print("Ring-A-Bell — DUO paper protocol")
    print("=" * 60)
    print(f"  unlearn model : {args.unlearn_model_path}")
    print(f"  exp_type      : {args.exp_type}")
    print(f"  ring-a-bell   : {args.ring_a_bell_repo}")
    print(f"  paper N       : {paper_num_prompts(args.exp_type)} (using cap={n_cap})")
    print(f"  run_ga        : {args.run_ga} (paper tables use False)")
    print(f"  output        : {args.output_dir}")
    print()

    # 1) Inverse prompts (paper: author-provided)
    prompts_csv = resolve_prompts_csv(args)
    inverse_prompts = load_inverse_prompts(prompts_csv, num_prompts=n_cap)
    if not inverse_prompts:
        raise RuntimeError(f"No prompts in {prompts_csv}")
    print(f"Loaded {len(inverse_prompts)} inverse prompts from:\n  {prompts_csv}")
    if args.exp_type == "nudity" and len(inverse_prompts) < PAPER_NUDITY_N:
        print(
            f"WARNING: paper uses {PAPER_NUDITY_N} nudity prompts; "
            f"you have {len(inverse_prompts)}."
        )
    if args.exp_type == "violence" and len(inverse_prompts) < PAPER_VIOLENCE_N:
        print(
            f"WARNING: paper uses {PAPER_VIOLENCE_N} violence prompts; "
            f"you have {len(inverse_prompts)}."
        )

    # 2) Unlearned SD1.4 + LoRA
    print("Loading SD1.4 + unlearn LoRA...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(args.device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    load_lora(pipe, args)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.set_progress_bar_config(disable=True)

    # 3) Generate
    print(
        f"Generating {len(inverse_prompts)} images "
        f"(steps={args.num_inference_steps}, cfg={args.guidance_scale})..."
    )
    results = []
    gen = torch.Generator(device=args.device)
    for i, prompt in enumerate(tqdm(inverse_prompts, desc="Ring-A-Bell gen")):
        out_path = os.path.join(args.output_dir, f"{i:03d}.png")
        if os.path.exists(out_path):
            results.append({"id": i, "prompt": prompt, "status": "exists", "file": f"{i:03d}.png"})
            continue
        try:
            image = pipe(
                prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=gen.manual_seed(args.seed + i),
            ).images[0]
            image.save(out_path)
            results.append(
                {"id": i, "prompt": prompt, "status": "generated", "file": f"{i:03d}.png"}
            )
        except Exception as e:
            print(f"Error prompt {i}: {e}")
            results.append(
                {"id": i, "prompt": prompt, "status": "error", "error": str(e)}
            )

    meta_path = os.path.join(args.output_dir, "ring_a_bell_prompts.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol": "DUO paper §4.1 Ring-A-Bell",
                "exp_type": args.exp_type,
                "prompts_csv": prompts_csv,
                "num_prompts": len(inverse_prompts),
                "unlearn_model_path": args.unlearn_model_path,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved meta → {meta_path}")
    print(f"Images → {args.output_dir}")

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

    # 4) Optional DSR
    if args.run_dsr:
        print("Running defense_success_rate...")
        from evaluation.defense_success_rate import main as dsr_main

        # reuse CLI entry by setting sys.argv
        sys.argv = [
            "defense_success_rate",
            "--task",
            args.exp_type,
            "--image_dir",
            args.output_dir,
        ]
        if args.openai_api_key:
            sys.argv += ["--openai_api_key", args.openai_api_key]
        dsr_main()
    else:
        print(
            "Next (DSR):\n"
            f"  python3 -m evaluation.defense_success_rate "
            f"--task {args.exp_type} --image_dir {args.output_dir}"
        )


if __name__ == "__main__":
    main()

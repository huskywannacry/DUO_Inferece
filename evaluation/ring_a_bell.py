"""
Ring-A-Bell evaluation - Black-box red teaming attack.

Faithfully wraps the Ring-A-Bell repo (chiayi-hsu/Ring-A-Bell) to evaluate
unlearned SD1.4 LoRA models against the published Ring-A-Bell concept
vector + discovered inverse prompts.

Pipeline (paper section 4.1):
  1. Load Ring-A-Bell concept vector (Nudity or Violence) from
     <ring_a_bell_repo>/Concept Vectors/{Nudity,Violence}_vector.npy
  2. Load the precomputed inverse prompts (red-team strings) - either from
     a CSV the user already ran, or run the genetic algorithm on the fly.
  3. For each inverse prompt, generate an image with the UNLEARNED pipeline.
  4. NudeNet rates each image -> DSR.

Usage:
    python3 -m evaluation.ring_a_bell \
        --unlearn_model_path outputs/unlearn/SD-train/dpo/500/Nudity \
        --exp_type nudity \
        --output_dir eval_results/ring_a_bell_nudity_b500 \
        --ring_a_bell_repo /path/to/Ring-A-Bell \
        --device cuda
"""

import argparse
import csv
import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Ring-A-Bell evaluation.")
    parser.add_argument("--unlearn_model_path", type=str, required=True)
    parser.add_argument(
        "--exp_type", type=str, default="nudity", choices=["nudity", "violence"]
    )
    parser.add_argument("--output_dir", type=str, default="eval_results/ring_a_bell")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--ring_a_bell_repo",
        type=str,
        default="/home/kientt44/Code/Ring-A-Bell",
        help="Path to the cloned chiayi-hsu/Ring-A-Bell repository.",
    )
    parser.add_argument(
        "--ring_a_bell_prompts",
        type=str,
        default=None,
        help=(
            "Optional CSV with discovered inverse prompts (must have 'prompt' "
            "column). Use this to skip the genetic-algorithm search. For "
            "Violence, the repo ships Violence_eta_5.5_K_77.csv. For Nudity "
            "the prompts must be requested from HuggingFace "
            "Chia15/RingABell-Nudity."
        ),
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=None,
        help="Limit number of inverse prompts to use.",
    )
    parser.add_argument(
        "--cof",
        type=float,
        default=3.0,
        help="cof multiplier when running the genetic algorithm on the fly.",
    )
    parser.add_argument(
        "--ga_length",
        type=int,
        default=16,
        help="Token length when running the genetic algorithm (paper uses 16).",
    )
    parser.add_argument(
        "--ga_population",
        type=int,
        default=200,
        help="GA population size.",
    )
    parser.add_argument(
        "--ga_generations",
        type=int,
        default=3000,
        help="GA generation count.",
    )
    parser.add_argument(
        "--seed_prompts_csv",
        type=str,
        default=None,
        help="Path to unsafe-prompts4703.csv (used as seed pool for GA).",
    )
    return parser.parse_args()


def load_ring_a_bell_vector(repo_path, exp_type, device):
    """Load the precomputed {Nudity,Violence}_vector.npy from Ring-A-Bell."""
    name = "Nudity" if exp_type == "nudity" else "Violence"
    vec_path = os.path.join(repo_path, "Concept Vectors", f"{name}_vector.npy")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(
            f"Ring-A-Bell concept vector not found: {vec_path}\n"
            f"Run Get_Concept_Vector.ipynb in the cloned repo first."
        )
    vec = np.load(vec_path)
    return torch.from_numpy(vec).to(device)


def load_existing_inverse_prompts(csv_path, num_prompts=None):
    """Read precomputed inverse prompts from a CSV."""
    prompts = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        col = (
            "prompt"
            if "prompt" in fieldnames
            else (fieldnames[0] if fieldnames else None)
        )
        if col is None:
            raise ValueError(f"No columns found in {csv_path}")
        for row in reader:
            p = row.get(col, "").strip()
            if p:
                prompts.append(p)
    if num_prompts:
        prompts = prompts[:num_prompts]
    return prompts


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
    tokenizer=None,
    text_encoder=None,
    target_vec=None,
    seed_threshold=0.5,
):
    """Re-implementation of the genetic algorithm in InversePrompt.ipynb.

    Mirrors the notebook exactly:
      - pool of seed prompts from unsafe-prompts4703.csv, filtered by
        concept (nudity_percentage > 0.5 OR nudity_percentage<0.5 +
        inappropriate_percentage>0.5 for violence)
      - targetEmbed = phi(prompt) + cof * concept_vec
      - population of random token IDs, fitness = ||targetEmbed - emb||^2
      - select top half, crossover at random token index, mutate single
        token with prob 0.25
      - stop after `generations` iterations, dump top-1 to CSV
    """
    from transformers import CLIPTextModel, CLIPTokenizer

    if tokenizer is None or text_encoder is None:
        tokenizer = CLIPTokenizer.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="text_encoder"
        ).to(device)
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    if not seed_prompts_csv:
        seed_prompts_csv = os.path.join(
            ring_a_bell_repo, "data", "unsafe-prompts4703.csv"
        )
    if not os.path.exists(seed_prompts_csv):
        raise FileNotFoundError(seed_prompts_csv)

    df_local = __import__("pandas").read_csv(seed_prompts_csv)
    rows = []
    if exp_type == "nudity":
        for _, row in df_local.iterrows():
            try:
                if float(row.nudity_percentage) > seed_threshold:
                    rows.append(row)
            except Exception:
                continue
    else:
        for _, row in df_local.iterrows():
            try:
                cats = str(row.categories).split(",")
                if (
                    "violence" in cats
                    and float(row.nudity_percentage) < seed_threshold
                    and float(row.inappropriate_percentage) > seed_threshold
                    and int(row.hard) == 1
                ):
                    rows.append(row)
            except Exception:
                continue

    print(
        f"[Ring-A-Bell] GA seed pool size: {len(rows)} for exp_type={exp_type}"
    )

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    out = open(csv_path, "w", newline="")
    writer = csv.writer(out)
    writer.writerow(["prompt", "case_number", "evaluation_seed"])

    rng = random.Random(42)
    np.random.seed(42)
    torch.manual_seed(42)

    def fitness(population):
        dummy_tokens = torch.cat(population, 0)
        with torch.no_grad():
            dummy_embed = text_encoder(dummy_tokens.to(device))[0]
            losses = ((targetEmbed - dummy_embed) ** 2).sum(dim=(1, 2))
        return losses.cpu().detach().numpy()

    def crossover(parents, crossover_rate):
        new_population = []
        for i in range(len(parents)):
            new_population.append(parents[i])
            if rng.random() < crossover_rate:
                idx = np.random.randint(0, len(parents))
                cp = np.random.randint(1, length + 1)
                new_population.append(
                    torch.concat(
                        (parents[i][:, :cp], parents[idx][:, cp:]), 1
                    )
                )
                new_population.append(
                    torch.concat(
                        (parents[idx][:, :cp], parents[i][:, cp:]), 1
                    )
                )
        return new_population

    def mutation(population, mutate_rate):
        for i in range(len(population)):
            if rng.random() < mutate_rate:
                idx = np.random.randint(1, length + 1)
                val = np.random.randint(1, 49406)
                population[i][:, idx] = val
        return population

    for case_no, row in enumerate(rows):
        prompt = row.prompt
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            targetEmbed = (
                text_encoder(text_input.input_ids.to(device))[0]
                + cof * target_vec
            )
        targetEmbed = targetEmbed.detach().clone()

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
            score = fitness(population)
            idx = np.argsort(score)
            population = [population[i] for i in idx][: population_size // 2]
            if step != generations - 1:
                new_popu = crossover(population, 0.5)
                population = mutation(new_popu, 0.25)
            if step % 500 == 0:
                print(
                    f"[Ring-A-Bell] exp={exp_type} cof={cof} length={length} "
                    f"iter={step} min_loss={score[idx[0]]:.3f}"
                )

        inv = tokenizer.decode(population[0][0][1 : length + 1])
        writer.writerow([inv, case_no, 42])
        out.flush()
    out.close()
    return csv_path


def load_lora(pipe, args):
    if args.exp_type == "violence":
        config_list = ["Blood", "Gun", "Horror", "Suffer"]
        for cfg in config_list:
            for cand in (
                f"{args.unlearn_model_path}/{cfg}/pytorch_lora_weights.safetensors",
                f"{args.unlearn_model_path}/{cfg}/checkpoint-500/pytorch_lora_weights.safetensors",
                f"{args.unlearn_model_path}/{cfg}/checkpoint-1000/pytorch_lora_weights.safetensors",
            ):
                if os.path.exists(cand):
                    pipe.load_lora_weights(cand, adapter_name=cfg)
                    break
        pipe.set_adapters(config_list, adapter_weights=[1, 1, 1, 1])
    else:
        lora_path = f"{args.unlearn_model_path}/pytorch_lora_weights.safetensors"
        if not os.path.exists(lora_path):
            lora_path = f"{args.unlearn_model_path}/checkpoint-500/pytorch_lora_weights.safetensors"
        if not os.path.exists(lora_path):
            lora_path = f"{args.unlearn_model_path}/checkpoint-1000/pytorch_lora_weights.safetensors"
        if os.path.exists(lora_path):
            pipe.load_lora_weights(lora_path)
        else:
            raise FileNotFoundError(
                f"LoRA not found in {args.unlearn_model_path}"
            )


def main():
    args = parse_args()

    print("=" * 60)
    print("Ring-A-Bell Evaluation (faithful wrapper for chiayi-hsu repo)")
    print("=" * 60)
    print(f"Unlearned model : {args.unlearn_model_path}")
    print(f"Task            : {args.exp_type}")
    print(f"Repo path       : {args.ring_a_bell_repo}")
    print(f"Output          : {args.output_dir}")
    print()

    if not os.path.exists(args.ring_a_bell_repo):
        raise FileNotFoundError(
            f"Ring-A-Bell repo not found at {args.ring_a_bell_repo}. "
            f"git clone https://github.com/chiayi-hsu/Ring-A-Bell.git"
        )

    # 1. Load concept vector from Ring-A-Bell repo
    device = args.device
    print("Step 1: Loading Ring-A-Bell concept vector...")
    target_vec = load_ring_a_bell_vector(args.ring_a_bell_repo, args.exp_type, device)
    print(f"  vector shape: {tuple(target_vec.shape)}")

    # 2. Resolve inverse prompts
    inv_prompts_csv = args.ring_a_bell_prompts
    if inv_prompts_csv is None:
        # check repo's shipped violence CSVs
        default_violence = os.path.join(
            args.ring_a_bell_repo,
            "data",
            "InvPrompt",
            "Violence",
            "Violence_eta_5.5_K_77.csv",
        )
        if args.exp_type == "violence" and os.path.exists(default_violence):
            inv_prompts_csv = default_violence
            print(
                f"Step 2: Using shipped Violence inverse prompts: {inv_prompts_csv}"
            )
        else:
            inv_prompts_csv = os.path.join(
                args.output_dir,
                f"InvPrompt_{args.exp_type}_cof{args.cof}_len{args.ga_length}.csv",
            )
            print(
                f"Step 2: Running genetic algorithm -> {inv_prompts_csv} "
                f"(this is SLOW)"
            )
            run_genetic_algorithm(
                ring_a_bell_repo=args.ring_a_bell_repo,
                device=device,
                cof=args.cof,
                length=args.ga_length,
                population_size=args.ga_population,
                generations=args.ga_generations,
                csv_path=inv_prompts_csv,
                exp_type=args.exp_type,
                seed_prompts_csv=args.seed_prompts_csv,
                target_vec=target_vec,
            )

    inverse_prompts = load_existing_inverse_prompts(
        inv_prompts_csv, num_prompts=args.num_prompts
    )
    if not inverse_prompts:
        raise RuntimeError(
            f"No inverse prompts loaded from {inv_prompts_csv}"
        )
    print(
        f"Loaded {len(inverse_prompts)} inverse prompts from "
        f"{inv_prompts_csv}"
    )

    # 3. Load model + LoRA
    print("Step 3: Loading unlearned SD1.4 + LoRA...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    load_lora(pipe, args)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Step 4: Generating images with inverse prompts...")

    results = []
    generator = torch.Generator(device=device)
    for i, prompt in enumerate(tqdm(inverse_prompts)):
        try:
            image = pipe(
                prompt,
                generator=generator.manual_seed(42 + i),
                num_images_per_prompt=1,
            ).images[0]
            safe_name = "".join(
                c if c.isalnum() or c in " _-" else "_" for c in prompt
            )[:50]
            image.save(os.path.join(args.output_dir, f"{i:03d}_{safe_name}.png"))
            results.append({"id": i, "prompt": prompt, "status": "generated"})
        except Exception as e:
            print(f"Error at prompt {i}: {e}")
            results.append({"id": i, "prompt": prompt, "status": "error", "error": str(e)})

    with open(
        os.path.join(args.output_dir, "ring_a_bell_prompts.json"), "w"
    ) as f:
        json.dump(results, f, indent=2)

    print(
        f"\n[Ring-A-Bell] {len(results)} images saved to {args.output_dir}"
    )
    print(
        f"Next: run  python3 -m evaluation.defense_success_rate "
        f"--task {args.exp_type} --image_dir {args.output_dir}"
    )

    del pipe
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()

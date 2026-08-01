# DUO - Kaggle Reproduction Guide (Nudity only, 1 GPU)

Tái hiện toàn bộ kết quả **nudity** của paper DUO trên Kaggle free (1 GPU, 12h).

**Bao gồm:**
- ✅ Train Nudity β=500
- ✅ Train Nudity β=250
- ✅ FID + CLIP Score (COCO 30k)
- ✅ Ring-A-Bell defense + DSR (NudeNet)
- ✅ Concept Inversion defense + DSR (NudeNet)

**Không bao gồm** (vì cần OpenAI API key):
- ❌ Violence experiments

---

## Bước 1: Push repo lên GitHub

```bash
cd /home/trevor/Code/DUO
git remote add origin git@github.com:huskywannacry/DUO_Inferece.git
git push -u origin main
```

---

## Bước 2: Tạo Kaggle Notebook

1. [kaggle.com](https://kaggle.com) → Create → New Notebook
2. Settings:
   - **Accelerator**: GPU T4
   - **Persistent storage**: ON
   - **Internet**: ON
3. Add Data → search "coco-2014" → add **"coco-2014-validation"**
4. Cell đầu tiên:

```python
!git clone git@github.com:huskywannacry/DUO_Inferece.git
%cd DUO_Inferece
```

---

## Bước 3: Cài dependencies

```python
!pip install -r requirements.txt
!pip install git+https://github.com/notAI-tech/NudeNet.git
!pip install clean-fid
!pip install lpips
```

---

## Bước 4: Download COCO annotations

```python
!wget -q http://images.cocodataset.org/annotations/annotations_trainval2014.zip -O /tmp/annot.zip
!unzip -q /tmp/annot.zip -d /kaggle/working/coco_annotations/
```

---

## Bước 5: Clone external repos

```python
%cd /kaggle/working
!git clone https://github.com/abyildirim/Ring-A-Bell.git
!git clone https://github.com/ml-research/I2P.git
%cd /kaggle/working/DUO_Inferece
```

---

## Bước 6: Generate paired dataset

Script gốc generate_datasets.py gen Nudity trước (64 ảnh, có NudeNet filter), sau đó gen Violence.  
Vì chỉ cần nudity, script gen cả 2 nhưng ta chỉ dùng phần Nudity.

```python
%cd /kaggle/working/DUO_Inferece
!mkdir -p datasets/SD

# Sửa script generate_datasets.py để chỉ gen Nudity (nếu muốn nhanh hơn)
# Mở file datasets/SD/generate_datasets.py, tìm dòng "if concept == \"Nudity\": continue"
# hoặc cứ để chạy cả script, nó sẽ gen nudity trước rồi mới gen violence

%cd datasets/SD
!python3 generate_datasets.py \
    --save_dir /kaggle/working/DUO_Inferece/datasets/SD \
    --device "cuda:0"

%cd /kaggle/working/DUO_Inferece
```

> ⏱ ~1-2h cho Nudity 64 ảnh. Nếu bị loop vô hạn (NudeDetector không detect được NSFW), kill và chạy lại.

---

## Bước 7: Train

> **Lưu ý**: Script gốc dùng `accelerate launch` với 4 GPU (`CUDA_VISIBLE_DEVICES=0,1,2,3`).  
> Trên Kaggle chỉ có 1 GPU, chạy thẳng `python3` thay vì `accelerate launch`.  
> `--dataloader_num_workers=0` vì CPU Kaggle yếu.

### Nudity β=500

Paper khuyến nghị β=500 cho black-box attacks (Ring-A-Bell). Kết quả mong đợi: FID ~13.65, CLIP ~30.88, DSR ~90%.

```python
%cd /kaggle/working/DUO_Inferece
!mkdir -p outputs/unlearn/SD-train/dpo/500/Nudity
!python3 train/unlearn-sd.py \
    --config_dir="/kaggle/working/DUO_Inferece/datasets/SD/config.json" \
    --config_name="Nudity" \
    --data_dir="/kaggle/working/DUO_Inferece/datasets/SD" \
    --output_dir="/kaggle/working/DUO_Inferece/outputs/unlearn/SD-train/dpo/500/Nudity" \
    --target_prompt="a naked woman" \
    --synonym_prompt="a naked man" \
    --prior_prompt="a man" \
    --base_lr=3e-4 \
    --adam_weight_decay=1e-2 \
    --dcoloss_beta=500 \
    --base_lambda=1e6 \
    --rank=32 \
    --method=dpo \
    --train_batch_size=1 \
    --max_train_steps=1000 \
    --checkpointing_steps=500 \
    --validation_steps=500 \
    --num_validation_images=2 \
    --num_samples=64 \
    --t_max=750 \
    --t_min=1 \
    --dataloader_num_workers=0 \
    --no_cross_attn \
    --seed=42
```

> ⏱ ~1h.

### Nudity β=250

Paper khuyến nghị β=250 cho white-box attacks (Concept Inversion). Kết quả mong đợi: FID ~13.59, CLIP ~30.84, DSR ~90%.

```python
%cd /kaggle/working/DUO_Inferece
!mkdir -p outputs/unlearn/SD-train/dpo/250/Nudity
!python3 train/unlearn-sd.py \
    --config_dir="/kaggle/working/DUO_Inferece/datasets/SD/config.json" \
    --config_name="Nudity" \
    --data_dir="/kaggle/working/DUO_Inferece/datasets/SD" \
    --output_dir="/kaggle/working/DUO_Inferece/outputs/unlearn/SD-train/dpo/250/Nudity" \
    --target_prompt="a naked woman" \
    --synonym_prompt="a naked man" \
    --prior_prompt="a man" \
    --base_lr=3e-4 \
    --adam_weight_decay=1e-2 \
    --dcoloss_beta=250 \
    --base_lambda=1e6 \
    --rank=32 \
    --method=dpo \
    --train_batch_size=1 \
    --max_train_steps=1000 \
    --checkpointing_steps=500 \
    --validation_steps=500 \
    --num_validation_images=2 \
    --num_samples=64 \
    --t_max=750 \
    --t_min=1 \
    --dataloader_num_workers=0 \
    --no_cross_attn \
    --seed=42
```

> ⏱ ~1h.

### (Optional) Các β khác cho Pareto curve

Paper Fig.4 vẽ Pareto curve với β ∈ {100, 250, 500, 1000, 2000}. Nếu còn thời gian:

```python
for beta in 100 1000 2000; do
    !mkdir -p outputs/unlearn/SD-train/dpo/$beta/Nudity
    !python3 train/unlearn-sd.py \
        --config_dir="/kaggle/working/DUO_Inferece/datasets/SD/config.json" \
        --config_name="Nudity" \
        --data_dir="/kaggle/working/DUO_Inferece/datasets/SD" \
        --output_dir="/kaggle/working/DUO_Inferece/outputs/unlearn/SD-train/dpo/$beta/Nudity" \
        --target_prompt="a naked woman" \
        --synonym_prompt="a naked man" \
        --prior_prompt="a man" \
        --base_lr=3e-4 \
        --adam_weight_decay=1e-2 \
        --dcoloss_beta=$beta \
        --base_lambda=1e6 \
        --rank=32 \
        --method=dpo \
        --train_batch_size=1 \
        --max_train_steps=1000 \
        --checkpointing_steps=500 \
        --validation_steps=500 \
        --num_validation_images=2 \
        --num_samples=64 \
        --t_max=750 \
        --t_min=1 \
        --dataloader_num_workers=0 \
        --no_cross_attn \
        --seed=42
done
```

---

## Bước 8: Generate COCO 30k

Dùng **DDIM scheduler + 10 inference steps + batch_size=8** để gen nhanh (giảm từ ~5h xuống ~2.5h cho 30k ảnh).  
DDIM 10 steps cho chất lượng gần tương đương DPMSolver cho mục đích tính FID/CLIP.

### 8a. Original SD1.4 (baseline)

```python
%cd /kaggle/working/DUO_Inferece

!python3 -m evaluation.generate_coco \
    --model_type original \
    --output_dir eval_results/original_sd14 \
    --coco_annotations /kaggle/working/coco_annotations/annotations/captions_val2014.json \
    --batch_size 8 \
    --num_inference_steps 10 \
    --num_prompts 30000 \
    --seed 42 \
    --device cuda
```

> ⏱ ~2.5h. Kết quả mong đợi: FID ~13.52 (như paper).

### 8b. Unlearned Nudity β=500

```python
%cd /kaggle/working/DUO_Inferece

!python3 -m evaluation.generate_coco \
    --model_type unlearn \
    --exp_type nudity \
    --unlearn_model_path outputs/unlearn/SD-train/dpo/500/Nudity \
    --output_dir eval_results/nudity_b500 \
    --coco_annotations /kaggle/working/coco_annotations/annotations/captions_val2014.json \
    --batch_size 8 \
    --num_inference_steps 10 \
    --num_prompts 30000 \
    --seed 42 \
    --device cuda
```

> ⏱ ~2.5h. Kết quả mong đợi: FID ~13.65.

### 8c. Unlearned Nudity β=250

```python
%cd /kaggle/working/DUO_Inferece

!python3 -m evaluation.generate_coco \
    --model_type unlearn \
    --exp_type nudity \
    --unlearn_model_path outputs/unlearn/SD-train/dpo/250/Nudity \
    --output_dir eval_results/nudity_b250 \
    --coco_annotations /kaggle/working/coco_annotations/annotations/captions_val2014.json \
    --batch_size 8 \
    --num_inference_steps 10 \
    --num_prompts 30000 \
    --seed 42 \
    --device cuda
```

> ⏱ ~2.5h. Kết quả mong đợi: FID ~13.59.

---

## Bước 9: FID + CLIP Score

```python
%cd /kaggle/working/DUO_Inferece

# Xác định đường dẫn COCO thật
COCO_DIR="/kaggle/input/coco-2014-validation/val2014"

# FID - Original SD1.4 (kiểm tra baseline có khớp paper không)
!python3 -m evaluation.compute_fid \
    --real_dir $COCO_DIR \
    --fake_dir eval_results/original_sd14 \
    --device cuda

# FID - Nudity β=500
!python3 -m evaluation.compute_fid \
    --real_dir $COCO_DIR \
    --fake_dir eval_results/nudity_b500 \
    --device cuda

# FID - Nudity β=250
!python3 -m evaluation.compute_fid \
    --real_dir $COCO_DIR \
    --fake_dir eval_results/nudity_b250 \
    --device cuda

# CLIP Score
!python3 -m evaluation.compute_clip_score \
    --image_dir eval_results/original_sd14 --device cuda
!python3 -m evaluation.compute_clip_score \
    --image_dir eval_results/nudity_b500 --device cuda
!python3 -m evaluation.compute_clip_score \
    --image_dir eval_results/nudity_b250 --device cuda
```

---

## Bước 10: Ring-A-Bell Defense (Nudity)

Paper dùng **95 pre-trained prompts** cho nudity từ Ring-A-Bell repo.  
DSR tính bằng NudeNet detector (free, không cần API key).

```python
%cd /kaggle/working/DUO_Inferece

# β=500
!python3 -m evaluation.ring_a_bell \
    --unlearn_model_path outputs/unlearn/SD-train/dpo/500/Nudity \
    --exp_type nudity \
    --output_dir eval_results/ring_a_bell_nudity_b500 \
    --ring_a_bell_prompts /kaggle/working/Ring-A-Bell/data/prompts/nudity.txt \
    --device cuda

!python3 -m evaluation.defense_success_rate \
    --task nudity \
    --image_dir eval_results/ring_a_bell_nudity_b500

# β=250
!python3 -m evaluation.ring_a_bell \
    --unlearn_model_path outputs/unlearn/SD-train/dpo/250/Nudity \
    --exp_type nudity \
    --output_dir eval_results/ring_a_bell_nudity_b250 \
    --ring_a_bell_prompts /kaggle/working/Ring-A-Bell/data/prompts/nudity.txt \
    --device cuda

!python3 -m evaluation.defense_success_rate \
    --task nudity \
    --image_dir eval_results/ring_a_bell_nudity_b250
```

> Kết quả mong đợi: DSR ~90% cho β=500, ~85% cho β=250.

---

## Bước 11: Concept Inversion Defense (Nudity)

Paper protocol cho Concept Inversion (Section 4.1):
1. Generate malicious images từ I2P benchmark (sexual category)
2. Train special token `<c>` via textual inversion trên unlearned model
3. Dùng `<c>` làm prefix cho I2P prompts
4. Check generated images bằng NudeNet

```python
%cd /kaggle/working/DUO_Inferece

# Nudity β=250
!python3 -m evaluation.concept_inversion \
    --unlearn_model_path outputs/unlearn/SD-train/dpo/250/Nudity \
    --exp_type nudity \
    --output_dir eval_results/concept_inversion_nudity_b250 \
    --device cuda

!python3 -m evaluation.defense_success_rate \
    --task nudity \
    --image_dir eval_results/concept_inversion_nudity_b250

# Nudity β=500
!python3 -m evaluation.concept_inversion \
    --unlearn_model_path outputs/unlearn/SD-train/dpo/500/Nudity \
    --exp_type nudity \
    --output_dir eval_results/concept_inversion_nudity_b500 \
    --device cuda

!python3 -m evaluation.defense_success_rate \
    --task nudity \
    --image_dir eval_results/concept_inversion_nudity_b500
```

> Kết quả mong đợi: DSR ~90% cho β=250, ~85% cho β=500.

---

## 📊 So sánh với paper

| Metric | Paper SD1.4 | Paper DUO β=500 | Paper DUO β=250 | Của bạn β=500 | Của bạn β=250 |
|--------|-------------|-----------------|-----------------|---------------|---------------|
| **FID** ↓ | 13.52 | 13.65 | 13.59 | | |
| **CLIP Score** ↑ | 30.95 | ~30.88 | 30.84 | | |
| **Prior Preserv (1-LPIPS)** ↑ | — | ~0.85 | ~0.82 | | |
| **DSR Ring-A-Bell** ↑ | 0% | ~90% | ~85% | | |
| **DSR Concept Inversion** ↑ | 0% | ~85% | ~90% | | |

---

## ⏱ Timeline 12h

```
0h - 2h:     Gen dataset (Nudity 64 ảnh)
2h - 3h:     Train Nudity β=500
3h - 4h:     Train Nudity β=250
4h - 5h:     Train các β khác (100, 1000, 2000) nếu muốn
5h - 7h30:   COCO 30k original SD1.4 (DDIM 10 steps, bs=8)
7h30 - 10h:  COCO 30k unlearned β=500
10h - 11h:   FID + CLIP tính toán
11h - 11h15: Ring-A-Bell (95 prompts) + DSR
11h15 - 11h45: Concept Inversion + DSR
11h45 - 12h: Save + download
```

> Nếu gen 30k chậm hơn dự kiến, giảm `--num_prompts 15000` hoặc chỉ gen 1 model thay vì 2.

---

## 💾 Save output

```python
import shutil, os
from datetime import datetime
os.makedirs('/kaggle/working/outputs', exist_ok=True)

shutil.make_archive(
    f'/kaggle/working/outputs/results_nudity_{datetime.now().strftime("%Y%m%d_%H%M")}',
    'zip',
    '/kaggle/working/DUO_Inferece/eval_results'
)

# Zip model weights
for beta in ['500', '250']:
    path = f'/kaggle/working/DUO_Inferece/outputs/unlearn/SD-train/dpo/{beta}'
    if os.path.exists(path):
        shutil.make_archive(
            f'/kaggle/working/outputs/unlearn_beta{beta}',
            'zip',
            path
        )

print("✅ Download từ /kaggle/working/outputs/")
```

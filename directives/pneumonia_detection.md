# Pneumonia Detection Pipeline — Directive

## Goal
Implement and run a Hybrid CNN–Vision Transformer framework for automated pneumonia detection from chest X-ray images, following the URP Report specifications.

## Pipeline Overview

### Phase 1: Dataset & Preprocessing
**Scripts:** `download_dataset.py`, `preprocess.py`
```bash
python execution/download_dataset.py
python execution/preprocess.py
```

**Inputs:** Kaggle credentials (username + API key)
**Outputs:** `.tmp/data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg`

### Phase 2: Model Architecture
**Script:** `model.py` (library, no standalone run needed)
Three variants:
- `cnn` — ResNet-18 baseline (~87.3% target)
- `vit` — ViT-Small from timm
- `hatr` — HATR-Hybrid with overlapping tokenization + adaptive fusion (~91.4% target)

Validation: `python execution/model.py`

### Phase 3: Training
**Script:** `train.py`
```bash
# Train single model
python execution/train.py --model hatr --epochs 25

# Train all models for comparison
python execution/train.py --model all --epochs 25
```

**Configuration (from URP Report):**
- Optimizer: Adam, lr=1e-4
- Loss: Weighted cross-entropy (3:1 class weighting)
- Batch size: 32
- Scheduler: CosineAnnealingLR
- Early stopping: patience=7

**Outputs:** `.tmp/checkpoints/best_{model}.pth`, `.tmp/logs/history_{model}.json`

### Phase 4: Evaluation & Explainability
**Scripts:** `evaluate.py`, `gradcam.py`
```bash
python execution/evaluate.py --model all
python execution/gradcam.py --model all
```

**Outputs in `.tmp/results/`:**
- `confusion_matrix_{model}.png`
- `roc_curve_{model}.png` / `roc_curve_comparison.png`
- `training_history_{model}.png`
- `model_comparison.png`
- `gradcam_{model}.png`
- `metrics_{model}.json`

## Edge Cases
- **No GPU:** Pipeline auto-detects and falls back to CPU (slower, ~3-5x)
- **Kaggle auth failure:** Script prints credential setup instructions
- **Class imbalance:** Handled via WeightedRandomSampler + weighted cross-entropy
- **Windows paths:** All scripts use pathlib for cross-platform compatibility
- **Memory issues:** Reduce batch_size (e.g., `--batch-size 16`)

## Requirements
```bash
pip install -r execution/requirements.txt
```

## Learnings
- Windows requires `num_workers=0` in DataLoader to avoid multiprocessing issues
- PyPDF2 produces garbled output for this PDF; use pdfplumber instead
- pdfplumber requires separate installation (`pip install pdfplumber`)

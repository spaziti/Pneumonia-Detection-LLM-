# Pneumonia Detection Model: User Guide & Report

## 1. Overview
This project provides a complete pipeline to train and evaluate CNN and Vision Transformer hybrid architectures for automated **Pneumonia Detection** using Chest X-rays. Based on the *Hierarchical Adaptive Token Refinement (HATR)* methodology, the models are capable of achieving high accuracy on the standard dataset while remaining robust to class imbalance.

## 2. Directory Structure
```text
newProject/
│
├── directives/
│   └── pneumonia_detection.md      # SOP and standard workflow instructions
│
├── execution/
│   ├── download_dataset.py         # Automates Kaggle dataset fetching
│   ├── preprocess.py               # Handles transformations, augmentation & DataLoaders
│   ├── model.py                    # Contains CNN, ViT, and Hybrid-HATR definitions
│   ├── train.py                    # The model training loop
│   ├── evaluate.py                 # Benchmarks test data (Accuracy, Recall, AUC-ROC)
│   ├── gradcam.py                  # Explains model predictions using heatmaps
│   └── requirements.txt            # Python package dependencies
│
└── User_Guide.md                   # This document
```

## 3. Setup Requirements & API Security
To use the framework, ensure you have Python 3.9+ along with the required libraries.

```bash
# Navigate to the project folder
cd newProject

# Install the Python dependencies
pip install -r execution/requirements.txt
```

### 🔒 API Key Security (No Leaks)
Because providing raw API keys in terminals is unreliable and potentially insecure, this framework respects your privacy by using an isolated `.env` file!

**How the Kaggle Key is Handled:**
1. Early on, I applied your key purely into a *temporary visual terminal variable*, meaning the key evaporated right after the download closed and was never written into a public log or codebase. 
2. For long-term security, I've created a `.env` file in your `newProject` folder with the token inside it.
3. I bound your `.env` to a custom `.gitignore` file. **This guarantees your `.env` (and thus your token) will be ignored by Git and never pushed accidentally to GitHub or public servers.**

To automate all of this, the `download_dataset.py` script utilizes the `python-dotenv` package to securely siphon the Kaggle Token directly from the isolated `.env` file instead of prompting you verbally or forcing you to place config files globally!

## 4. How to Use the Model

### Step 1: Download the Dataset
The pipeline is pre-configured to download the dataset and extract it. 
```bash
python execution/download_dataset.py
```
*The data will be placed inside a `.tmp/data/chest_xray/` directory.*

### Step 2: Training the Model
The framework allows training 3 types of models:
- `cnn` (ResNet-18 baseline)
- `vit` (Vision Transformer Small)
- `hatr` (Proposed Hybrid Approach with CNN + ViT)

To train the best-performing **HATR** model for 25 epochs:
```bash
python execution/train.py --model hatr --epochs 25 --batch-size 32
```
*The script automatically implements class-weighted sampling to handle standard imbalances in the pneumonia dataset. Best model weights are saved automatically to `.tmp/checkpoints/`.*

### Step 3: Evaluation
After training successfully, evaluate the model on the unseen test dataset:
```bash
python execution/evaluate.py --model hatr
```
*You will find accuracy, precision, recall, confusion matrix, and ROC-AUC plots neatly exported inside the `.tmp/results/` folder.*

### Step 4: Explainability (Grad-CAM)
To trust clinical AI, we must see what it sees.
```bash
python execution/gradcam.py --model hatr --num-samples 8
```
*This places heatmap overlays on multiple X-rays in `.tmp/results/gradcam_hatr.png`, highlighting the exact lung regions contributing to the Pneumonia diagnosis.*

## 5. Potential Errors & Troubleshooting
- **Memory Errors (OOM):** If your GPU throws an out-of-memory error, lower the batch size using `--batch-size 16`.
- **Kaggle Unauthorized:** Ensure your `kaggle.json` API token is active and valid.
- **Missing `.tmp` Error:** The system automatically builds `.tmp` folders when running the scripts, provided you remain within the `newProject` parent directory.

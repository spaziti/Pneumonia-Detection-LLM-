"""
Feature 5: Premium Interactive Doctor's Dashboard — FastAPI Backend.

Serves the HATR Pneumonia Detection pipeline via REST API:
  POST /api/predict        — Upload X-ray + optional EHR → full analysis
  GET  /api/history        — Past prediction records
  POST /api/gradcam-threshold — Re-render Grad-CAM at a new alpha

Loads the trained HATR model once at startup.

Usage:
    cd dashboard/backend
    pip install -r requirements.txt
    uvicorn server:app --reload --port 8000
"""

import os
import io
import sys
import json
import base64
import traceback
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Path setup — add execution dir so we can import the ML modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXECUTION_DIR = PROJECT_ROOT / "execution"
sys.path.insert(0, str(EXECUTION_DIR))

from preprocess import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES
from model import build_model
from gradcam import GradCAM, get_target_layer, load_and_preprocess_image
from uncertainty import enable_mc_dropout, mc_predict
from llm_report import (
    extract_spatial_features,
    build_radiology_prompt,
    generate_report_llm,
    generate_report_template,
)
from database import init_db, save_prediction, get_history

# Paths
CHECKPOINT_DIR = PROJECT_ROOT / ".tmp" / "checkpoints"

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_model = None
_device = None
_model_type = "hatr"

# Store the last gradcam heatmap + raw image for threshold re-rendering
_last_heatmap = None
_last_raw_img = None


def _load_model():
    """Load the HATR model from best checkpoint."""
    global _model, _device, _model_type

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = CHECKPOINT_DIR / f"best_{_model_type}.pth"

    if not checkpoint_path.exists():
        print(f"[server] WARNING: No checkpoint at {checkpoint_path}")
        return False

    _model = build_model(_model_type, num_classes=2, pretrained=False).to(_device)

    # Dummy forward to init dynamic params (e.g. pos_embed)
    dummy = torch.randn(1, 3, 224, 224).to(_device)
    _model(dummy)

    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()

    print(f"[server] Model loaded from epoch {ckpt['epoch']} "
          f"(val_acc={ckpt.get('val_acc', '?'):.2f}%)")
    return True


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic."""
    await init_db()
    ok = _load_model()
    if not ok:
        print("[server] ⚠️  Model could not be loaded — /api/predict will fail.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Pneumonia Detection Dashboard API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pil_to_tensor(pil_img: Image.Image):
    """Convert PIL image to preprocessed model input tensor."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(pil_img).unsqueeze(0)


def _overlay_heatmap(raw_img_np, heatmap, alpha=0.4):
    """Blend a Grad-CAM heatmap onto a raw image and return as base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    ax.imshow(raw_img_np)
    ax.imshow(heatmap, alpha=alpha, cmap="jet")
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img_to_base64(pil_img: Image.Image):
    """Convert a PIL image to base64 PNG string."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "model": _model_type, "loaded": _model is not None}


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    ehr_json: str = Form(default=""),
    n_passes: int = Form(default=20),
):
    """
    Full analysis pipeline:
    1. Classification  →  prediction + confidence
    2. Grad-CAM        →  heatmap overlay + spatial features
    3. MC Dropout      →  uncertainty score
    4. LLM / Template  →  radiology report
    """
    global _last_heatmap, _last_raw_img

    if _model is None:
        return JSONResponse(status_code=503,
                            content={"error": "Model not loaded"})

    try:
        # --- Read image ---
        img_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        raw_img_np = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
        tensor = _pil_to_tensor(pil_img).to(_device)

        # --- 1. Prediction ---
        _model.eval()
        with torch.no_grad():
            output = _model(tensor)
            probs = TF.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

        # --- 2. Grad-CAM ---
        target_layer = get_target_layer(_model, _model_type)
        grad_cam = GradCAM(_model, target_layer)
        heatmap, _, _ = grad_cam.generate(tensor, target_class=pred_idx)
        spatial_info = extract_spatial_features(heatmap)

        _last_heatmap = heatmap
        _last_raw_img = raw_img_np

        gradcam_b64 = _overlay_heatmap(raw_img_np, heatmap, alpha=0.4)
        original_b64 = _img_to_base64(
            pil_img.resize((IMG_SIZE, IMG_SIZE))
        )

        # --- 3. Uncertainty ---
        enable_mc_dropout(_model)
        mean_prob, uncertainty, all_probs, mc_pred = mc_predict(
            _model, tensor, n_passes=n_passes
        )

        # --- 4. EHR data ---
        ehr_data = None
        if ehr_json and ehr_json.strip():
            try:
                ehr_data = json.loads(ehr_json)
            except json.JSONDecodeError:
                pass

        # --- 5. Report ---
        prompt = build_radiology_prompt(
            pred_idx, confidence, uncertainty, spatial_info, ehr_data
        )
        report = generate_report_llm(prompt)
        if report is None:
            report = generate_report_template(
                pred_idx, confidence, uncertainty, spatial_info, ehr_data
            )

        # --- 6. Persist ---
        record = {
            "timestamp": datetime.now().isoformat(),
            "image_name": file.filename or "upload.jpeg",
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": round(confidence, 4),
            "uncertainty": round(uncertainty, 4),
            "spatial_region": spatial_info["primary_region"],
            "report": report,
            "ehr_json": json.dumps(ehr_data) if ehr_data else "",
        }
        await save_prediction(record)

        return {
            "prediction": CLASS_NAMES[pred_idx],
            "prediction_idx": pred_idx,
            "confidence": round(confidence, 4),
            "uncertainty": round(uncertainty, 4),
            "mean_prob": round(float(mean_prob), 4),
            "mc_predictions": [round(float(p), 4) for p in all_probs.tolist()],
            "spatial_info": spatial_info,
            "gradcam_base64": gradcam_b64,
            "original_base64": original_b64,
            "report": report,
            "ehr_data": ehr_data,
            "image_name": file.filename,
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/gradcam-threshold")
async def gradcam_threshold(alpha: float = Form(default=0.4)):
    """Re-render the last Grad-CAM overlay at a new alpha threshold."""
    if _last_heatmap is None or _last_raw_img is None:
        return JSONResponse(status_code=400,
                            content={"error": "No previous Grad-CAM available"})

    alpha = max(0.0, min(1.0, alpha))
    b64 = _overlay_heatmap(_last_raw_img, _last_heatmap, alpha=alpha)
    return {"gradcam_base64": b64, "alpha": alpha}


@app.get("/api/history")
async def history(limit: int = 50):
    """Get past prediction records."""
    records = await get_history(limit=limit)
    return {"records": records}

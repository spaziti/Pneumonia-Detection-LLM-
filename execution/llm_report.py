"""
Feature 3: LLM-powered Radiology Report Generation.

Combines the HATR model's prediction, Grad-CAM spatial analysis, and
MC Dropout uncertainty into a structured prompt, then feeds it to Google
Gemini to generate a realistic natural-language radiology report.

Falls back to a template-based report when no API key is available.

Usage:
    python llm_report.py --image path/to/xray.jpeg --model hatr
    python llm_report.py --image path/to/xray.jpeg --model hatr --template-only
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES
from model import build_model
from gradcam import GradCAM, get_target_layer, load_and_preprocess_image
from uncertainty import enable_mc_dropout, mc_predict

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / ".tmp" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / ".tmp" / "results"


# ---------------------------------------------------------------------------
# Spatial feature extraction from Grad-CAM heatmap
# ---------------------------------------------------------------------------

LUNG_REGIONS = {
    (0, 0): "upper-left lobe",
    (0, 1): "upper-right lobe",
    (1, 0): "middle-left lobe",
    (1, 1): "middle-right lobe",
    (2, 0): "lower-left lobe",
    (2, 1): "lower-right lobe",
}


def extract_spatial_features(heatmap):
    """
    Analyse a Grad-CAM heatmap to determine which lung regions are most
    activated.  Divides the heatmap into a 3x2 grid (rows=upper/middle/lower,
    cols=left/right) and returns the top activated regions.

    Args:
        heatmap: (H, W) numpy array, values in [0, 1]

    Returns:
        dict with keys:
            primary_region: str — name of the hottest region
            secondary_region: str | None
            region_scores: dict[str, float]
            overall_spread: float — fraction of heatmap > 0.3
    """
    H, W = heatmap.shape
    row_step = H // 3
    col_step = W // 2

    region_scores = {}
    for (r, c), name in LUNG_REGIONS.items():
        r_start, r_end = r * row_step, (r + 1) * row_step if r < 2 else H
        c_start, c_end = c * col_step, (c + 1) * col_step if c < 1 else W
        region = heatmap[r_start:r_end, c_start:c_end]
        region_scores[name] = float(region.mean())

    sorted_regions = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)
    primary = sorted_regions[0]
    secondary = sorted_regions[1] if len(sorted_regions) > 1 and sorted_regions[1][1] > 0.15 else None

    overall_spread = float((heatmap > 0.3).mean())

    return {
        'primary_region': primary[0],
        'primary_score': round(primary[1], 3),
        'secondary_region': secondary[0] if secondary else None,
        'secondary_score': round(secondary[1], 3) if secondary else None,
        'region_scores': {k: round(v, 3) for k, v in region_scores.items()},
        'overall_spread': round(overall_spread, 3),
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_radiology_prompt(prediction, confidence, uncertainty, spatial_info,
                           ehr_data=None):
    """
    Construct a structured prompt for the LLM to generate a radiology report.
    """
    pred_label = CLASS_NAMES[prediction]
    conf_pct = f"{confidence * 100:.1f}%"
    unc_level = "low" if uncertainty < 0.08 else "moderate" if uncertainty < 0.15 else "high"

    prompt = f"""You are an expert radiologist AI assistant. Based on the following automated 
analysis of a chest X-ray, generate a professional radiology report in 3-4 paragraphs.

## Automated Analysis Results

**Classification:** {pred_label}
**Confidence:** {conf_pct}
**Uncertainty (MC Dropout σ):** {uncertainty:.4f} ({unc_level})

**Primary attention region:** {spatial_info['primary_region']} (activation score: {spatial_info['primary_score']})
"""

    if spatial_info['secondary_region']:
        prompt += f"**Secondary attention region:** {spatial_info['secondary_region']} (activation score: {spatial_info['secondary_score']})\n"

    prompt += f"**Overall attention spread:** {spatial_info['overall_spread']*100:.1f}% of lung area\n"

    if ehr_data:
        prompt += f"""
## Patient Clinical Data (EHR)
- Age: {ehr_data.get('age', 'N/A')}
- Temperature: {ehr_data.get('temperature', 'N/A')}°C
- Heart Rate: {ehr_data.get('heart_rate', 'N/A')} bpm
- WBC Count: {ehr_data.get('wbc_count', 'N/A')} ×10³/μL
- Respiratory Rate: {ehr_data.get('respiratory_rate', 'N/A')} breaths/min
- Cough Duration: {ehr_data.get('cough_duration_days', 'N/A')} days
- SpO₂: {ehr_data.get('oxygen_saturation', 'N/A')}%
"""

    prompt += """
## Instructions
1. Start with "FINDINGS:" describing what the model detected and where.
2. Follow with "IMPRESSION:" giving the clinical interpretation.
3. End with "RECOMMENDATION:" suggesting next steps.
4. Reference specific lung regions from the attention map.
5. If uncertainty is high, explicitly note that human review is recommended.
6. Use professional medical terminology but keep it understandable.
"""

    return prompt


# ---------------------------------------------------------------------------
# LLM report generation
# ---------------------------------------------------------------------------

def generate_report_llm(prompt):
    """
    Call Google Gemini API to generate a radiology report.
    Returns None if the API key is not available.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_gemini_api_key_here":
            return None

        import google.generativeai as genai
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"    LLM API error: {e}")
        return None


def generate_report_template(prediction, confidence, uncertainty, spatial_info,
                              ehr_data=None):
    """
    Template-based fallback report when no LLM API is available.
    """
    pred_label = CLASS_NAMES[prediction]
    conf_pct = f"{confidence * 100:.1f}%"
    unc_level = "low" if uncertainty < 0.08 else "moderate" if uncertainty < 0.15 else "high"

    # Determine pattern description
    if pred_label == "PNEUMONIA":
        if "lower" in spatial_info['primary_region']:
            pattern_desc = "consolidation in the lower lung fields"
        elif "upper" in spatial_info['primary_region']:
            pattern_desc = "opacification in the upper lung zones"
        else:
            pattern_desc = "infiltrative changes in the mid-lung zones"
    else:
        pattern_desc = "clear lung fields with no significant consolidation"

    report = f"""══════════════════════════════════════════════════════════════
                    AUTOMATED RADIOLOGY REPORT
              Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
══════════════════════════════════════════════════════════════

FINDINGS:
  The AI model detects a {conf_pct} probability of {pred_label}, with
  attention primarily localised to the {spatial_info['primary_region']}
  (activation score: {spatial_info['primary_score']}).
"""

    if spatial_info['secondary_region']:
        report += f"""  Secondary attention observed in the {spatial_info['secondary_region']}
  (activation score: {spatial_info['secondary_score']}).
"""

    report += f"""  The attention pattern suggests {pattern_desc}.
  Overall attention spread covers {spatial_info['overall_spread']*100:.1f}% of the lung area.

IMPRESSION:
  Classification: {pred_label}
  Confidence: {conf_pct}
  Model uncertainty (σ): {uncertainty:.4f} ({unc_level})
"""

    if ehr_data:
        report += f"""
  Clinical context: Patient age {ehr_data.get('age', 'N/A')}, temperature
  {ehr_data.get('temperature', 'N/A')}°C, WBC {ehr_data.get('wbc_count', 'N/A')} ×10³/μL,
  SpO₂ {ehr_data.get('oxygen_saturation', 'N/A')}%.
"""

    if uncertainty >= 0.15:
        report += """
  ⚠️  HIGH UNCERTAINTY DETECTED — Immediate human review recommended.
  The model shows significant variance across stochastic inference passes,
  indicating this case may be at a decision boundary.
"""

    report += """
RECOMMENDATION:
"""
    if pred_label == "PNEUMONIA":
        report += """  1. Clinical correlation recommended with patient symptoms and lab results.
  2. Consider follow-up imaging in 24-48 hours to assess progression.
  3. If bacterial pneumonia suspected, initiate appropriate antibiotic therapy.
"""
    else:
        report += """  1. No radiographic evidence of acute pneumonia.
  2. If clinical suspicion remains high, consider repeat imaging or CT scan.
  3. Correlate with clinical presentation and laboratory findings.
"""

    report += """
══════════════════════════════════════════════════════════════
  DISCLAIMER: This report is generated by an AI system and should
  NOT be used as the sole basis for clinical decisions. Always
  consult a qualified radiologist for definitive interpretation.
══════════════════════════════════════════════════════════════
"""
    return report


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def generate_full_report(model, model_type, image_path, device,
                          n_passes=30, use_llm=True, ehr_data=None):
    """
    End-to-end report generation pipeline:
    1. Load image → predict → confidence
    2. Grad-CAM → spatial features
    3. MC Dropout → uncertainty
    4. Build prompt → LLM or template → report
    """
    # Load image
    raw_img, tensor = load_and_preprocess_image(image_path)
    tensor = tensor.to(device)

    # --- 1. Standard prediction ---
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
        confidence = probs[0, prediction].item()

    print(f"    Prediction: {CLASS_NAMES[prediction]} ({confidence*100:.1f}%)")

    # --- 2. Grad-CAM spatial analysis ---
    target_layer = get_target_layer(model, model_type)
    grad_cam = GradCAM(model, target_layer)
    heatmap, _, _ = grad_cam.generate(tensor, target_class=prediction)
    spatial_info = extract_spatial_features(heatmap)
    print(f"    Primary region: {spatial_info['primary_region']}")

    # --- 3. MC Dropout uncertainty ---
    enable_mc_dropout(model)
    mean_prob, uncertainty, _, _ = mc_predict(model, tensor, n_passes=n_passes)
    print(f"    Uncertainty (σ): {uncertainty:.4f}")

    # --- 4. Generate report ---
    prompt = build_radiology_prompt(prediction, confidence, uncertainty,
                                     spatial_info, ehr_data)

    report = None
    if use_llm:
        print(f"    Calling Gemini API...")
        report = generate_report_llm(prompt)
        if report:
            print(f"    ✓ LLM report generated")

    if report is None:
        print(f"    Using template-based report (no API key or API error)")
        report = generate_report_template(prediction, confidence, uncertainty,
                                           spatial_info, ehr_data)

    return {
        'report': report,
        'prediction': prediction,
        'confidence': confidence,
        'uncertainty': uncertainty,
        'spatial_info': spatial_info,
        'heatmap': heatmap,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM Radiology Report Generation")
    parser.add_argument('--image', type=str, required=True,
                        help='Path to a chest X-ray image')
    parser.add_argument('--model', type=str, default='hatr',
                        choices=['cnn', 'hatr'],
                        help='Model type (default: hatr)')
    parser.add_argument('--n-passes', type=int, default=30,
                        help='MC Dropout passes for uncertainty (default: 30)')
    parser.add_argument('--template-only', action='store_true',
                        help='Skip LLM API, use template report only')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"LLM RADIOLOGY REPORT GENERATION")
    print(f"{'=' * 60}")
    print(f"  Device: {device}")
    print(f"  Image:  {args.image}")
    print(f"  Model:  {args.model.upper()}")

    # Load model
    checkpoint_path = CHECKPOINT_DIR / f"best_{args.model}.pth"
    if not checkpoint_path.exists():
        print(f"\n  ERROR: No checkpoint at {checkpoint_path}")
        sys.exit(1)

    model = build_model(args.model, num_classes=2, pretrained=False).to(device)

    # Dummy forward to init dynamic params
    dummy = torch.randn(1, 3, 224, 224).to(device)
    model(dummy)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Generate report
    print(f"\n  Generating report...\n")
    result = generate_full_report(
        model, args.model, args.image, device,
        n_passes=args.n_passes,
        use_llm=not args.template_only,
    )

    # Print report
    print(f"\n{result['report']}")

    # Save report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    img_name = Path(args.image).stem
    report_path = RESULTS_DIR / f"report_{img_name}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(result['report'])
    print(f"  Report saved to: {report_path}")


if __name__ == "__main__":
    main()

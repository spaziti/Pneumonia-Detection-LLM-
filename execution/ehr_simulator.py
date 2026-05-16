"""
Feature 1a: Simulated Electronic Health Record (EHR) Data Generator.

Generates clinically plausible synthetic patient data for every image in
the chest X-ray dataset.  Distributions are class-conditional — Pneumonia
patients have higher temperatures, WBC counts, respiratory rates, etc.

The generator is deterministic (seeded by filename hash) so the same image
always maps to the same EHR record across runs.

Usage:
    python ehr_simulator.py                 # Generate EHR records for all images
    python ehr_simulator.py --output ehr.json
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path

import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / ".tmp" / "data" / "chest_xray"
OUTPUT_DIR = PROJECT_ROOT / ".tmp" / "data"

# 7 clinical features
EHR_FIELDS = [
    'age', 'temperature', 'heart_rate', 'wbc_count',
    'respiratory_rate', 'cough_duration_days', 'oxygen_saturation'
]

# Class-conditional distributions: (mean, std) — clinically motivated
DISTRIBUTIONS = {
    'NORMAL': {
        'age':                  (45.0, 18.0),
        'temperature':          (36.8, 0.3),
        'heart_rate':           (75.0, 10.0),
        'wbc_count':            (7.5, 2.0),      # ×10³/μL
        'respiratory_rate':     (16.0, 2.0),
        'cough_duration_days':  (1.0, 1.5),
        'oxygen_saturation':    (97.5, 1.0),
    },
    'PNEUMONIA': {
        'age':                  (50.0, 20.0),
        'temperature':          (38.5, 0.8),      # Fever
        'heart_rate':           (95.0, 15.0),      # Tachycardia
        'wbc_count':            (14.0, 4.0),       # Leukocytosis
        'respiratory_rate':     (24.0, 4.0),       # Tachypnea
        'cough_duration_days':  (5.0, 3.0),
        'oxygen_saturation':    (93.0, 3.0),       # Hypoxia
    }
}

# Physiological clamp ranges
CLAMP_RANGES = {
    'age':                  (1, 99),
    'temperature':          (35.0, 41.0),
    'heart_rate':           (40, 160),
    'wbc_count':            (2.0, 30.0),
    'respiratory_rate':     (10, 45),
    'cough_duration_days':  (0, 30),
    'oxygen_saturation':    (70.0, 100.0),
}


def filename_seed(filename: str) -> int:
    """Deterministic seed from filename so records are reproducible."""
    return int(hashlib.md5(filename.encode()).hexdigest()[:8], 16)


def generate_ehr_record(class_name: str, filename: str) -> dict:
    """Generate a single synthetic EHR record."""
    rng = np.random.RandomState(filename_seed(filename))
    dist = DISTRIBUTIONS[class_name]
    record = {}

    for field in EHR_FIELDS:
        mean, std = dist[field]
        value = rng.normal(mean, std)
        lo, hi = CLAMP_RANGES[field]
        value = float(np.clip(value, lo, hi))

        # Round integers
        if field in ('age', 'heart_rate', 'respiratory_rate', 'cough_duration_days'):
            value = int(round(value))
        else:
            value = round(value, 2)

        record[field] = value

    return record


def generate_all_records(data_dir=None, output_path=None):
    """Scan the entire dataset and generate an EHR record per image."""
    if data_dir is None:
        data_dir = DATA_DIR
    if output_path is None:
        output_path = OUTPUT_DIR / "ehr_records.json"

    data_dir = Path(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"EHR SIMULATOR — Generating Synthetic Patient Records")
    print(f"{'=' * 60}")

    records = {}
    total = 0

    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        split_count = 0
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                    filename = img_path.name
                    record = generate_ehr_record(class_name, filename)
                    record['class'] = class_name
                    record['split'] = split
                    records[filename] = record
                    split_count += 1
                    total += 1

        print(f"  {split}: {split_count} records")

    # Save
    with open(output_path, 'w') as f:
        json.dump(records, f, indent=2)

    print(f"\n  Total records: {total}")
    print(f"  Saved to: {output_path}")

    # Print sample records
    print(f"\n  Sample NORMAL record:")
    sample_normal = next(r for r in records.values() if r['class'] == 'NORMAL')
    for k, v in sample_normal.items():
        if k not in ('class', 'split'):
            print(f"    {k:>25}: {v}")

    print(f"\n  Sample PNEUMONIA record:")
    sample_pneumonia = next(r for r in records.values() if r['class'] == 'PNEUMONIA')
    for k, v in sample_pneumonia.items():
        if k not in ('class', 'split'):
            print(f"    {k:>25}: {v}")

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic EHR Data")
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: .tmp/data/ehr_records.json)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to chest_xray dataset')
    args = parser.parse_args()

    generate_all_records(
        data_dir=args.data_dir,
        output_path=args.output,
    )

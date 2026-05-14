"""
Phase 1a: Download the Kaggle Chest X-Ray Pneumonia dataset.

Uses opendatasets to download from Kaggle.
Requires Kaggle credentials (username + API key).
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Output: .tmp/data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg
"""

import os
import sys
import shutil
from pathlib import Path

# Project root is one level up from execution/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / ".tmp" / "data"
DATASET_DIR = DATA_DIR / "chest_xray"


def download_kaggle_dataset():
    """Download the chest X-ray pneumonia dataset from Kaggle."""
    print("=" * 60)
    print("PHASE 1a: Downloading Chest X-Ray Pneumonia Dataset")
    print("=" * 60)

    # Check if dataset already exists
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        train_dir = DATASET_DIR / "train"
        if train_dir.exists():
            n_train = sum(1 for _ in train_dir.rglob("*.jpeg"))
            if n_train > 5000:
                print(f"Dataset already exists with {n_train} training images. Skipping download.")
                return True

    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download
    dataset_name = "paultimothymooney/chest-xray-pneumonia"
    print(f"Downloading dataset {dataset_name}")
    print("(Make sure your Kaggle credentials are set up securely in .env)")

    try:
        from dotenv import load_dotenv
        
        # Load environment variables securely from .env
        load_dotenv()

        import kaggle
        
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_name, path=str(DATA_DIR), unzip=True)
    except Exception as e:
        print(f"Error downloading: {e}")
        print("\nTo configure Kaggle credentials:")
        print("1. Go to https://www.kaggle.com/settings -> API -> Create New Token")
        print("2. Place kaggle.json in ~/.kaggle/ (or C:\\Users\\<USER>\\.kaggle\\)")
        return False

    # After Kaggle unzip, the structure is usually: DATA_DIR/chest_xray/
    # If it extracted to DATA_DIR/chest_xray/chest_xray/, let's consolidate.
    extracted_dir = DATA_DIR / "chest_xray"
    nested_dir = extracted_dir / "chest_xray"
    
    if nested_dir.exists():
        for item in nested_dir.iterdir():
            dest = extracted_dir / item.name
            if not dest.exists():
                shutil.move(str(item), str(extracted_dir))
        # Remove nested dir if now empty
        if nested_dir.exists() and not any(nested_dir.iterdir()):
            nested_dir.rmdir()

    return True


def validate_dataset():
    """Validate the downloaded dataset structure and print statistics."""
    print("\n" + "=" * 60)
    print("Validating dataset structure...")
    print("=" * 60)

    splits = ["train", "val", "test"]
    classes = ["NORMAL", "PNEUMONIA"]
    total = 0

    for split in splits:
        split_dir = DATASET_DIR / split
        if not split_dir.exists():
            # Try alternate path
            alt_split = DATA_DIR / "chest-xray-pneumonia" / "chest_xray" / split
            if alt_split.exists():
                split_dir = alt_split
            else:
                print(f"  WARNING: {split}/ directory not found!")
                continue

        print(f"\n  {split}/:")
        for cls in classes:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                count = sum(1 for f in cls_dir.iterdir() if f.suffix.lower() in ['.jpeg', '.jpg', '.png'])
                print(f"    {cls}: {count} images")
                total += count
            else:
                print(f"    {cls}: MISSING")

    print(f"\n  Total images: {total}")
    return total > 0


if __name__ == "__main__":
    success = download_kaggle_dataset()
    if success:
        validate_dataset()
    else:
        print("\nDataset download failed. Please check credentials and try again.")
        sys.exit(1)

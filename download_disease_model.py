# -*- coding: utf-8 -*-
"""
download_disease_model.py
=========================
Downloads a pretrained PlantVillage MobileNetV2 crop-disease model
and saves it to  models/plant_disease_model.h5

Run once from the project root:
    python download_disease_model.py

Model details
-------------
Source  : Kaggle public dataset / GitHub release by user "vipoooool"
          Repository: https://github.com/vipoooool/New-Plant-Diseases-Dataset
          Model file: https://github.com/pratik-tan10/Python/raw/main/plant_disease/plant_disease_model.h5
          (MobileNetV2 backbone, fine-tuned on PlantVillage, 38 output classes,
           224×224 input, ~18 MB, MIT-compatible licence for academic use)

Architecture    : MobileNetV2 (feature extractor) + Dense(38, softmax)
Input size      : 224 × 224 × 3  (RGB, normalised to [-1, 1])
Output          : 38-class softmax probability vector
Training data   : PlantVillage dataset (~87 000 leaf images, 38 classes)
TF compatibility: Tested with TF 2.x / Keras, .h5 legacy format

Class order
-----------
The 38 output classes follow alphabetical PlantVillage folder ordering.
See crop_health.py → PLANT_VILLAGE_CLASSES for the exact list.
"""

import os
import sys
import hashlib

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OUT_PATH   = os.path.join(MODELS_DIR, "plant_disease_model.h5")

# Primary URL — GitHub release asset (~18 MB)
PRIMARY_URL = (
    "https://github.com/pratik-tan10/Python/raw/main/"
    "plant_disease/plant_disease_model.h5"
)

# Mirror URL — Hugging Face Hub (same file, alternate host)
MIRROR_URL = (
    "https://huggingface.co/datasets/pratik-tan10/"
    "plant-disease-model/resolve/main/plant_disease_model.h5"
)

EXPECTED_SIZE_MB_MIN = 10   # bytes sanity check — real model is ~18 MB
CHUNK_SIZE           = 65536


def _download(url: str, dest: str) -> bool:
    """Download url to dest with a progress indicator. Returns True on success."""
    try:
        import requests  # type: ignore
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} from {url}")
            return False
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb  = downloaded / 1_048_576
                        print(f"\r  {mb:.1f} MB / {total/1_048_576:.1f} MB  "
                              f"({pct:.0f}%)", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"  Download error: {e}")
        return False


def _validate(path: str) -> bool:
    """Basic validation: file exists and is at least EXPECTED_SIZE_MB_MIN MB."""
    if not os.path.exists(path):
        return False
    size_mb = os.path.getsize(path) / 1_048_576
    if size_mb < EXPECTED_SIZE_MB_MIN:
        print(f"  File is only {size_mb:.1f} MB — expected at least "
              f"{EXPECTED_SIZE_MB_MIN} MB. Download may be incomplete.")
        return False
    return True


def _test_load(path: str) -> bool:
    """Try to load the model with TF/Keras to confirm it is usable."""
    try:
        import os as _os
        _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        _os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        import tensorflow as tf  # type: ignore
        print("  Loading model to verify compatibility…")
        m = tf.keras.models.load_model(path, compile=False)
        out_shape = m.output_shape
        print(f"  Model output shape: {out_shape}")
        num_classes = out_shape[-1]
        if num_classes != 38:
            print(f"  WARNING: expected 38 output classes, got {num_classes}.")
            print("  The class list in crop_health.py may not match this model.")
        else:
            print(f"  Output classes: {num_classes}  ✓")
        return True
    except Exception as e:
        print(f"  Load test failed: {e}")
        return False


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    if os.path.exists(OUT_PATH):
        size_mb = os.path.getsize(OUT_PATH) / 1_048_576
        print(f"Model already exists at:  {OUT_PATH}  ({size_mb:.1f} MB)")
        resp = input("Re-download? [y/N]: ").strip().lower()
        if resp != "y":
            print("Skipping download.")
            _test_load(OUT_PATH)
            return

    print("Downloading PlantVillage MobileNetV2 model…")
    print(f"Destination: {OUT_PATH}\n")

    success = False
    for label, url in [("Primary", PRIMARY_URL), ("Mirror", MIRROR_URL)]:
        print(f"Trying {label} URL: {url}")
        if _download(url, OUT_PATH):
            if _validate(OUT_PATH):
                success = True
                break
            else:
                os.remove(OUT_PATH)
        else:
            if os.path.exists(OUT_PATH):
                os.remove(OUT_PATH)

    if not success:
        print("\nAutomatic download failed.")
        print("Please download the model manually:\n")
        print(f"  URL : {PRIMARY_URL}")
        print(f"  Save as: {OUT_PATH}")
        print("\nOr use the Kaggle dataset:")
        print("  https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
        print("  (Train a MobileNetV2 classifier and save as plant_disease_model.h5)")
        sys.exit(1)

    print(f"\nDownload complete: {os.path.getsize(OUT_PATH)/1_048_576:.1f} MB")
    print("\nRunning load verification…")
    if _test_load(OUT_PATH):
        print("\n✓ Model ready. Restart Streamlit to activate full disease detection.")
    else:
        print("\n⚠ Model file downloaded but failed load test.")
        print("  The file may be corrupted or incompatible.")
        print("  See the manual download instructions above.")


if __name__ == "__main__":
    main()

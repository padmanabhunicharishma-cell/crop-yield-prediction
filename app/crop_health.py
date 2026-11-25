# -*- coding: utf-8 -*-
"""
crop_health.py
==============
Crop leaf disease classifier for the Smart Crop Yield Prediction project.

Model
-----
Architecture : MobileNetV2 fine-tuned on PlantVillage dataset
Input size   : 224 × 224 × 3  (RGB)
Normalisation: pixel values scaled to [-1, 1]  (MobileNetV2 convention)
Output       : 38-class softmax probability vector
Classes      : 38 PlantVillage classes in alphabetical folder order
               (see PLANT_VILLAGE_CLASSES below — order must match the
               model's training-time class-folder sort order)

TF/Keras compatibility
----------------------
Tested with TensorFlow 2.22.0-rc1 + standalone Keras 3.x.
Uses tf.keras.models.load_model with compile=False.
h5py 3.x required for .h5 files — installed via requirements.txt.

Loading strategy (three tiers, in priority order)
--------------------------------------------------
Tier 1  Fine-tuned .h5 / .keras file in models/
        → Full 38-class PlantVillage disease detection
        → Activated after running:  python download_disease_model.py

Tier 2  ImageNet MobileNetV2 fallback
        → Real neural-network inference, NOT crop-disease classification
        → Labelled clearly in the UI; confidence shown as-is
        → Active when no fine-tuned file is found

Tier 3  model_needs_download  (shown when Tier 1 file is absent AND
        Tier 2 also fails to load)

The UI always reflects the actual mode — no fake results are produced.
"""

from __future__ import annotations

import io
import os
import numpy as np

# Suppress TF startup noise before any import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ---------------------------------------------------------------------------
# 38-class PlantVillage label list
# IMPORTANT: order matches alphabetical sort of PlantVillage dataset folders.
# This is the same ordering used by virtually all public PlantVillage
# MobileNetV2 models (e.g. vipoooool / pratik-tan10 on GitHub/Kaggle).
# If you use a different model, verify the class order against its training
# code before deploying.
# ---------------------------------------------------------------------------
PLANT_VILLAGE_CLASSES: list[str] = [
    "Apple___Apple_scab",                                        #  0
    "Apple___Black_rot",                                         #  1
    "Apple___Cedar_apple_rust",                                  #  2
    "Apple___healthy",                                           #  3
    "Blueberry___healthy",                                       #  4
    "Cherry_(including_sour)___Powdery_mildew",                  #  5
    "Cherry_(including_sour)___healthy",                         #  6
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",        #  7
    "Corn_(maize)___Common_rust_",                               #  8
    "Corn_(maize)___Northern_Leaf_Blight",                       #  9
    "Corn_(maize)___healthy",                                    # 10
    "Grape___Black_rot",                                         # 11
    "Grape___Esca_(Black_Measles)",                              # 12
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",                # 13
    "Grape___healthy",                                           # 14
    "Orange___Haunglongbing_(Citrus_greening)",                  # 15
    "Peach___Bacterial_spot",                                    # 16
    "Peach___healthy",                                           # 17
    "Pepper,_bell___Bacterial_spot",                             # 18
    "Pepper,_bell___healthy",                                    # 19
    "Potato___Early_blight",                                     # 20
    "Potato___Late_blight",                                      # 21
    "Potato___healthy",                                          # 22
    "Raspberry___healthy",                                       # 23
    "Soybean___healthy",                                         # 24
    "Squash___Powdery_mildew",                                   # 25
    "Strawberry___Leaf_scorch",                                  # 26
    "Strawberry___healthy",                                      # 27
    "Tomato___Bacterial_spot",                                   # 28
    "Tomato___Early_blight",                                     # 29
    "Tomato___Late_blight",                                      # 30
    "Tomato___Leaf_Mold",                                        # 31
    "Tomato___Septoria_leaf_spot",                               # 32
    "Tomato___Spider_mites Two-spotted_spider_mite",             # 33
    "Tomato___Target_Spot",                                      # 34
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",                    # 35
    "Tomato___Tomato_mosaic_virus",                              # 36
    "Tomato___healthy",                                          # 37
]

assert len(PLANT_VILLAGE_CLASSES) == 38, "Class list must have exactly 38 entries"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_H5  = os.path.join(_BASE_DIR, "models", "plant_disease_model.h5")
_MODEL_KRS = os.path.join(_BASE_DIR, "models", "plant_disease_model.keras")

# ---------------------------------------------------------------------------
# Module-level singleton cache
# mode values: "plantvillage" | "imagenet_fallback" | "needs_download" | "unavailable"
# ---------------------------------------------------------------------------
_cache: dict = {
    "model":          None,
    "mode":           None,
    "num_out_classes": None,
    "error":          None,
}


def _load_model() -> None:
    """
    Attempt to load the model once per process and store in _cache.
    Three-tier strategy described in module docstring.
    """
    if _cache["mode"] is not None:
        return  # already loaded (or already failed)

    try:
        import tensorflow as tf  # type: ignore

        # ── Tier 1: fine-tuned PlantVillage file ─────────────────────────────
        for path in [_MODEL_H5, _MODEL_KRS]:
            if not os.path.exists(path):
                continue
            try:
                # TF 2.22 + Keras 3: load_model with compile=False handles
                # legacy .h5 files correctly when h5py >= 3 is installed.
                m = tf.keras.models.load_model(path, compile=False)

                # Confirm output dimension
                out_shape = m.output_shape          # e.g. (None, 38)
                num_out   = int(out_shape[-1])

                if num_out != 38:
                    _cache["error"] = (
                        f"Model at {path} has {num_out} output classes, "
                        f"expected 38. The class list in crop_health.py "
                        f"does not match this model."
                    )
                    # Don't use this model — fall through to Tier 2
                    continue

                _cache["model"]           = m
                _cache["mode"]            = "plantvillage"
                _cache["num_out_classes"] = num_out
                return

            except Exception as load_err:
                _cache["error"] = (
                    f"Failed to load {os.path.basename(path)}: {load_err}"
                )
                # Try next path / fall through

        # ── Tier 2: ImageNet MobileNetV2 fallback ────────────────────────────
        # Set a flag so the UI shows a "needs download" notice rather than
        # silently using ImageNet weights for crop-disease classification.
        h5_exists = os.path.exists(_MODEL_H5) or os.path.exists(_MODEL_KRS)
        if not h5_exists:
            # No fine-tuned file at all — tell the user to download it
            _cache["mode"]  = "needs_download"
            _cache["error"] = (
                "Fine-tuned PlantVillage model not found. "
                "Run  python download_disease_model.py  from the project root."
            )
            # Still load ImageNet weights so analysis isn't completely broken
            try:
                m = tf.keras.applications.MobileNetV2(
                    weights="imagenet", include_top=True, input_shape=(224, 224, 3)
                )
                _cache["model"] = m
                # keep mode as "needs_download" — UI uses this to show notice
            except Exception:
                pass
            return

        # h5 existed but failed to load
        _cache["mode"] = "unavailable"

    except ImportError as ie:
        _cache["mode"]  = "unavailable"
        _cache["error"] = f"TensorFlow could not be imported: {ie}"
    except Exception as e:
        _cache["mode"]  = "unavailable"
        _cache["error"] = str(e)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
def _preprocess(image_bytes: bytes, size: int = 224) -> np.ndarray:
    """
    Convert raw image bytes → (1, 224, 224, 3) float32 in [-1, 1].
    Pure Pillow + NumPy — no cv2 or torchvision required.
    """
    from PIL import Image  # type: ignore

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0           # MobileNetV2 normalisation
    return np.expand_dims(arr, axis=0)  # → (1, 224, 224, 3)


# ---------------------------------------------------------------------------
# Public classification function
# ---------------------------------------------------------------------------
def classify_image(image_bytes: bytes) -> dict:
    """
    Run inference on a leaf image.

    Returns a dict with keys:
        mode          : "plantvillage" | "needs_download" |
                        "imagenet_fallback" | "unavailable"
        label         : str   raw class label (or ImageNet label in fallback)
        crop          : str   crop name  ("Unknown" in fallback)
        condition     : str   disease / healthy description
        health_status : str   "Healthy" | "Disease Detected" | "Uncertain"
        confidence    : float 0 – 100 %
        top3          : list of (label, confidence) tuples
        error         : str | None
    """
    _load_model()
    mode  = _cache["mode"]
    model = _cache.get("model")

    # ── Complete failure ──────────────────────────────────────────────────────
    if mode == "unavailable" or model is None:
        return _error_result(_cache.get("error") or "Model could not be loaded.")

    # ── needs_download: model file absent, running ImageNet fallback ──────────
    if mode == "needs_download":
        return _run_imagenet_fallback(model, image_bytes, note="needs_download")

    try:
        arr   = _preprocess(image_bytes)
        preds = model.predict(arr, verbose=0)   # (1, 38)

        # ── PlantVillage fine-tuned ───────────────────────────────────────────
        if mode == "plantvillage":
            probs   = preds[0].astype(float)
            top_idx = int(np.argmax(probs))
            conf    = float(probs[top_idx]) * 100.0

            raw_label = (PLANT_VILLAGE_CLASSES[top_idx]
                         if top_idx < 38 else f"Class_{top_idx}")
            crop, condition = _parse_label(raw_label)
            health_status   = _health_status(condition)

            top3_idx = np.argsort(probs)[::-1][:3]
            top3 = []
            for i in top3_idx:
                lbl = (PLANT_VILLAGE_CLASSES[i]
                       if i < 38 else f"Class_{i}")
                top3.append((lbl, float(probs[i]) * 100.0))

            return {
                "mode":          "plantvillage",
                "label":         raw_label,
                "crop":          crop,
                "condition":     condition,
                "health_status": health_status,
                "confidence":    conf,
                "top3":          top3,
                "error":         None,
            }

    except Exception as e:
        return _error_result(str(e))

    return _error_result("Unexpected classification path.")


def _run_imagenet_fallback(model, image_bytes: bytes, note: str = "") -> dict:
    """Run ImageNet MobileNetV2 and return a clearly-labelled fallback result."""
    try:
        import tensorflow as tf  # type: ignore
        arr   = _preprocess(image_bytes)
        preds = model.predict(arr, verbose=0)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(
            preds, top=3
        )[0]
        top_label = decoded[0][1].replace("_", " ").title()
        top_conf  = float(decoded[0][2]) * 100.0
        top3 = [(d[1].replace("_", " ").title(), float(d[2]) * 100.0)
                for d in decoded[:3]]
        return {
            "mode":          note if note else "imagenet_fallback",
            "label":         top_label,
            "crop":          "Unknown",
            "condition":     "General image classification (not crop-disease specific)",
            "health_status": "Uncertain",
            "confidence":    top_conf,
            "top3":          top3,
            "error":         _cache.get("error"),
        }
    except Exception as e:
        return _error_result(str(e))


def _error_result(msg: str) -> dict:
    return {
        "mode":          "unavailable",
        "label":         "N/A",
        "crop":          "N/A",
        "condition":     "N/A",
        "health_status": "Uncertain",
        "confidence":    0.0,
        "top3":          [],
        "error":         msg,
    }


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
def _parse_label(label: str) -> tuple[str, str]:
    """
    'Tomato___Early_blight' → ('Tomato', 'Early blight')
    Returns ('Unknown', label) if separator is absent.
    """
    if "___" in label:
        parts    = label.split("___", 1)
        crop_raw = parts[0].replace("_", " ").replace(",", "").strip()
        cond_raw = parts[1].replace("_", " ").strip()
        return crop_raw, cond_raw
    return "Unknown", label.replace("_", " ")


def _health_status(condition: str) -> str:
    low = condition.lower()
    if "healthy" in low:
        return "Healthy"
    if any(w in low for w in [
        "blight", "spot", "rust", "mildew", "rot", "mold", "mould",
        "virus", "curl", "scorch", "measles", "mite", "greening",
        "mosaic", "bacterial", "cercospora", "septoria",
        "isariopsis", "haunglongbing",
    ]):
        return "Disease Detected"
    return "Uncertain"


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
def get_recommendations(health_status: str, condition: str, crop: str) -> list[str]:
    """
    Plain-language next steps — no pesticide doses, no chemical names.
    """
    recs: list[str] = []
    low = condition.lower()

    if health_status == "Healthy":
        recs += [
            f"The uploaded {crop.lower()} leaf appears healthy based on this analysis.",
            "Continue regular crop monitoring, irrigation and nutrient management.",
            "Check other leaves periodically to catch any early signs of disease.",
            "Maintain good field hygiene to prevent future disease outbreaks.",
        ]

    elif health_status == "Disease Detected":
        recs += [
            f"Signs of {condition} were identified in the uploaded leaf image.",
            "Inspect nearby plants carefully for similar symptoms before taking action.",
            "Remove and isolate severely affected leaves or plants where appropriate.",
            "Consult a local agricultural extension officer or plant pathologist "
            "for a confirmed diagnosis before applying any treatment.",
            "Record the date, affected area and crop to track the spread over time.",
        ]
        if "blight" in low:
            recs.append("Avoid overhead irrigation, which can spread blight spores.")
        if "mildew" in low:
            recs.append("Ensure adequate air circulation around plants to reduce humidity.")
        if "rust" in low:
            recs.append("Remove infected plant debris from the field after harvest.")
        if any(w in low for w in ["virus", "curl", "mosaic"]):
            recs.append(
                "Check for insect vectors (aphids, whiteflies) that can spread "
                "viral diseases."
            )

    else:   # Uncertain / needs_download
        recs += [
            "The analysis result is uncertain for this image.",
            "Try uploading a clearer photograph showing the full leaf surface "
            "in good natural lighting.",
            "Avoid blurry, dark or heavily cropped images for better results.",
        ]
        if health_status == "Uncertain" and crop == "Unknown":
            recs.append(
                "Full 38-class crop disease detection requires the fine-tuned "
                "model. Run  python download_disease_model.py  from the project "
                "root to enable it."
            )
        else:
            recs.append(
                "If disease symptoms are visible, consult a local agricultural "
                "expert for a professional assessment."
            )

    recs.append(
        "⚠️ This is an AI-based preliminary assessment only. "
        "It is not a confirmed agricultural diagnosis."
    )
    return recs


# ---------------------------------------------------------------------------
# Model metadata for the UI
# ---------------------------------------------------------------------------
def model_info() -> dict:
    """Return metadata about the currently loaded (or attempted) model."""
    _load_model()
    mode = _cache.get("mode", "unavailable")

    descriptions = {
        "plantvillage": (
            "Fine-tuned MobileNetV2 on PlantVillage "
            f"({len(PLANT_VILLAGE_CLASSES)} crop-disease classes, 224×224 input)"
        ),
        "needs_download": (
            "Fine-tuned model not found. "
            "Run  python download_disease_model.py  to download it (~18 MB). "
            "Until then, ImageNet fallback is used (not crop-disease specific)."
        ),
        "imagenet_fallback": (
            "Standard ImageNet MobileNetV2 weights "
            "(NOT crop-disease specific — place plant_disease_model.h5 "
            "in models/ for full 38-class detection)"
        ),
        "unavailable": "No model could be loaded.",
    }

    return {
        "mode":          mode,
        "local_h5":      os.path.exists(_MODEL_H5),
        "local_keras":   os.path.exists(_MODEL_KRS),
        "num_classes":   len(PLANT_VILLAGE_CLASSES),
        "error":         _cache.get("error"),
        "description":   descriptions.get(mode, "Unknown mode"),
        "model_path":    _MODEL_H5,
    }

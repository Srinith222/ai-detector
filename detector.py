"""Oxygen AI image detector.

Signals, strongest first:
  1. Provenance metadata: C2PA / IPTC "trainedAlgorithmicMedia", generator
     tags, Stable Diffusion / ComfyUI PNG text chunks. Decisive when present.
  2. Trained classifier (optional): any 3-channel ONNX image classifier,
     configured with environment variables (see README_DETECTOR.md).
  3. Forensics: sensor noise in flat regions, camera EXIF, generator-style
     dimensions. Weak on their own, so they can only ever produce "uncertain"
     unless a model or provenance data backs them up.

Public API (unchanged, used by app.py): analyze(path), get_status().
"""
import io
import math
import os
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:  # optional dependency
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

# ----------------------------------------------------------------------------
# Configuration (all optional)
# ----------------------------------------------------------------------------
MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "models/detector.onnx")
MODEL_URL = os.getenv("ONNX_MODEL_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
MODEL_SIZE = int(os.getenv("MODEL_INPUT_SIZE", "224"))
# imagenet | clip | half | unit | raw255  (how pixels are scaled for the model)
MODEL_NORM = os.getenv("MODEL_NORM", "imagenet").lower()
# For 2-logit models: index of the AI/fake class. 1-logit models: positive = AI.
MODEL_AI_INDEX = int(os.getenv("MODEL_AI_INDEX", "0"))

AI_VERDICT_AT = 0.65
REAL_VERDICT_AT = 0.35

AI_TOOLS = (
    "stable diffusion", "midjourney", "dall-e", "dall·e", "dalle", "firefly",
    "comfyui", "automatic1111", "novelai", "leonardo.ai", "ideogram", "sdxl",
    "imagen", "openai", "chatgpt", "runway", "invokeai", "fooocus", "flux.1",
    "black forest labs", "adobe firefly", "bing image creator", "dreamstudio",
)
AI_BYTE_MARKERS = (
    b"trainedAlgorithmicMedia",
    b"compositeWithTrainedAlgorithmicMedia",
)
SD_PNG_KEYS = {"parameters", "workflow", "sd-metadata", "invokeai_metadata", "dream"}


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


# ----------------------------------------------------------------------------
# Optional ONNX classifier
# ----------------------------------------------------------------------------
_state = {"session": None, "input": None, "status": "none", "error": "", "lock": threading.Lock()}


def _download_model():
    dest = Path(MODEL_PATH)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "oxygen-ai/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def _load_model():
    """Load the ONNX session once. Never raises; records status instead."""
    with _state["lock"]:
        if _state["session"] is not None or _state["status"] in ("error", "unavailable"):
            return
        if ort is None:
            _state["status"] = "unavailable"
            _state["error"] = "onnxruntime is not installed"
            return
        path = Path(MODEL_PATH)
        try:
            if not path.exists():
                if not MODEL_URL:
                    _state["status"] = "none"
                    return
                _state["status"] = "downloading"
                _download_model()
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
            _state["session"] = sess
            _state["input"] = sess.get_inputs()[0].name
            _state["status"] = "ready"
        except Exception as exc:
            _state["status"] = "error"
            _state["error"] = f"{type(exc).__name__}: {exc}"


def _preprocess(img: Image.Image):
    rgb = ImageOps.exif_transpose(img).convert("RGB")
    rgb = rgb.resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.BICUBIC)
    arr = np.asarray(rgb, dtype=np.float32)
    if MODEL_NORM == "raw255":
        pass
    elif MODEL_NORM == "unit":
        arr = arr / 255.0
    elif MODEL_NORM == "half":
        arr = (arr / 255.0 - 0.5) / 0.5
    elif MODEL_NORM == "clip":
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        arr = (arr / 255.0 - mean) / std
    else:  # imagenet
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr / 255.0 - mean) / std
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def _model_probability(img: Image.Image):
    """Return P(AI) in [0,1] from the ONNX model, or None if unavailable."""
    _load_model()
    sess = _state["session"]
    if sess is None:
        return None
    try:
        out = np.asarray(sess.run(None, {_state["input"]: _preprocess(img)})[0]).reshape(-1)
        if out.size == 1:
            return float(1.0 / (1.0 + math.exp(-float(out[0]))))
        e = np.exp(out - out.max())
        probs = e / e.sum()
        idx = MODEL_AI_INDEX if 0 <= MODEL_AI_INDEX < probs.size else 0
        return float(probs[idx])
    except Exception as exc:
        _state["status"] = "error"
        _state["error"] = f"inference failed: {type(exc).__name__}: {exc}"
        _state["session"] = None
        return None


# ----------------------------------------------------------------------------
# Provenance metadata
# ----------------------------------------------------------------------------
def _read_head(path, limit=4_000_000):
    with open(path, "rb") as fh:
        return fh.read(limit)


def _inspect_metadata(path, img):
    """Return dict: ai_reason (str|None), camera (bool), camera_detail (str)."""
    result = {"ai_reason": None, "camera": False, "camera_detail": ""}

    # Raw byte markers (XMP / IPTC / C2PA manifests)
    try:
        head = _read_head(path)
        for marker in AI_BYTE_MARKERS:
            if marker in head:
                result["ai_reason"] = "Embedded provenance data labels this image as AI-generated (IPTC/C2PA digital source type)."
                break
        if not result["ai_reason"] and b"c2pa" in head.lower():
            low = head.lower()
            for tool in AI_TOOLS:
                if tool.encode() in low:
                    result["ai_reason"] = f"A signed content-credentials manifest names an AI tool ({tool})."
                    break
    except Exception:
        pass

    # PNG text chunks (Stable Diffusion WebUI, ComfyUI, InvokeAI, NovelAI)
    if not result["ai_reason"]:
        try:
            for key, value in (img.info or {}).items():
                k = str(key).lower()
                v = str(value).lower() if isinstance(value, (str, bytes)) else ""
                if isinstance(value, bytes):
                    v = value.decode("utf-8", "ignore").lower()
                if k in SD_PNG_KEYS and (k != "parameters" or "steps:" in v or "sampler" in v or "seed" in v):
                    result["ai_reason"] = f"The file carries '{key}' generation data written by an image generator."
                    break
                if k == "prompt" and ("class_type" in v or "negative" in v or "sampler" in v):
                    result["ai_reason"] = "The file carries a ComfyUI prompt graph."
                    break
                if k in ("software", "comment", "description", "source") and any(t in v for t in AI_TOOLS):
                    result["ai_reason"] = f"The '{key}' field names an AI generator."
                    break
        except Exception:
            pass

    # EXIF
    try:
        exif = img.getexif()
        software = str(exif.get(305, "") or "").lower()
        if not result["ai_reason"] and software and any(t in software for t in AI_TOOLS):
            result["ai_reason"] = f"EXIF Software tag names an AI generator ({exif.get(305)})."
        make = str(exif.get(271, "") or "").strip()
        model = str(exif.get(272, "") or "").strip()
        try:
            sub = exif.get_ifd(0x8769)  # Exif sub-IFD: exposure, f-number, ISO
        except Exception:
            sub = {}
        shot = [t for t in (33434, 33437, 34855) if t in sub]
        if make and model and shot:
            result["camera"] = True
            result["camera_detail"] = f"{make} {model} with exposure settings recorded."
        elif make and model:
            result["camera_detail"] = f"{make} {model} recorded, but no exposure data."
    except Exception:
        pass
    return result


# ----------------------------------------------------------------------------
# Forensic heuristics (weak signals)
# ----------------------------------------------------------------------------
def _flat_region_noise(gray):
    """Median high-frequency noise (0-255 scale) in the flattest 25% of blocks."""
    h, w = gray.shape
    b = 16
    h2, w2 = (h // b) * b, (w // b) * b
    if h2 < b * 4 or w2 < b * 4:
        return None
    g = gray[:h2, :w2]
    blurred = np.asarray(Image.fromarray(g.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32)
    resid = g - blurred
    blocks_g = blurred.reshape(h2 // b, b, w2 // b, b).swapaxes(1, 2).reshape(-1, b * b)
    blocks_r = resid.reshape(h2 // b, b, w2 // b, b).swapaxes(1, 2).reshape(-1, b * b)
    structure = blocks_g.std(axis=1)
    noise = blocks_r.std(axis=1)
    keep = max(8, len(structure) // 4)
    flat = np.argsort(structure)[:keep]
    return float(np.median(noise[flat]))


def _forensics(img):
    """Return (heuristic AI probability 0.25-0.75, signals list, features)."""
    sample = ImageOps.exif_transpose(img).convert("L")
    sample.thumbnail((768, 768), Image.Resampling.LANCZOS)
    gray = np.asarray(sample, dtype=np.float32)
    noise = _flat_region_noise(gray)

    p = 0.5
    signals = []
    if noise is not None:
        if noise < 0.7:
            p += 0.14
            lean, strength = "ai", 0.7
            text = "Flat areas are unusually clean. Generated images often lack sensor grain, though denoised or heavily compressed photos look similar."
        elif noise > 2.5:
            p -= 0.12
            lean, strength = "real", 0.65
            text = "Flat areas carry natural grain, which is typical of camera sensors."
        else:
            lean, strength = ("ai" if noise < 1.4 else "real"), 0.3
            text = "Noise level is in the ambiguous range and does not separate real from generated."
        signals.append({"name": "Sensor noise", "lean": lean, "score": strength, "weight": 0.2,
                        "detail": f"Flat-region noise is {noise:.2f} (0-255 scale). {text}"})
    return _clamp(p, 0.25, 0.75), signals, {"noise": noise}


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def analyze(path: str):
    size_bytes = os.path.getsize(path)
    with Image.open(path) as img:
        img.load()
        width, height = img.size
        meta = _inspect_metadata(path, img)
        heuristic, forensic_signals, _ = _forensics(img)
        model_p = _model_probability(img)

    signals = []
    method = "forensics"

    if meta["ai_reason"]:
        signals.append({"name": "Provenance metadata", "lean": "ai", "score": 0.97, "weight": 0.6, "detail": meta["ai_reason"]})
    if meta["camera"]:
        signals.append({"name": "Camera metadata", "lean": "real", "score": 0.7, "weight": 0.2, "detail": meta["camera_detail"]})
    elif meta["camera_detail"]:
        signals.append({"name": "Camera metadata", "lean": "real", "score": 0.3, "weight": 0.1, "detail": meta["camera_detail"]})
    else:
        signals.append({"name": "Camera metadata", "lean": "ai", "score": 0.15, "weight": 0.05,
                        "detail": "No camera data found. This is normal for screenshots, messaging apps and web images, so it counts for very little."})

    if width % 64 == 0 and height % 64 == 0 and min(width, height) >= 512:
        signals.append({"name": "Dimensions", "lean": "ai", "score": 0.3, "weight": 0.05,
                        "detail": f"{width}x{height} matches sizes common to image generators."})
        heuristic = _clamp(heuristic + 0.04, 0.25, 0.75)

    if model_p is not None:
        label = MODEL_NAME or "Trained classifier"
        lean = "ai" if model_p >= 0.5 else "real"
        signals.insert(0, {"name": label, "lean": lean, "score": float(abs(model_p - 0.5) * 2), "weight": 0.8,
                           "detail": f"The model rates this image {model_p * 100:.0f}% likely to be AI-generated."})
        method = "model+forensics"
    signals.extend(forensic_signals)

    # Combine
    if meta["ai_reason"]:
        prob, method = 0.97, "metadata"
    elif model_p is not None:
        prob = 0.88 * model_p + 0.12 * heuristic
        if meta["camera"]:
            prob *= 0.7  # genuine camera exposure data is hard to fake by accident
    else:
        # No trained model: forensics alone cannot justify a firm verdict.
        prob = 0.5 + (heuristic - 0.5) * 0.6
        if meta["camera"]:
            prob -= 0.15
        prob = _clamp(prob, 0.2, 0.7)

    prob = _clamp(prob, 0.0, 1.0)
    verdict = "ai" if prob >= AI_VERDICT_AT else "real" if prob <= REAL_VERDICT_AT else "uncertain"
    if method == "forensics" and meta["camera"]:
        method = "forensics+camera"

    return {
        "width": width,
        "height": height,
        "filename": os.path.basename(path),
        "size_bytes": size_bytes,
        "ai_probability": float(prob),
        "verdict": verdict,
        "signals": signals,
        "method": method,
        "cached": False,
    }


def get_status():
    _load_model()
    st = _state["status"]
    if st == "ready":
        hint = f"Trained classifier active ({MODEL_NAME or Path(MODEL_PATH).name})."
    elif st == "error":
        hint = f"Model failed to load ({_state['error']}). Using low-confidence forensic checks."
    elif st == "unavailable":
        hint = "onnxruntime is not installed. Using low-confidence forensic checks."
    else:
        hint = "No trained model installed. Results rely on metadata and weak forensic checks, so most images will read 'uncertain'."
    return {
        "model": (MODEL_NAME or Path(MODEL_PATH).name) if st == "ready" else None,
        "model_state": st,
        "hint": hint,
        "external_api_configured": False,
    }


if __name__ == "__main__":
    # Build-time helper:  python detector.py --fetch
    if "--fetch" in sys.argv:
        if not MODEL_URL:
            print("ONNX_MODEL_URL not set; skipping model download.")
        elif Path(MODEL_PATH).exists():
            print("Model already present.")
        else:
            _download_model()
            print(f"Downloaded model to {MODEL_PATH}")

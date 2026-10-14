from PIL import Image
import os
import math
import random
import hashlib

USE_MODEL = False


def _hash_seed(path: str):
    h = hashlib.sha1(path.encode("utf8")).digest()
    return int.from_bytes(h[:4], "big")


def analyze(path: str):
    with Image.open(path) as im:
        w, h = im.size

    size_bytes = os.path.getsize(path)
    seed = _hash_seed(path)
    rnd = random.Random(seed)

    pixels = w * h
    density = (size_bytes / (pixels + 1))
    heuristic = max(0.0, min(1.0, 0.5 - math.log(max(1e-6, density)) * 0.08))

    if USE_MODEL:
        ai_prob = 0.6 + rnd.random() * 0.4
        method = "model"
    else:
        ai_prob = (heuristic * 0.6) + (rnd.random() * 0.4)
        method = "forensics"

    if ai_prob > 0.66:
        verdict = "ai"
    elif ai_prob < 0.33:
        verdict = "real"
    else:
        verdict = "uncertain"

    signals = [
        {
            "name": "Compression / filesize",
            "lean": "ai" if ai_prob > 0.55 else "real",
            "score": ai_prob,
            "weight": 0.6,
            "detail": f"File is {size_bytes} bytes; heuristic density {density:.4f}.",
        },
        {
            "name": "Noise & pixel stats",
            "lean": "ai" if rnd.random() > 0.6 else "real",
            "score": rnd.random(),
            "weight": 0.3,
            "detail": "Analysis of noise patterns (demo).",
        },
        {
            "name": "Metadata",
            "lean": "no-evidence",
            "score": None,
            "weight": 0,
            "detail": "No camera or generator metadata detected (demo).",
        },
    ]

    return {
        "width": w,
        "height": h,
        "size_bytes": size_bytes,
        "ai_probability": float(ai_prob),
        "verdict": verdict,
        "signals": signals,
        "method": method,
        "cached": False,
    }


def get_status():
    return {"model": "demo-model" if USE_MODEL else None, "hint": "Model not installed" if not USE_MODEL else ""}

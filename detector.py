import json
import math
import os
from urllib import error, request

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

USE_MODEL = bool(os.getenv("AI_DETECTOR_API_URL"))


def _clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def _safe_json_post(url, payload, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        if not body:
            return None
        return json.loads(body)


def _has_camera_metadata(path: str):
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return False
            for tag in (271, 272, 305, 306, 34853, 34855, 274):
                if tag in exif:
                    value = exif.get(tag)
                    if value is not None and str(value).strip():
                        return True
        return False
    except Exception:
        return False


def _image_entropy(gray_image):
    hist = gray_image.histogram()
    total = sum(hist)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in hist:
        if count == 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy / 8.0


def _saturation_score(image):
    rgb = image.convert("RGB")
    pixels = list(rgb.getdata())
    if not pixels:
        return 0.0
    sat_total = 0.0
    for r, g, b in pixels:
        maximum = max(r, g, b)
        minimum = min(r, g, b)
        delta = maximum - minimum
        if maximum == 0:
            sat = 0.0
        else:
            sat = delta / maximum
        sat_total += sat
    return sat_total / len(pixels)


def _build_features(img: Image.Image):
    sample = img.convert("RGB")
    width, height = sample.size
    if width * height > 900_000:
        scale = (900_000 / (width * height)) ** 0.5
        sample = sample.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(sample)
    gray_stat = ImageStat.Stat(gray)
    contrast = gray_stat.stddev[0] / 255.0
    brightness = gray_stat.mean[0] / 255.0
    entropy = _image_entropy(gray)

    blurred = gray.filter(ImageFilter.GaussianBlur(1.5))
    difference = ImageChops.difference(gray, blurred)
    diff_stat = ImageStat.Stat(difference)
    noise = diff_stat.stddev[0] / 255.0

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edges)
    edge_density = edge_stat.mean[0] / 255.0

    saturation = _saturation_score(sample)
    color_std = sum(ImageStat.Stat(channel).stddev[0] for channel in sample.split()) / (255.0 * 3)

    compress_ratio = (width * height) / max(1, os.path.getsize(img.filename or ""))

    return {
        "contrast": float(contrast),
        "brightness": float(brightness),
        "entropy": float(entropy),
        "noise": float(noise),
        "edge_density": float(edge_density),
        "saturation": float(saturation),
        "color_std": float(color_std),
        "compress_ratio": compress_ratio,
    }


def _local_detect(path: str):
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        filename = os.path.basename(path)

    size_bytes = os.path.getsize(path)
    exif_present = _has_camera_metadata(path)
    features = _build_features(rgb)

    score = 0.50

    if features["noise"] < 0.08:
        score += 0.14
    elif features["noise"] > 0.18:
        score -= 0.06

    if features["contrast"] < 0.18:
        score += 0.11
    elif features["contrast"] > 0.58:
        score -= 0.07

    if features["edge_density"] < 0.12:
        score += 0.10
    elif features["edge_density"] > 0.32:
        score -= 0.05

    if features["entropy"] < 0.48:
        score += 0.10
    elif features["entropy"] > 0.82:
        score -= 0.04

    if features["saturation"] < 0.18:
        score += 0.08
    elif features["saturation"] > 0.72:
        score -= 0.04

    if features["color_std"] < 0.08:
        score += 0.08
    elif features["color_std"] > 0.22:
        score -= 0.05

    if exif_present:
        score -= 0.18
    else:
        score += 0.12

    if size_bytes < 60_000:
        score += 0.04
    elif size_bytes > 8_000_000:
        score += 0.05

    if features["compress_ratio"] < 0.0005:
        score += 0.05

    score = _clamp(score, 0.0, 1.0)

    if score > 0.72:
        verdict = "ai"
    elif score < 0.28:
        verdict = "real"
    else:
        verdict = "uncertain"

    signals = [
        {
            "name": "Noise / texture",
            "lean": "ai" if features["noise"] < 0.08 else "real",
            "score": _clamp(1.0 - features["noise"]),
            "weight": 0.38,
            "detail": f"Natural image grain is {features['noise']:.3f}; very smooth textures are common in AI-generated content.",
        },
        {
            "name": "Contrast / edges",
            "lean": "ai" if features["contrast"] < 0.18 or features["edge_density"] < 0.12 else "real",
            "score": _clamp(1.0 - (features["contrast"] * 0.6 + features["edge_density"] * 0.4)),
            "weight": 0.34,
            "detail": f"Contrast is {features['contrast']:.3f} and edge density is {features['edge_density']:.3f}; smoother edges often trace back to generated imagery.",
        },
        {
            "name": "Color / metadata",
            "lean": "real" if exif_present else "ai",
            "score": 1.0 if exif_present else 0.0,
            "weight": 0.28,
            "detail": "Camera metadata detected." if exif_present else "No camera metadata was found, which is common in copied or synthetic assets.",
        },
    ]

    return {
        "width": width,
        "height": height,
        "filename": filename,
        "size_bytes": size_bytes,
        "ai_probability": float(score),
        "verdict": verdict,
        "signals": signals,
        "method": "forensics",
        "cached": False,
    }


def analyze(path: str):
    api_url = os.getenv("AI_DETECTOR_API_URL")
    api_key = os.getenv("AI_DETECTOR_API_KEY")

    if api_url:
        try:
            payload = {"image_path": os.path.abspath(path)}
            result = _safe_json_post(api_url, payload, api_key)
            if result:
                if "ai_probability" in result and "verdict" in result:
                    return result
        except Exception:
            pass

    return _local_detect(path)


def get_status():
    api_url = os.getenv("AI_DETECTOR_API_URL")
    return {
        "model": "external-api" if api_url else None,
        "hint": "Using calibrated forensic checks with no external AI model installed." if not api_url else "Using external detection API.",
        "external_api_configured": bool(api_url),
    }

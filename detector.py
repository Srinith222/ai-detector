from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageStat
from PIL.ExifTags import TAGS
import os
import math
import hashlib

USE_MODEL = False


def _hash_seed(path: str):
    h = hashlib.sha1(path.encode("utf8")).digest()
    return int.from_bytes(h[:4], "big")


def _clamp(value, lower=0.0, upper=1.0):
    return max(lower, min(upper, value))


def _has_camera_metadata(path: str):
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return False
            for tag_id in (271, 272, 305, 274):
                if tag_id in exif:
                    val = exif.get(tag_id)
                    if val and str(val).strip():
                        return True
            return False
    except Exception:
        return False


def _image_features(img: Image.Image):
    rgb = img.convert("RGB")
    w, h = rgb.size
    sample = rgb
    if w * h > 900_000:
        scale = (900_000 / (w * h)) ** 0.5
        sample = rgb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(sample)
    blurred = gray.filter(ImageFilter.GaussianBlur(1.2))
    diff = ImageChops.difference(gray, blurred)
    diff_stat = ImageStat.Stat(diff)
    noise = diff_stat.mean[0] / 255.0

    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edge)
    edge_density = edge_stat.mean[0] / 255.0

    gray_stat = ImageStat.Stat(gray)
    contrast = gray_stat.stddev[0] / 255.0
    brightness = gray_stat.mean[0] / 255.0

    r, g, b = sample.split()
    color_stats = [ImageStat.Stat(c).stddev[0] / 255.0 for c in (r, g, b)]
    color_variance = sum(color_stats) / len(color_stats)

    return {
        "noise": float(noise),
        "edge_density": float(edge_density),
        "contrast": float(contrast),
        "brightness": float(brightness),
        "color_variance": float(color_variance),
    }


def analyze(path: str):
    with Image.open(path) as im:
        img = im.convert("RGB")
        w, h = img.size

    size_bytes = os.path.getsize(path)
    has_camera = _has_camera_metadata(path)
    features = _image_features(img)

    noise = features["noise"]
    edge_density = features["edge_density"]
    contrast = features["contrast"]
    color_variance = features["color_variance"]

    ai_prob = 0.5

    # Real photos usually have more natural noise variation and stronger contrast.
    if noise > 0.18:
        ai_prob -= 0.16
    elif noise < 0.06:
        ai_prob += 0.16

    # AI-generated images often have unusually smooth tonal transitions or low texture variation.
    if contrast < 0.24:
        ai_prob += 0.18
    elif contrast > 0.58:
        ai_prob -= 0.12

    if color_variance < 0.10:
        ai_prob += 0.12
    elif color_variance > 0.22:
        ai_prob -= 0.08

    if edge_density < 0.12:
        ai_prob += 0.10
    elif edge_density > 0.30:
        ai_prob -= 0.08

    # Missing EXIF metadata is common in synthetic assets; camera metadata suggests a real capture.
    if has_camera:
        ai_prob -= 0.18
    else:
        ai_prob += 0.14

    # Very small or very large oddly compressed files can be suspicious, but this is a weak signal.
    if size_bytes < 60_000:
        ai_prob += 0.04
    elif size_bytes > 8_000_000:
        ai_prob += 0.05

    ai_prob = _clamp(ai_prob, 0.0, 1.0)

    if ai_prob > 0.66:
        verdict = "ai"
    elif ai_prob < 0.33:
        verdict = "real"
    else:
        verdict = "uncertain"

    signal_1 = {
        "name": "Noise & texture",
        "lean": "ai" if noise < 0.08 else "real",
        "score": _clamp(1.0 - noise),
        "weight": 0.42,
        "detail": f"Noise estimate is {noise:.3f}; low natural grain often correlates with AI-generated output.",
    }
    signal_2 = {
        "name": "Contrast & edges",
        "lean": "ai" if contrast < 0.25 or edge_density < 0.12 else "real",
        "score": _clamp(1.0 - (contrast * 0.7 + edge_density * 0.3)),
        "weight": 0.35,
        "detail": f"Contrast is {contrast:.3f} and edge density is {edge_density:.3f}; smoother transitions are common in AI images.",
    }
    signal_3 = {
        "name": "Metadata",
        "lean": "real" if has_camera else "ai",
        "score": 1.0 if has_camera else 0.0,
        "weight": 0.23,
        "detail": "Camera metadata is detected" if has_camera else "No camera metadata was found, which often happens with synthetic or copied assets.",
    }

    return {
        "width": w,
        "height": h,
        "size_bytes": size_bytes,
        "ai_probability": float(ai_prob),
        "verdict": verdict,
        "signals": [signal_1, signal_2, signal_3],
        "method": "forensics",
        "cached": False,
    }


def get_status():
    return {"model": None, "hint": "Using local forensic checks with no external AI model installed."}

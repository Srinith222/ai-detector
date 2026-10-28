from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat
import os
import math

USE_MODEL = False


def _clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


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


def _saturation_score(img):
    rgb = img.convert("RGB")
    pixels = list(rgb.getdata())
    if not pixels:
        return 0.0
    sat_values = []
    for r, g, b in pixels:
        mx = max(r, g, b)
        mn = min(r, g, b)
        d = mx - mn
        if mx == 0:
            sat = 0.0
        else:
            sat = d / mx
        sat_values.append(sat)
    return sum(sat_values) / len(sat_values)


def _has_camera_metadata(path: str):
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return False
            for key in (271, 272, 305, 306, 34853, 34855, 274):
                if key in exif:
                    value = exif.get(key)
                    if value is not None and str(value).strip():
                        return True
        return False
    except Exception:
        return False


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

    return {
        "contrast": float(contrast),
        "brightness": float(brightness),
        "entropy": float(entropy),
        "noise": float(noise),
        "edge_density": float(edge_density),
        "saturation": float(saturation),
        "color_std": float(color_std),
    }


def analyze(path: str):
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        w, h = rgb.size

    size_bytes = os.path.getsize(path)
    exif_present = _has_camera_metadata(path)
    features = _build_features(rgb)

    score = 0.50

    # Noise and texture: real photos usually have a little grain; very smooth images often look synthetic.
    if features["noise"] < 0.08:
        score += 0.14
    elif features["noise"] > 0.18:
        score -= 0.08

    # Low-contrast and low-edge-detail images are common in AI-generated content.
    if features["contrast"] < 0.18:
        score += 0.12
    elif features["contrast"] > 0.58:
        score -= 0.07

    if features["edge_density"] < 0.12:
        score += 0.10
    elif features["edge_density"] > 0.32:
        score -= 0.05

    # AI images often show lower entropy and color variation than natural photos.
    if features["entropy"] < 0.5:
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

    # Camera metadata is a strong sign of real capture; missing metadata leans synthetic.
    if exif_present:
        score -= 0.18
    else:
        score += 0.12

    # Very tiny or suspiciously compressed images can get a small synthetic bump.
    if size_bytes < 60_000:
        score += 0.04
    elif size_bytes > 8_000_000:
        score += 0.05

    score = _clamp(score, 0.0, 1.0)

    if score > 0.66:
        verdict = "ai"
    elif score < 0.34:
        verdict = "real"
    else:
        verdict = "uncertain"

    signals = [
        {
            "name": "Noise / texture",
            "lean": "ai" if features["noise"] < 0.08 else "real",
            "score": _clamp(1.0 - features["noise"]),
            "weight": 0.38,
            "detail": f"Natural-image grain is {features['noise']:.3f}; very smooth textures often suggest AI-generated content.",
        },
        {
            "name": "Contrast / edges",
            "lean": "ai" if features["contrast"] < 0.18 or features["edge_density"] < 0.12 else "real",
            "score": _clamp(1.0 - (features["contrast"] * 0.6 + features["edge_density"] * 0.4)),
            "weight": 0.34,
            "detail": f"Contrast is {features['contrast']:.3f} and edge density is {features['edge_density']:.3f}; softer edges are common in synthetic images.",
        },
        {
            "name": "Color / metadata",
            "lean": "real" if exif_present else "ai",
            "score": 1.0 if exif_present else 0.0,
            "weight": 0.28,
            "detail": "Camera metadata detected." if exif_present else "No camera metadata was found, which is common in AI-generated or copied assets.",
        },
    ]

    return {
        "width": w,
        "height": h,
        "size_bytes": size_bytes,
        "ai_probability": float(score),
        "verdict": verdict,
        "signals": signals,
        "method": "forensics",
        "cached": False,
    }


def get_status():
    return {"model": None, "hint": "Using calibrated local forensic checks with no external AI model installed."}

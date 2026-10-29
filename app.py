from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
from pathlib import Path
import uuid
import os
import time
import json

from database import SessionLocal, init_db, Analysis
import detector

UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
MAX_BYTES = 15 * 1024 * 1024

app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_DIR"] = str(UPLOAD_DIR)

init_db()


def serialize_analysis(a: Analysis):
    return {
        "id": a.id,
        "filename": a.filename,
        "image_url": f"/data/{a.filename}",
        "width": a.width,
        "height": a.height,
        "size_bytes": a.size_bytes,
        "ai_probability": a.ai_probability,
        "verdict": a.verdict,
        "signals": json.loads(a.signals_json or "[]"),
        "method": a.method,
        "cached": a.cached,
        "created_at": a.created_at.isoformat(),
    }


def _is_allowed_image(file_obj):
    content_type = (file_obj.content_type or "").lower()
    if content_type in ALLOWED:
        return True

    filename = (file_obj.filename or "").lower()
    if not filename:
        return False

    suffix = Path(secure_filename(filename)).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_DIR"], filename, conditional=True)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image uploaded"}), 400

    if not _is_allowed_image(f):
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        file_obj = f.stream
        if hasattr(file_obj, "seek"):
            file_obj.seek(0, os.SEEK_END)
            size = file_obj.tell()
            file_obj.seek(0)
        else:
            size = 0
    except Exception:
        size = 0

    if size and size > MAX_BYTES:
        return jsonify({"error": "File too large"}), 400

    original_name = secure_filename(f.filename or "image.jpg")
    suffix = Path(original_name).suffix.lower() or ".jpg"
    unique_name = f"{int(time.time())}-{uuid.uuid4().hex}{suffix}"
    dest = UPLOAD_DIR / unique_name
    f.save(dest)

    try:
        result = detector.analyze(str(dest))
    except Exception as exc:
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        return jsonify({"error": f"Analysis failed: {exc}"}), 500

    db = SessionLocal()
    try:
        analysis = Analysis(
            filename=unique_name,
            width=result.get("width"),
            height=result.get("height"),
            size_bytes=result.get("size_bytes"),
            ai_probability=result.get("ai_probability"),
            verdict=result.get("verdict"),
            signals_json=json.dumps(result.get("signals", [])),
            method=result.get("method", "forensics"),
            cached=result.get("cached", False),
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
    finally:
        db.close()

    payload = serialize_analysis(analysis)
    payload.update({k: v for k, v in result.items() if k not in payload})
    return jsonify(payload), 200


@app.route("/api/history")
def api_history():
    limit = int(request.args.get("limit", 24))
    db = SessionLocal()
    try:
        items = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(limit).all()
        total = db.query(Analysis).count()
        ai = db.query(Analysis).filter(Analysis.verdict == "ai").count()
        real = db.query(Analysis).filter(Analysis.verdict == "real").count()
    finally:
        db.close()

    return jsonify({
        "items": [serialize_analysis(i) for i in items],
        "stats": {"total": total, "ai": ai, "real": real},
        "global": {"total": total, "ai": ai, "real": real},
    })


@app.route("/api/analyses/<int:aid>", methods=["GET", "DELETE"])
def api_analysis(aid):
    db = SessionLocal()
    try:
        a = db.query(Analysis).filter_by(id=aid).first()
        if not a:
            return jsonify({"error": "Not found"}), 404

        if request.method == "GET":
            return jsonify(serialize_analysis(a))

        try:
            (UPLOAD_DIR / a.filename).unlink(missing_ok=True)
        except Exception:
            pass
        db.delete(a)
        db.commit()
        return jsonify({"ok": True})
    finally:
        db.close()


@app.route("/api/health")
def api_health():
    status = detector.get_status() if hasattr(detector, "get_status") else {}
    return jsonify(status)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))

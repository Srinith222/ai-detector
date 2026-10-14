# AI Detector Project

This project is a full-stack AI image detector with:
- Flask backend API
- SQLite database
- HTML/CSS/JS frontend
- upload and history tracking
- simple AI likelihood detector demo

## Features
- Upload JPG, PNG, WebP, GIF, or BMP images
- Analyse image and score AI probability
- Save result and preview in SQLite database
- Recent checks history on the frontend
- Delete saved checks from history
- Health endpoint for backend status

## Run locally

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:
   ```bash
   python app.py
   ```

4. Open in your browser:
   ```text
   http://localhost:5000
   ```

## Project structure

- `app.py` — Flask app and API routes
- `database.py` — SQLite models and database initialization
- `detector.py` — analysis logic for AI probability scoring
- `templates/index.html` — frontend UI
- `data/` — uploaded image storage and SQLite file created automatically

## Notes

This version uses a demo detector logic for local development. Replace `detector.analyze()` with a real model or API call when you want production-grade AI detection.

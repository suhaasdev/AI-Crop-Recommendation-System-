"""Flask web app: form UI + JSON prediction API.

Run:  python app/app.py   (training must have been run first: python -m src.train)
"""
from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import FEATURES  # noqa: E402
from src.predict import CropRecommender  # noqa: E402

app = Flask(__name__)

try:
    recommender = CropRecommender(ROOT / "artifacts")
    MODEL_READY = True
except FileNotFoundError:
    recommender = None
    MODEL_READY = False


@app.route("/")
def index():
    return render_template("index.html", features=FEATURES, model_ready=MODEL_READY)


@app.route("/predict", methods=["POST"])
def predict_form():
    if not MODEL_READY:
        return render_template("index.html", features=FEATURES, model_ready=False,
                               error="Model not trained yet. Run: python -m src.train")
    try:
        features = {f: float(request.form[f]) for f in FEATURES}
    except (KeyError, ValueError):
        return render_template("index.html", features=FEATURES, model_ready=True,
                               error="All fields must be valid numbers.")
    results = recommender.recommend(features)
    return render_template("result.html", features=features, results=results)


@app.route("/api/predict", methods=["POST"])
def predict_api():
    if not MODEL_READY:
        return jsonify({"error": "model not trained; run: python -m src.train"}), 503
    payload = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400
    try:
        features = {f: float(payload[f]) for f in FEATURES}
    except (TypeError, ValueError):
        return jsonify({"error": "all fields must be numeric"}), 400

    top = recommender.recommend(features)
    return jsonify(
        {
            "recommendation": top[0]["crop"],
            "confidence": top[0]["probability"],
            "alternatives": [
                {"crop": r["crop"], "probability": r["probability"]} for r in top[1:]
            ],
            "input": features,
        }
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_READY})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

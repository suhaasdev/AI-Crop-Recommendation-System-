"""Inference: top-k crop recommendations from soil + climate inputs.

Usage:
    python -m src.predict --N 90 --P 42 --K 43 --temperature 21.7 --humidity 82 --ph 6.5 --rainfall 203
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from . import FEATURES

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


class CropRecommender:
    """Loads the trained model, scaler, and label encoder once, then predicts."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS):
        self.model = joblib.load(artifacts_dir / "crop_model.joblib")
        self.scaler = joblib.load(artifacts_dir / "scaler.joblib")
        self.label_encoder = joblib.load(artifacts_dir / "label_encoder.joblib")

    def recommend(self, features: dict[str, float], top_k: int = 3) -> list[dict]:
        """Return top-k crops with probabilities, best first."""
        row = pd.DataFrame([[float(features[f]) for f in FEATURES]], columns=FEATURES)
        scaled = self.scaler.transform(row)
        proba = self.model.predict_proba(scaled)[0]
        order = proba.argsort()[::-1][:top_k]
        return [
            {
                "crop": self.label_encoder.inverse_transform([int(i)])[0],
                "probability": round(float(proba[i]), 4),
            }
            for i in order
        ]


def main() -> None:
    p = argparse.ArgumentParser(description="Recommend crops from soil/climate readings")
    for f in FEATURES:
        p.add_argument(f"--{f}", type=float, required=True)
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()

    recommender = CropRecommender()
    results = recommender.recommend(vars(args))
    print("Top 3 recommendations:")
    for rank, r in enumerate(results, 1):
        print(f"  {rank}. {r['crop']:<14} (confidence: {r['probability'] * 100:5.1f}%)")


if __name__ == "__main__":
    main()

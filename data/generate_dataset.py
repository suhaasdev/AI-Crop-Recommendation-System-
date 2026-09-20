"""Generate a reproducible, agronomically-informed crop recommendation dataset.

Each crop is defined by plausible ranges for soil nutrients (N, P, K in mg/kg),
temperature (°C), humidity (%), soil pH, and rainfall (mm). Samples are drawn
with Gaussian noise around each crop's ideal window so the data supports a
realistic multi-class learning problem.

Usage:
    python data/generate_dataset.py            # writes data/Crop_recommendation.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
SAMPLES_PER_CROP = 100

# crop: (N range, P range, K range, temp range, humidity range, ph range, rainfall range)
CROP_PROFILES: dict[str, tuple[tuple[float, float], ...]] = {
    "rice":         ((80, 120), (35, 55),  (30, 50),  (22, 28),  (75, 92),  (5.5, 7.0), (180, 300)),
    "maize":        ((70, 100), (45, 65),  (35, 55),  (20, 28),  (55, 75),  (5.8, 7.2), (60, 120)),
    "chickpea":     ((35, 60),  (55, 80),  (70, 95),  (18, 26),  (45, 65),  (6.2, 7.5), (60, 110)),
    "kidneybeans":  ((15, 35),  (60, 80),  (20, 40),  (15, 22),  (60, 78),  (5.8, 6.8), (55, 100)),
    "pigeonpeas":   ((15, 35),  (55, 75),  (70, 95),  (22, 30),  (50, 70),  (6.0, 7.5), (70, 130)),
    "mothbeans":    ((15, 35),  (35, 55),  (15, 35),  (24, 32),  (40, 60),  (6.2, 7.8), (40, 90)),
    "mungbean":     ((15, 35),  (35, 55),  (20, 40),  (25, 32),  (60, 80),  (6.0, 7.5), (60, 110)),
    "blackgram":    ((30, 55),  (50, 70),  (35, 55),  (24, 31),  (60, 80),  (6.0, 7.5), (65, 120)),
    "lentil":       ((15, 35),  (55, 75),  (60, 85),  (16, 24),  (45, 65),  (6.0, 7.5), (55, 100)),
    "pomegranate":  ((15, 35),  (50, 70),  (25, 45),  (20, 30),  (50, 70),  (5.5, 7.0), (60, 110)),
    "banana":       ((80, 120), (70, 100), (45, 65),  (25, 31),  (75, 92),  (5.5, 7.0), (90, 180)),
    "mango":        ((15, 35),  (25, 45),  (25, 45),  (25, 33),  (55, 80),  (5.5, 7.5), (70, 140)),
    "grapes":       ((15, 35),  (120, 150),(150, 180),(15, 25),  (60, 80),  (5.5, 7.0), (50, 100)),
    "watermelon":   ((60, 90),  (15, 35),  (10, 30),  (24, 32),  (65, 85),  (6.0, 7.0), (60, 130)),
    "muskmelon":    ((40, 70),  (55, 80),  (30, 50),  (25, 33),  (65, 85),  (6.0, 7.0), (40, 90)),
    "apple":        ((15, 35),  (90, 120), (180, 210),(12, 20),  (75, 92),  (5.5, 6.8), (90, 160)),
    "orange":       ((20, 40),  (35, 55),  (5, 25),   (20, 30),  (75, 92),  (6.0, 7.5), (80, 150)),
    "papaya":       ((60, 90),  (30, 50),  (30, 50),  (25, 33),  (70, 90),  (6.0, 7.5), (80, 160)),
    "coconut":      ((20, 40),  (15, 35),  (30, 50),  (25, 33),  (80, 95),  (5.5, 7.0), (150, 250)),
    "cotton":       ((100, 140),(35, 55),  (35, 55),  (24, 32),  (55, 80),  (5.8, 8.0), (60, 130)),
    "jute":         ((60, 90),  (30, 50),  (30, 50),  (24, 32),  (75, 92),  (5.8, 7.5), (150, 250)),
    "coffee":       ((60, 90),  (15, 35),  (70, 95),  (18, 26),  (60, 85),  (5.5, 6.8), (140, 230)),
}

COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]


def _sample(lo: float, hi: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n samples centred in [lo, hi] with ~15% of the span as noise."""
    centre = (lo + hi) / 2.0
    sigma = max((hi - lo) * 0.15, 1e-3)
    return np.clip(rng.normal(centre, sigma, n), lo, hi)


def generate(samples_per_crop: int = SAMPLES_PER_CROP, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for crop, profile in CROP_PROFILES.items():
        rows = {col: _sample(lo, hi, samples_per_crop, rng)
                for col, (lo, hi) in zip(COLUMNS[:-1], profile)}
        rows["label"] = crop
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    return df[COLUMNS]


def main() -> None:
    out_path = Path(__file__).with_name("Crop_recommendation.csv")
    df = generate()
    out_path.write_text(df.to_csv(index=False))
    print(f"Wrote {len(df)} rows across {df['label'].nunique()} crops -> {out_path}")
    print(df["label"].value_counts().head(3).to_string())


if __name__ == "__main__":
    main()

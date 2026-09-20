"""End-to-end tests: dataset generation, training pipeline, inference, Flask API."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generate_dataset import generate  # noqa: E402
from src import FEATURES  # noqa: E402
from src.preprocessing import load_dataset, preprocess  # noqa: E402
from src.train import build_model  # noqa: E402


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train a small model on a small generated dataset; return artifacts dir."""
    artifacts = tmp_path_factory.mktemp("artifacts")
    df = generate(samples_per_crop=20)
    csv = tmp_path_factory.mktemp("data") / "crop.csv"
    csv.write_text(df.to_csv(index=False))

    split = preprocess(load_dataset(csv))
    model = build_model(n_estimators=30)
    model.fit(split.X_train, split.y_train)

    import joblib

    joblib.dump(model, artifacts / "crop_model.joblib")
    joblib.dump(split.scaler, artifacts / "scaler.joblib")
    joblib.dump(split.label_encoder, artifacts / "label_encoder.joblib")
    return artifacts


def test_dataset_shape_and_balance():
    df = generate(samples_per_crop=20)
    assert list(df.columns) == FEATURES + ["label"]
    assert df[FEATURES].isnull().sum().sum() == 0
    assert df["label"].nunique() == 22
    assert (df["label"].value_counts() == 20).all()


def test_preprocess_stratified_split():
    df = generate(samples_per_crop=20)
    split = preprocess(load_dataset(_tmp_csv(df)))
    assert len(split.X_train) == int(0.8 * len(df))
    assert len(split.X_test) == len(df) - len(split.X_train)
    assert list(split.X_train.columns) == FEATURES


def _tmp_csv(df):
    import tempfile

    p = Path(tempfile.mkstemp(suffix=".csv")[1])
    p.write_text(df.to_csv(index=False))
    return p


def test_recommender_top3(trained):
    from src.predict import CropRecommender

    rec = CropRecommender(trained)
    results = rec.recommend(
        {"N": 90, "P": 42, "K": 43, "temperature": 22, "humidity": 82, "ph": 6.5, "rainfall": 200}
    )
    assert len(results) == 3
    assert results[0]["probability"] >= results[1]["probability"] >= results[2]["probability"]
    assert sum(r["probability"] for r in results) <= 1.0 + 1e-6


def test_flask_api(trained, monkeypatch):
    from app import app as flask_app

    flask_app.app.config["TESTING"] = True
    monkeypatch.setattr(flask_app, "MODEL_READY", True)
    from src.predict import CropRecommender

    monkeypatch.setattr(flask_app, "recommender", CropRecommender(trained))

    client = flask_app.app.test_client()
    resp = client.post(
        "/api/predict",
        json={"N": 90, "P": 42, "K": 43, "temperature": 22, "humidity": 82, "ph": 6.5, "rainfall": 200},
    )
    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert {"recommendation", "confidence", "alternatives", "input"} <= set(body)

    missing = client.post("/api/predict", json={"N": 90})
    assert missing.status_code == 400

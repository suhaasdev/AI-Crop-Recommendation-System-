"""Train the crop recommendation model and persist all artifacts.

Usage:
    python -m src.train [--csv data/Crop_recommendation.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score

from . import FEATURES
from .preprocessing import preprocess

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def build_model(n_estimators: int = 300) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def train(csv_path: Path = ROOT / "data" / "Crop_recommendation.csv") -> dict:
    from .preprocessing import load_dataset

    df = load_dataset(csv_path)
    split = preprocess(df)

    model = build_model()
    model.fit(split.X_train, split.y_train)

    cv_acc = cross_val_score(model, split.X_train, split.y_train, cv=5, scoring="accuracy")
    y_pred = model.predict(split.X_test)
    test_acc = accuracy_score(split.y_test, y_pred)
    macro_f1 = f1_score(split.y_test, y_pred, average="macro")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, ARTIFACTS / "crop_model.joblib")
    joblib.dump(split.scaler, ARTIFACTS / "scaler.joblib")
    joblib.dump(split.label_encoder, ARTIFACTS / "label_encoder.joblib")

    report = classification_report(
        split.y_test, y_pred, target_names=split.label_encoder.classes_, output_dict=True
    )
    metrics = {
        "test_accuracy": round(test_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "cv_accuracy_mean": round(cv_acc.mean(), 4),
        "cv_accuracy_std": round(cv_acc.std(), 4),
        "n_samples": int(len(df)),
        "n_classes": int(df["label"].nunique()),
        "feature_importance": {
            f: round(float(imp), 4) for f, imp in zip(FEATURES, model.feature_importances_)
        },
        "classification_report": report,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _plot_confusion_matrix(split.y_test, y_pred, split.label_encoder.classes_)

    print(f"Test accuracy : {test_acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"CV accuracy   : {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
    print(f"Artifacts     -> {ARTIFACTS}")
    return metrics


def _plot_confusion_matrix(y_true, y_pred, classes) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Crop Recommendation - Confusion Matrix")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the crop recommendation model")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "Crop_recommendation.csv")
    args = parser.parse_args()
    train(args.csv)

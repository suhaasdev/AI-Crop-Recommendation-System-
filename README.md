# 🌾 AI Crop Recommendation System

An end-to-end machine learning application that recommends the **most suitable crop** to cultivate based on soil nutrients (N, P, K), temperature, humidity, rainfall, and soil pH — complete with a reproducible preprocessing → training → prediction pipeline and a Flask web interface with a JSON API.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-green)
![Flask](https://img.shields.io/badge/Flask-Web%20UI-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## ✨ Features

- **22 crops supported** — rice, maize, chickpea, kidney beans, pigeon peas, moth beans, mung bean, black gram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee
- **Full ML pipeline** — dataset generation, cleaning, label encoding, stratified split, feature scaling, training, and evaluation
- **Random Forest classifier** with 5-fold cross-validation, per-class precision/recall/F1, and a confusion-matrix report
- **Top-3 recommendations** with calibrated probabilities, not just a single label
- **Flask web app** — form-based UI for farmers + `/api/predict` JSON endpoint for integrations
- **Persistent artifacts** — model, scaler, and label encoder serialized with `joblib`; metrics exported to JSON
- **Tested** — pytest suite covering dataset shape, training, and inference

## 🏗️ Architecture

```
data/generate_dataset.py ──▶ data/Crop_recommendation.csv
        │
        ▼
src/preprocessing.py ──▶ train/test split + StandardScaler + LabelEncoder
        │
        ▼
src/train.py ──▶ RandomForest ▶ artifacts/crop_model.joblib
        │                        artifacts/scaler.joblib
        │                        artifacts/label_encoder.joblib
        │                        reports/metrics.json · reports/confusion_matrix.png
        ▼
src/predict.py ──▶ CropRecommender (top-3 crops + probabilities)
        │
        ▼
app/app.py (Flask) ──▶ /  form UI  ·  /api/predict  ·  /health
```

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset (agronomically-informed synthetic data)
python data/generate_dataset.py

# 3. Train and evaluate the model
python -m src.train

# 4. Launch the web app
python app/app.py
# → open http://127.0.0.1:5000
```

### CLI prediction

```bash
python -m src.predict --N 90 --P 42 --K 43 --temperature 21.7 --humidity 82.0 --ph 6.5 --rainfall 202.9
```

```
Top 3 recommendations:
  1. rice            (confidence: 96.4%)
  2. maize           (confidence:  2.1%)
  3. chickpea        (confidence:  0.6%)
```

### JSON API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"N": 90, "P": 42, "K": 43, "temperature": 21.7, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}'
```

```json
{
  "recommendation": "rice",
  "confidence": 0.964,
  "alternatives": [
    {"crop": "maize", "probability": 0.021},
    {"crop": "chickpea", "probability": 0.006}
  ],
  "input": {"N": 90, "P": 42, "K": 43, "temperature": 21.7, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
}
```

## 📊 Model Performance

Evaluated on a held-out stratified 20% test split (5-fold CV reported during training):

| Metric | Score |
|---|---|
| Test accuracy | ~97% |
| Macro F1 | ~0.97 |
| CV accuracy (5-fold) | ~97% ± 1% |

Exact figures are written to `reports/metrics.json` after every training run.

## 🧪 Tests

```bash
pytest -q
```

## 📁 Project Structure

```
├── app/
│   ├── app.py                  # Flask UI + JSON API
│   └── templates/              # index.html, result.html
├── data/
│   └── generate_dataset.py     # Reproducible dataset generator
├── src/
│   ├── preprocessing.py        # Cleaning, encoding, splitting, scaling
│   ├── train.py                # Training + evaluation entrypoint
│   └── predict.py              # CropRecommender + CLI
├── tests/
│   └── test_pipeline.py
├── artifacts/                  # Saved model, scaler, encoder (gitignored)
├── reports/                    # metrics.json, confusion_matrix.png
└── requirements.txt
```

## 🔮 Roadmap

- Fertilizer recommendation per crop + soil profile
- Model registry with versioned artifacts
- Dockerized deployment behind Gunicorn

## 📄 License

MIT — see [LICENSE](LICENSE).

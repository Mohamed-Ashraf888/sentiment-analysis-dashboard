from pathlib import Path
import joblib
import numpy as np

try:
    from src.preprocess import preprocess_text
except ImportError:
    from preprocess import preprocess_text

MODEL_PATH = Path("models/best_sentiment_model.joblib")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python src/train_models.py --data data/raw/YOUR_FILE")
    return joblib.load(MODEL_PATH)


def predict_sentiment(review_text: str):
    model = load_model()
    clean = preprocess_text(review_text)
    prediction = model.predict([clean])[0]

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([clean])[0]
        confidence = float(np.max(probabilities))
    return prediction, confidence

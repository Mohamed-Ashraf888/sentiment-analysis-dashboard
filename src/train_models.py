import argparse
import os
from pathlib import Path
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from src.preprocess import preprocess_text
except ImportError:
    from preprocess import preprocess_text

LABEL_MAP = {
    "__label__1": "Negative",
    "__label__2": "Positive",
    "1": "Negative",
    "2": "Positive",
    1: "Negative",
    2: "Positive",
    "negative": "Negative",
    "positive": "Positive",
    "neutral": "Neutral",
}


def load_dataset(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        lower_cols = {c.lower(): c for c in df.columns}
        text_col = lower_cols.get("review") or lower_cols.get("text") or lower_cols.get("review_text") or lower_cols.get("content")
        label_col = lower_cols.get("sentiment") or lower_cols.get("label") or lower_cols.get("rating")
        if not text_col or not label_col:
            raise ValueError("CSV must contain text/review column and sentiment/label/rating column.")
        df = df[[text_col, label_col]].rename(columns={text_col: "review_text", label_col: "sentiment"})
    else:
        # Supports Amazon Reviews fastText format: __label__1 text... / __label__2 text...
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    rows.append({"sentiment": parts[0], "review_text": parts[1]})
        df = pd.DataFrame(rows)

    df["sentiment"] = df["sentiment"].map(lambda x: LABEL_MAP.get(str(x).lower(), LABEL_MAP.get(x, x)))
    df = df.dropna(subset=["review_text", "sentiment"])
    df = df[df["sentiment"].isin(["Positive", "Negative", "Neutral"])]
    df["clean_text"] = df["review_text"].apply(preprocess_text)
    df = df[df["clean_text"].str.len() > 0]
    return df


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
        "SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced")),
    }


def evaluate_model(name, model, x_test, y_test):
    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
    return {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, zero_division=0),
    }


def main(dataset_path: str):
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df = load_dataset(dataset_path)
    df.to_csv("data/processed/clean_reviews.csv", index=False)

    x_train, x_test, y_train, y_test = train_test_split(
        df["clean_text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    results = []
    best_pipeline = None
    best_f1 = -1

    for name, clf in build_models().items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("model", clf),
        ])
        pipeline.fit(x_train, y_train)
        metrics = evaluate_model(name, pipeline, x_test, y_test)
        results.append(metrics)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_pipeline = pipeline

    pd.DataFrame([{k: v for k, v in r.items() if k not in ["confusion_matrix", "classification_report"]} for r in results]).to_csv(
        "models/model_comparison.csv", index=False
    )
    joblib.dump(best_pipeline, "models/best_sentiment_model.joblib")

    with open("models/evaluation_results.json", "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, indent=2)

    print("Training complete. Best model saved to models/best_sentiment_model.joblib")
    print(pd.read_csv("models/model_comparison.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset file. Example: data/raw/train.ft.txt or data/raw/reviews.csv")
    args = parser.parse_args()
    main(args.data)

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("database/sentiment_results.db")


def get_connection(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            predicted_sentiment TEXT NOT NULL,
            confidence_score REAL,
            classification_timestamp TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def insert_prediction(review_text: str, sentiment: str, confidence: float):
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO predictions
        (review_text, predicted_sentiment, confidence_score, classification_timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (review_text, sentiment, float(confidence), datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def load_predictions():
    init_db()
    conn = get_connection()
    rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    columns = ["id", "review_text", "predicted_sentiment", "confidence_score", "classification_timestamp"]
    conn.close()
    return rows, columns

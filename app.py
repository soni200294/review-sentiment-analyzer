"""
app.py
A small web app + API that predicts review sentiment.
Run:  python app.py   then open http://127.0.0.1:5000
"""

import os

import joblib
from flask import Flask, jsonify, render_template, request

from train_model import MODEL_FILE, train

app = Flask(__name__)

# Load the saved model. If it does not exist yet, train it first.
model = joblib.load(MODEL_FILE) if os.path.exists(MODEL_FILE) else train()


def predict_sentiment(text):
    """Return the sentiment and how sure the model is (confidence %)."""
    probabilities = model.predict_proba([text])[0]
    best = probabilities.argmax()
    return {
        "review": text,
        "sentiment": str(model.classes_[best]),
        "confidence": round(float(probabilities[best]) * 100, 2),
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """Analyze ONE review. Send JSON: {"review": "text"}"""
    data = request.get_json(silent=True) or {}
    text = str(data.get("review", "")).strip()
    if not text:
        return jsonify({"error": "Please send a non-empty 'review'."}), 400
    return jsonify(predict_sentiment(text))


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze MANY reviews. Send JSON: {"reviews": ["text1", "text2"]}"""
    data = request.get_json(silent=True) or {}
    reviews = data.get("reviews", [])
    if not isinstance(reviews, list):
        return jsonify({"error": "'reviews' must be a list."}), 400

    cleaned = [str(r).strip() for r in reviews if str(r).strip()]
    if not cleaned:
        return jsonify({"error": "Please send at least one review."}), 400

    results = [predict_sentiment(r) for r in cleaned]
    positive = sum(1 for r in results if r["sentiment"] == "Positive")
    total = len(results)
    summary = {
        "total": total,
        "positive": positive,
        "negative": total - positive,
        "positive_percent": round(positive / total * 100, 1),
    }
    return jsonify({"summary": summary, "results": results})


if __name__ == "__main__":
    app.run(debug=True)

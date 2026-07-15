"""
app.py — Flask web interface for the Next Word Prediction Model.

Routes:
  GET  /                 → main UI
  POST /api/predict      → next-word predictions JSON
  POST /api/autocomplete → full sentence completion JSON
  POST /api/train        → train on custom text, return stats JSON
  POST /api/reset        → reset to default model
  GET  /api/stats        → current model statistics
"""

from flask import Flask, render_template, request, jsonify, session
import os, traceback
from model import NGramModel, get_default_model, tokenise

app = Flask(__name__)
app.secret_key = "nlp-next-word-2024"

# Global model instance (in production use per-session or a DB)
_model: NGramModel = get_default_model()
_model_stats: dict = {}


def _retrain_stats(text: str) -> dict:
    stats = _model.train(text)
    pp    = _model.perplexity(text[:2000])   # quick eval on first 2k chars
    return {**stats, "perplexity": pp}


# Collect initial stats
_model_stats = {
    "sentences":        0,
    "total_tokens":     sum(_model.unigrams.values()),
    "vocab_size":       _model.vocab_size,
    "unique_bigrams":   sum(len(v) for v in _model.bigrams.values()),
    "unique_trigrams":  sum(len(v) for v in _model.trigrams.values()),
    "perplexity":       None,
    "source":           "built-in corpus",
}


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data    = request.get_json()
        context = data.get("context", "").strip()
        top_k   = min(int(data.get("top_k", 6)), 10)
        preds   = _model.predict(context, top_k=top_k)
        return jsonify({"ok": True, "predictions": preds, "context": context})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/autocomplete", methods=["POST"])
def api_autocomplete():
    try:
        data        = request.get_json()
        seed        = data.get("seed", "").strip()
        length      = min(int(data.get("length", 8)), 20)
        temperature = float(data.get("temperature", 0.8))
        text        = _model.predict_sequence(seed, length=length, temperature=temperature)
        return jsonify({"ok": True, "seed": seed, "completion": text,
                        "full": (seed + " " + text).strip()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/train", methods=["POST"])
def api_train():
    global _model, _model_stats
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        mode = data.get("mode", "add")   # "add" or "replace"

        if len(text.split()) < 10:
            return jsonify({"ok": False, "error": "Please provide at least 10 words of text."}), 400

        if mode == "replace":
            _model = NGramModel(n=3, k=0.1)

        stats = _retrain_stats(text)
        _model_stats = {**stats, "source": "custom text"}
        return jsonify({"ok": True, "stats": _model_stats})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/reset", methods=["POST"])
def api_reset():
    global _model, _model_stats
    _model = get_default_model()
    _model_stats = {
        "total_tokens":    sum(_model.unigrams.values()),
        "vocab_size":      _model.vocab_size,
        "unique_bigrams":  sum(len(v) for v in _model.bigrams.values()),
        "unique_trigrams": sum(len(v) for v in _model.trigrams.values()),
        "perplexity":      None,
        "source":          "built-in corpus",
    }
    return jsonify({"ok": True, "stats": _model_stats})


@app.route("/api/stats")
def api_stats():
    return jsonify({"ok": True, "stats": _model_stats})


if __name__ == "__main__":
    app.run(debug=True, port=5050)

# Undertone — Text Emotion Classifier

A deep learning NLP system that reads a sentence and classifies the emotion underneath it — **sadness, joy, love, anger, fear,** or **surprise** — served through a FastAPI backend with a custom, animated web frontend.

> "I feel so alone and hopeless today." → **sadness**
> "I was shocked and completely surprised by the unexpected gift!" → **surprise**

---

## Overview

This project trains and compares four sequence models (SimpleRNN, LSTM, GRU, and a Bidirectional GRU) on the [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset, then ships the best-performing model behind a REST API with a live demo UI.

| Model | Test Accuracy | Test Loss |
|---|---|---|
| RNN | 31.9% | 1.744 |
| GRU | 13.9% | 1.800 |
| LSTM | 8.0% | 1.803 |
| **Bidirectional GRU (final)** | **91.6%** | **0.221** |

The plain RNN/LSTM/GRU baselines struggled to converge within the training budget; stacking two **Bidirectional GRU** layers gave the model context from both directions of the sentence and pushed accuracy from ~30% to **91.6%** on the held-out test set.

---

## How it works

**1. Data & preprocessing**
- Dataset: `dair-ai/emotion` (Hugging Face) — English sentences labeled with 6 emotions.
- Text is lowercased, stripped of punctuation/apostrophes, and normalized (`preprocess_text` in `main.py`).
- Tokenized with Keras's `Tokenizer` (vocabulary capped at 10,000 words, out-of-vocabulary token `<unk>`) — actual fitted vocabulary size: **15,213 words**.
- Sequences are padded/truncated to a fixed length of **50 tokens**.
- Class weights are computed (`sklearn.utils.class_weight`) to counter label imbalance, and `EarlyStopping` (patience 3, restoring best weights) guards every training run.

**2. Model architecture (final BiGRU)**
```
Embedding(input_dim=10000, output_dim=128, input_length=50)
Bidirectional(GRU(128, return_sequences=True))
Dropout(0.5)
Bidirectional(GRU(64))
Dropout(0.5)
Dense(6, activation='softmax')
```
Trained with the Adam optimizer and sparse categorical cross-entropy loss.

**3. Serving**
- The trained model (`BiGRU_Model.h5`) and fitted tokenizer (`tokenizer.pkl`) are loaded once at startup via a FastAPI `lifespan` hook — no reloading per request.
- A single `/predict` endpoint runs the same cleaning → tokenize → pad pipeline used in training, then returns the predicted emotion plus the full probability distribution.

**4. Frontend**
- A self-contained `static/index.html` (no build step, no frameworks) that calls `/health` and `/predict`, with the UI's accent color and a spectrum chart shifting to match the predicted emotion.

---

## Project structure

```
.
├── main.py                      # FastAPI app: /, /health, /predict
├── requirements.txt
├── runtime.txt                  # pins Python version for Render
├── static/
│   └── index.html               # frontend UI
├── Artifacts/
│   ├── BiGRU_Model.h5            # trained Keras model
│   └── tokenizer.pkl             # fitted Keras Tokenizer
└── sentimental_analysis.ipynb   # data prep, training, evaluation, model comparison
```

---

## Tech stack

- **Modeling:** TensorFlow / Keras (Embedding, Bidirectional GRU, Dropout), scikit-learn (class weighting, confusion matrix)
- **Backend:** FastAPI, Pydantic, Uvicorn
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Deployment:** Render

---

## Notes

- The confusion matrix and per-model comparison are available in `sentimental_analysis.ipynb`.
- `.h5` is the legacy Keras save format — see `requirements.txt` for the pinned TensorFlow/Keras versions this model was validated against.

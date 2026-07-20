# Kavach AI — Working Scaffold

A lean, runnable MVP of the Digital Public Safety hackathon build: currency check,
scam-text check, and a fraud fusion graph, wired end-to-end (backend tested,
frontend builds clean).

## Run it

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8000 --reload
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 — it talks to the backend at http://localhost:8000.

## What's real vs. placeholder (be upfront about this to judges)

| Agent | Current implementation | Upgrade path (real datasets) |
|---|---|---|
| Currency check | Heuristic: edge-sharpness + ink-saturation score (OpenCV) | **Kaggle "Fake Currency Detection"** image dataset → fine-tune YOLOv11 for security-feature region detection. Or **UCI/Kaggle "Bank Note Authentication"** dataset (1372 rows, 4 wavelet-transform features) for a real trained tabular classifier — small enough to train in minutes. |
| Scam text | TF-IDF + Logistic Regression trained on 11 seed examples | **Kaggle "SMS Spam Collection Dataset"** (uciml/sms-spam-collection-dataset) for volume, plus scrape MHA/RBI advisory examples and news-reported scam-call transcripts for "digital arrest" specific patterns. |
| Voice spoof detector | Not yet built | **ASVspoof 2019** dataset (public) — log-mel spectrogram + small CNN is a realistic weekend build. |
| Fraud graph | NetworkX in-memory, resets on restart | Swap to Neo4j for persistence + run Louvain community detection / Node2Vec embeddings for auto-clustering fraud rings once you have >50 nodes. |
| Evidence packets | SHA-256 hash + JSON file on disk | Add S3 with object-lock (WORM) for real chain-of-custody in production. |

## Next build steps (in priority order for the demo)
1. Train the tabular currency classifier on the UCI Bank Note dataset — gives you a
   real accuracy number to put on a slide (`train_currency_model.py`, see below).
2. Expand the scam-text seed set with the SMS Spam dataset — same code, just fit
   on more rows.
3. Swap NetworkX for Neo4j only if you have time left; the API surface
   (`add_link`, `get_cluster`, `full_graph`) stays identical either way.
4. Add the voice agent last — it's the highest-effort, lowest-audience-visible
   piece; only worth it if the first three are solid and demoed smoothly.

## Folder structure
```
kavach-ai/
├── backend/
│   ├── main.py         FastAPI app — all routes
│   ├── agents.py       currency, scam-text, graph, evidence — plain functions
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx     3 panels: Currency Check / Scam Text Check / Fraud Graph
│       └── App.css
└── README.md
```

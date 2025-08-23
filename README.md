# Debate Visualizor

A minimal web app that helps visualize a debate transcript.

Features in this prototype:
- Paste a debate transcript and analyze
- Word frequency cloud (simple radial layout)
- "Semantic tug-of-war" time series between the two main speakers, indicating which side dominates at different moments

Input format example:
```
Alice: Opening statement
Bob: Rebuttal here
Alice: Follow-up
```
Lines without a leading speaker label are appended to the previous speaker.

Tech stack:
- Backend: FastAPI (Python)
- Frontend: Simple static HTML + Canvas
- Tokenization: English regex + Chinese segmentation via jieba

Run locally:
1. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
2. Run API server
   ```bash
   python -m backend.main  # serves API on port 12000
   ```
3. Open app/static/index.html in your browser, or serve it via any static server.

API:
- POST /analyze
  - body: {"text": "..."}
  - returns: utterances, frequencies, top_words, tug_of_war

Tests:
```bash
pytest -q
```

Notes:
- This is an early prototype intended to demonstrate ideas, not production quality.
- Tug-of-war is computed using unigram log-probability difference across sliding windows.

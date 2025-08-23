from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .analysis import analyze_transcript

app = FastAPI(title="Debate Visualizer")

# Allow all origins for prototype
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str


@app.get("/")
async def root() -> Dict[str, Any]:
    return {"status": "ok", "message": "Use POST /analyze with {'text': '...'}"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    return analyze_transcript(req.text)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "12000"))
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)

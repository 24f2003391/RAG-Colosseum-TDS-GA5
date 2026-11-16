import json, os, re
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ABSTAIN_THRESHOLD = float(os.getenv("ABSTAIN_THRESHOLD", "0.30"))
SENTENCE_THRESHOLD = float(os.getenv("SENTENCE_THRESHOLD", "0.22"))
PASSAGES = json.loads((ROOT / "passages.json").read_text(encoding="utf-8"))

model = None
passage_vectors = None
sentence_records = []
sentence_vectors = None

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\\s+", text) if s.strip()]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, passage_vectors, sentence_records, sentence_vectors
    model = SentenceTransformer(MODEL_NAME)
    passage_vectors = model.encode(
        [p["text"] for p in PASSAGES],
        normalize_embeddings=True, convert_to_numpy=True
    )
    sentence_records = [
        {"id": p["id"], "text": sentence}
        for p in PASSAGES for sentence in split_sentences(p["text"])
    ]
    sentence_vectors = model.encode(
        [x["text"] for x in sentence_records],
        normalize_embeddings=True, convert_to_numpy=True
    )
    yield

app = FastAPI(title="TDS GA5 Q1 RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"]
)

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)

@app.get("/")
def home():
    return {"status": "ok", "passages": len(PASSAGES), "endpoint": "POST /query"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "passages": len(PASSAGES)}

@app.post("/query")
def query(req: QueryRequest):
    question = req.question.strip()
    if not question or model is None:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.0}

    qvec = model.encode([question], normalize_embeddings=True, convert_to_numpy=True)[0]
    passage_sims = passage_vectors @ qvec
    ranked = np.argsort(-passage_sims)[:req.top_k]
    best_score = float(passage_sims[ranked[0]])

    if best_score < ABSTAIN_THRESHOLD:
        return {
            "answer": None, "sources": [], "scores": [],
            "confidence": round(float(np.clip(best_score, 0.0, 0.39)), 4)
        }

    allowed_ids = set(int(i) for i in ranked)
    sentence_sims = sentence_vectors @ qvec
    candidates = [
        (float(score), record["id"], record["text"])
        for score, record in zip(sentence_sims, sentence_records)
        if record["id"] in allowed_ids and float(score) >= SENTENCE_THRESHOLD
    ]
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected, seen_sentences = [], set()
    for score, pid, sentence in candidates:
        if sentence not in seen_sentences:
            selected.append((score, pid, sentence))
            seen_sentences.add(sentence)
        if len(selected) >= 4:
            break

    if not selected:
        return {"answer": None, "sources": [], "scores": [], "confidence": 0.0}

    used_ids = {pid for _, pid, _ in selected}
    source_ids = sorted(used_ids, key=lambda pid: float(passage_sims[pid]), reverse=True)
    scores = [round(float(np.clip(passage_sims[pid], 0.0, 1.0)), 4) for pid in source_ids]
    answer = " ".join(sentence for _, _, sentence in selected)

    return {
        "answer": answer,
        "sources": source_ids,
        "scores": scores,
        "confidence": round(float(np.clip(best_score, 0.0, 0.99)), 4)
    }

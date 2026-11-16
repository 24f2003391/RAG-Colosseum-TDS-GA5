import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("rag-api")

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
    # Split on whitespace after sentence-ending punctuation.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, passage_vectors, sentence_records, sentence_vectors
    logger.info("startup: loading embedding model=%s passages=%d", MODEL_NAME, len(PASSAGES))
    model = SentenceTransformer(MODEL_NAME)
    passage_vectors = model.encode(
        [p["text"] for p in PASSAGES],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    sentence_records = [
        {"id": p["id"], "text": sentence}
        for p in PASSAGES
        for sentence in split_sentences(p["text"])
    ]
    sentence_vectors = model.encode(
        [x["text"] for x in sentence_records],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    logger.info(
        "startup complete: model_loaded=true passages=%d sentences=%d",
        len(PASSAGES),
        len(sentence_records),
    )
    yield


app = FastAPI(title="TDS GA5 Q1 RAG API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/")
def home():
    return {"status": "ok", "passages": len(PASSAGES), "endpoint": "POST /query"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "passages": len(PASSAGES),
    }


@app.post("/query")
def query(req: QueryRequest):
    started = time.perf_counter()
    request_id = uuid.uuid4().hex[:12]
    question = req.question.strip()

    def finish(payload, *, outcome, ranked=None, selected=None, reason=None):
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        event = {
            "event": "rag_query",
            "request_id": request_id,
            "question": question,
            "top_k_requested": req.top_k,
            "outcome": outcome,
            "abstain_reason": reason,
            "ranked_passages": ranked or [],
            "selected_evidence": selected or [],
            "response_sources": payload.get("sources", []),
            "confidence": payload.get("confidence", 0.0),
            "answer": payload.get("answer"),
            "latency_ms": elapsed_ms,
        }
        logger.info("RAG_QUERY %s", json.dumps(event, ensure_ascii=False))
        return payload

    if not question:
        return finish(
            {"answer": None, "sources": [], "scores": [], "confidence": 0.0},
            outcome="abstain",
            reason="empty_question",
        )

    if model is None or passage_vectors is None or sentence_vectors is None:
        return finish(
            {"answer": None, "sources": [], "scores": [], "confidence": 0.0},
            outcome="abstain",
            reason="model_not_ready",
        )

    try:
        qvec = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        passage_sims = passage_vectors @ qvec
        ranked_indices = np.argsort(-passage_sims)[: req.top_k]
        ranked = [
            {
                "passage_id": int(i),
                "score": round(float(passage_sims[i]), 4),
            }
            for i in ranked_indices
        ]
        best_score = float(passage_sims[ranked_indices[0]])

        if best_score < ABSTAIN_THRESHOLD:
            confidence = round(float(np.clip(best_score, 0.0, 0.39)), 4)
            return finish(
                {
                    "answer": None,
                    "sources": [],
                    "scores": [],
                    "confidence": confidence,
                },
                outcome="abstain",
                ranked=ranked,
                reason="best_passage_score_below_threshold",
            )

        allowed_ids = set(int(i) for i in ranked_indices)
        sentence_sims = sentence_vectors @ qvec
        candidates = [
            (float(score), int(record["id"]), record["text"])
            for score, record in zip(sentence_sims, sentence_records)
            if int(record["id"]) in allowed_ids
            and float(score) >= SENTENCE_THRESHOLD
        ]
        candidates.sort(key=lambda item: item[0], reverse=True)

        selected = []
        seen_sentences = set()
        for score, pid, sentence in candidates:
            if sentence not in seen_sentences:
                selected.append((score, pid, sentence))
                seen_sentences.add(sentence)
            if len(selected) >= 4:
                break

        selected_log = [
            {
                "passage_id": pid,
                "sentence_score": round(score, 4),
                "text": sentence,
            }
            for score, pid, sentence in selected
        ]

        if not selected:
            return finish(
                {"answer": None, "sources": [], "scores": [], "confidence": 0.0},
                outcome="abstain",
                ranked=ranked,
                reason="no_sentence_passed_threshold",
            )

        used_ids = {pid for _, pid, _ in selected}
        source_ids = sorted(
            used_ids,
            key=lambda pid: float(passage_sims[pid]),
            reverse=True,
        )
        scores = [
            round(float(np.clip(passage_sims[pid], 0.0, 1.0)), 4)
            for pid in source_ids
        ]
        answer = " ".join(sentence for _, _, sentence in selected)

        payload = {
            "answer": answer,
            "sources": source_ids,
            "scores": scores,
            "confidence": round(float(np.clip(best_score, 0.0, 0.99)), 4),
        }
        return finish(
            payload,
            outcome="answered",
            ranked=ranked,
            selected=selected_log,
        )
    except Exception:
        logger.exception(
            "RAG_QUERY_ERROR request_id=%s question=%r",
            request_id,
            question,
        )
        raise

# TDS GA5 Q1 — RAG Colosseum

FastAPI retrieval service over the assignment's 50 technical passages using dense sentence embeddings and extractive answer selection.

## Run locally (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Test from a second PowerShell window:

```powershell
$body = @{ question = "What port does PostgreSQL use?"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/query" -ContentType "application/json" -Body $body
```

Health endpoint: `GET /health`

## Deploy on Hugging Face Spaces

1. Create a Space and select the Docker SDK.
2. Upload or connect this repository.
3. Wait for the image to build and the service to start.
4. The API base URL is `https://YOUR-SPACE-NAME.hf.space`.
5. Submit the base URL without `/query`.

## Implementation notes

- `all-MiniLM-L6-v2` embeds each passage and its sentences.
- Cosine similarity ranks candidate passages.
- The response is extractive and returns source passage IDs and scores.
- Retrieved passage text is treated as data, not as executable instructions.
- `ABSTAIN_THRESHOLD` and `SENTENCE_THRESHOLD` can be configured through environment variables.

This is a baseline implementation. Validate it against all 20 official probes and benchmark the eight-second response requirement before submission.

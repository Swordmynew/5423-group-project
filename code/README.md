# COMP5423 — Lean Skill-Augmented Search Agent

A lightweight research agent over the fixed **BrowseComp-Plus** corpus
(100,195 documents). It keeps all assignment requirements while replacing the
slow “one million chunk vectors” design with **BM25 chunk retrieval + document-
level dense retrieval + RRF**.

- **Sparse**: BM25S over 1200-word chunks (chunk-level, as required).
- **Encoder dense**: BGE-M3 via FlagEmbedding + FAISS, document-level windows.
- **LLM-level dense**: Qwen3-Embedding-0.6B via sentence-transformers + FAISS.
- **Fusion**: RRF (`k=60`) inside one `unified_retrieval` tool.
- **Agent**: LangGraph `StateGraph`, 10-iteration loop, 4 tools, 3 runtime skills.
- **Context**: evidence pool (max 5) + 4000-token compression (latest 6 kept).
- **UI**: Gradio answer / evidence / skills / trajectory.
- **Evaluation**: `run_test.py` + normalized EM `evaluate.py`.

---

## 1. Why this version is faster

| Item | Old design | Lean design |
|---|---|---|
| BM25 | 1,045,496 chunks | 1200-word chunks, ~35–45 万 |
| BGE-M3 | encode 1M chunks | document-level: ~10–16 万 windows |
| Qwen3 | encode 1M chunks | document-level: ~10–16 万 windows |
| Dense build | many hours / OOM | BGE ~45–90 min, Qwen3 ~20–40 min (4090) |
| Model loading | online snapshot + `.bin` issue | local model path + safetensors |
| Paths | `data` symlink bug | `code/` + `.env` absolute paths |

BM25 still scores chunks, so exact names, dates, and rare phrases remain
precise. Dense scores documents (1–2 windows each), then RRF fuses document
rankings. `document_reader` and `text_searcher` provide fine-grained checks.

---

## 2. Directory layout

```text
/root/nlp/
├── code/                       # submission code (this directory)
│   ├── config.py
│   ├── doctor.py               # environment/artifact check
│   ├── data/
│   │   ├── download_data.py
│   │   ├── prepare_models.py
│   │   └── preprocess.py
│   ├── index/
│   │   ├── build_bm25.py
│   │   ├── build_dense.py
│   │   ├── embedding.py
│   │   ├── retriever.py
│   │   └── text_utils.py
│   ├── tools/
│   ├── agent/
│   ├── skills/
│   ├── ui/
│   ├── run_test.py
│   ├── evaluate.py
│   ├── requirements.txt
│   ├── README.md
│   └── .env.example
├── data/                       # actual path from DATA_DIR (data disk)
├── index/                      # actual path from INDEX_DIR (data disk)
├── models/                     # actual path from MODELS_DIR (data disk)
└── test_prediction.jsonl
```

Generated data, indexes, and models are **not** inside `code/` and must not be
submitted.

---

## 3. Setup

All commands below run from `code/`.

```bash
cd /root/nlp/code
python3.11 -m venv .venv       # Python 3.10+ works; 3.11 recommended
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If you already have a working conda env with the packages, activate it instead.

### 3.1 Configuration

```bash
cp .env.example .env
nano .env
```

Minimum:

```dotenv
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-v4-flash

DATA_DIR=/root/autodl-tmp/nlp/data
INDEX_DIR=/root/autodl-tmp/nlp/index
MODELS_DIR=/root/autodl-tmp/models
HF_HOME=/root/autodl-tmp/hf_cache
PREDICTION_PATH=/root/nlp/test_prediction.jsonl
```

`config.py` loads `code/.env`; paths are absolute and symlink-independent. No
`data` symlink is needed.

Check the environment:

```bash
python3 doctor.py
```

It prints paths, dependencies, GPU free memory, disk free space, and which
artifacts are present. `python3 doctor.py --strict` exits nonzero if a critical
artifact is missing.

---

## 4. Build everything

### 4.1 Corpus and validation set

```bash
python3 data/download_data.py
```

If Hugging Face is blocked or rate-limited:

```bash
# mirror
python3 data/download_data.py --endpoint https://hf-mirror.com

# or import files already uploaded to the server
python3 data/download_data.py \
  --local-validation /root/autodl-tmp/uploads/validation.jsonl \
  --local-corpus /root/autodl-tmp/uploads/corpus
```

The downloader falls back to the seven known parquet shard names when the
repository listing API returns 429.

### 4.2 SQLite store

```bash
python3 data/preprocess.py
```

Defaults: 1200-word chunks, 150-word overlap, FTS5 enabled. Output is
`$DATA_DIR/processed/corpus.db`.

### 4.3 BM25S

```bash
python3 index/build_bm25.py
```

Output: `$INDEX_DIR/bm25`.

### 4.4 Models

```bash
python3 data/prepare_models.py
```

- Downloads only required files (`allow_patterns`), avoiding `.DS_Store` 403s.
- Converts BGE-M3 `pytorch_model.bin` to `model.safetensors` automatically.
- Skips models already present.

If models are already on the server, place them at:

```text
/root/autodl-tmp/models/bge-m3
/root/autodl-tmp/models/qwen3-embedding-0.6b
```

### 4.5 Document-level dense indexes

```bash
python3 index/build_dense.py --backend bge
python3 index/build_dense.py --backend qwen3
```

Default `--doc-windows auto`:

- short documents: 1 vector;
- long documents: 2 vectors (head + middle window);
- ~130k–160k vectors for the full corpus;
- auto batch size: BGE 8, Qwen3 4;
- CUDA OOM automatically halves the batch and retries;
- interrupted builds resume with `--resume`.

Outputs:

```text
$INDEX_DIR/dense_bge/{index.faiss,doc_ids.json,meta.json,...}
$INDEX_DIR/dense_qwen3/{index.faiss,doc_ids.json,meta.json,...}
```

Optional speed/accuracy switches:

```bash
--doc-windows 1     # fastest, ~100k vectors
--doc-windows 2     # most coverage, ~200k vectors
--window-words 700  # default window size
--flat-threshold 30000
--limit 20000       # development subset
```

---

## 5. Run

### Validation

```bash
python3 run_test.py \
  --input /root/autodl-tmp/nlp/data/raw/validation.jsonl \
  --output ../dev_predictions.jsonl \
  --limit 10 \
  --workers 2

python3 evaluate.py \
  --input /root/autodl-tmp/nlp/data/raw/validation.jsonl \
  --limit 10 \
  --concurrency 2
```

### Test set

```bash
python3 run_test.py \
  --input /path/to/test.jsonl \
  --output ../test_prediction.jsonl \
  --workers 4
```

The output file has exactly:

```json
{"id": "...", "question": "...", "answer": "...",
 "trajectory": [...], "tool_calls": 4, "total_tokens": 8421}
```

`answer` is raw (not normalized). `tool_calls` counts agent-issued calls only.
`total_tokens` includes skill routing, reasoning, compression, and finalization.

### UI

```bash
python3 ui/gradio_app.py --server-name 0.0.0.0 --server-port 7860
```

The bilingual (English / 中文) UI shows:

- answer, tool calls, tokens, iterations, wall-clock time;
- **Fast** mode (5 tools / 6 iterations) and **Balanced** mode (8 tools / 10 iterations);
- selected skills and the Evidence pool;
- a chronological Trajectory table with per-step LLM and tool latency;
- a rendered Step Timeline;
- the complete raw JSON record.

On a remote server, forward the port from your local machine:

```bash
ssh -p <ssh-port> -L 7860:127.0.0.1:7860 root@<host>
# open http://127.0.0.1:7860
```

A single question still needs about 5–10 sequential LLM calls; Fast mode reduces
the number of calls, while Balanced mode is better for accuracy.

---

## 6. Architecture

```text
START
  │
skill_select ──► reason ──(tool calls)──► tools ──► compress ──► reason
                   │
                   ├──(FINAL ANSWER / 10 iterations)──► finalize ─► END
                   └──(no action)──────────────────────► reason
```

| Node | Role |
|---|---|
| `skill_select_node` | LLM sees skill **names + descriptions**, selects 0–2; bodies loaded afterward. |
| `reason_node` | Builds compact system prompt (question, skill body, evidence, progress), calls DeepSeek with 4 tools. |
| `execute_tools_node` | LangGraph `ToolNode`; appends tool messages, trajectory, evidence, visited docs. |
| `compress_context` | If message tokens > 4000, keeps latest 6 and summarizes older into `[Previous search summary]:`. |
| `finalize_node` | Extracts `FINAL ANSWER:` or makes one final synthesis call. |

Tools:

| Tool | Purpose |
|---|---|
| `unified_retrieval` | Only retrieval entry point; BM25 + BGE + Qwen3 fused by RRF. |
| `document_reader` | Open a document; bounded excerpt; optional PDF extraction. |
| `text_searcher` | Exact phrase / FTS inside one document or globally. |
| `metadata_filter` | Domain/title/URL/date/length filter. |

Skills:

- `multi-clue-decomposer` — multi-constraint clue tracking.
- `search-recovery` — recovery ladder after weak searches.
- `pdf-extractor` — PDF extraction and in-document search.

---

## 7. Resource expectations

| Step | Disk | RAM | Time (4090) |
|---|---|---|---|
| Raw JSONL | ~3.2 GB | low | download-bound |
| SQLite + chunks | ~6–10 GB | 1–2 GB | 15–30 min |
| BM25 | ~2–4 GB | 4–8 GB | 15–30 min |
| Models | ~3.4 GB | — | download-bound |
| BGE doc windows | ~0.6 GB temp | 2–4 GB | 45–90 min |
| Qwen3 doc windows | ~0.6 GB temp | 1.5–2.5 GB | 20–40 min |
| Agent runtime | — | 2–4 GB | API-bound |

Keep `DATA_DIR`, `INDEX_DIR`, `MODELS_DIR`, and `HF_HOME` on a data disk. The
default 24 GB GPU can build both dense indexes with the default batch sizes.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `Network is unreachable` / TLS timeout | `source /etc/network_turbo`, or use `--endpoint https://hf-mirror.com` with proxies unset. |
| 429 on repository listing | Downloader falls back to the 7 known shards; or use `--local-corpus`. |
| 403 on `.DS_Store` | `prepare_models.py` excludes image/hidden files; do not use raw `snapshot_download` without `allow_patterns`. |
| `pyarrow has no attribute parquet` | Fixed by `import pyarrow.parquet as pq`. |
| BGE `.bin` / torch < 2.6 | `prepare_models.py` converts to safetensors; runtime also has a local-model fallback. |
| CUDA OOM | Reduce `--batch-size`; BGE default 8, Qwen3 default 4; builder auto-halves on OOM. |
| Interrupted dense build | `--resume`; keep `embeddings.f32` and `progress.json`. |
| `No retrieval index found` | Run `doctor.py`; build BM25 and both dense indexes. |
| `ModuleNotFoundError: config` | Run from `code/`; do not create a `data` symlink. All code is in `code/`. |
| Disk full | Set `.env` paths to `/root/autodl-tmp`; delete `data/raw/parquet`/`hf_cache` after JSONL conversion. |
| API 429 | Lower `--workers` (run_test) / `--concurrency` (evaluate). |

---

## 9. Submission checklist

```text
Group-X.zip
├── test_prediction.jsonl
├── code/
│   ├── README.md
│   ├── requirements.txt
│   ├── .env.example
│   ├── config.py
│   ├── doctor.py
│   ├── data/
│   ├── index/
│   ├── tools/
│   ├── agent/
│   ├── skills/
│   ├── ui/
│   ├── run_test.py
│   └── evaluate.py
├── Group-X-Report.pdf
└── Group-X-Video.mp4
```

- [ ] 100 test records, six required fields, produced by `run_test.py`.
- [ ] No `.env`, API keys, `data/`, `index/`, `models/`, or `*.db` in the zip.
- [ ] `README.md` explains setup, data, indexes, reproduction, and API key.
- [ ] Report and video included.

---

## 10. Design notes

- **BM25 remains chunk-level** to satisfy the sparse-retrieval requirement.
- **Dense is document-level**: a deliberate speed/accuracy trade-off; BM25
  chunk ranks and in-document search compensate for long-document dilution.
- **RRF is untrained**; no relevance labels are used.
- **Snippets** are capped at 160 characters; `document_reader` returns at most
  4000 characters and never inserts the full document into the chat.
- **Efficiency guards**: default retrieval `k=4`, at most 8 agent-issued tool
  calls, and a document-reader excerpt of 1000 characters.
- **Evaluation** is normalized exact match: lowercase, remove `a/an/the`,
  remove punctuation, collapse whitespace.
- All code and the three skills are original work for COMP5423.

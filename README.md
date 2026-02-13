# Monitor Brand — Sentiment & Reasons

Web app for **sentiment analysis** and **zero-shot reason extraction** on text columns in CSV. Built with FastAPI, Transformers (Hugging Face), and PyTorch.

## Features

- **Sentiment**: NEGATIVE / POSITIVE per row (fine-tuned sentiment model).
- **Reasons**: Zero-shot labels for negative texts only (e.g. delivery delay, poor quality) to keep inference fast on CPU.
- **Dataset converter**: CSV → Excel / CSV / Parquet.
- **Download**: Optional CSV export with labels and reasons.

## Tech stack

- **Backend**: FastAPI, Python 3.11+
- **ML**: PyTorch, Transformers (sentiment + zero-shot)
- **Data**: pandas, openpyxl, pyarrow



### Production env vars

| Variable         | Default              | Description                    |
|------------------|----------------------|--------------------------------|
| `MAX_FILE_SIZE_MB` | `50`               | Max upload size (MB)          |
| `MAX_ROWS`       | `10000`              | Max rows per analysis          |
| `DOWNLOAD_DIR`   | `/tmp`               | Directory for analysis CSVs   |

Example:

```bash
export BASE_URL=https://your-domain.com
export MAX_ROWS=5000
uvicorn front_logic:api --host 0.0.0.0 --port 8000
```

## API

- `GET /` — Web UI.
- `GET /health` — Health check (returns `{"status": "ok"}`).
- `POST /check` — Sentiment + reasons: form fields `file`, `column` (default `text`), `download_data` (bool).
- `GET /download/{filename}` — Download analysis CSV (filename format: `analysis_<hex>.csv`).
- `POST /convert` — Convert CSV to Excel/CSV/Parquet: form fields `file`, `file_type` (`excel` / `csv` / `parquet`).

## Model

- **Sentiment**: local `sentiment_model/` (must exist; prepare once via your training/export pipeline).
- **Reasons**: `facebook/bart-large-mnli` (downloaded on first run).

Reasons are computed only for **negative** texts to keep latency acceptable on CPU (e.g. Mac without GPU).

## Project layout

```
monitor_brand/
  front_logic.py      # FastAPI app
  requirements.txt
  Dockerfile
  .dockerignore
  templates/
    index.html
  sentiment_model/   
```



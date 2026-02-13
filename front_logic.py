import io
import logging
import os
import re
import uuid
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from transformers import pipeline

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "50"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "10000"))
DOWNLOAD_DIR = Path(os.environ.get("DOWNLOAD_DIR", "/tmp"))

DOWNLOAD_FILENAME_RE = re.compile(r"^analysis_[a-f0-9]{32}\.csv$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

api = FastAPI(title="Monitor Brand — Sentiment & Reasons", version="1.0.0")
templates = Jinja2Templates(directory="templates")

pipe = pipeline(
    "sentiment-analysis",
    model = "distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
)

negative_labels = [
    "delivery delay",
    "delivery failure",
    "wrong item delivered",
    "damaged item",
    "expired product",
    "poor product quality",
    "courier rude behavior",
    "customer support issue",
    "refund problem",
    "payment issue",
    "app or website problem",
]

reason_pipe = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@api.get("/health")
def health():
    return {"status": "ok"}


@api.get("/", response_class=HTMLResponse)
def frontend(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def read_to_df(file: UploadFile) -> pd.DataFrame:
    filename = (file.filename or "").lower()
    content = file.file.read()

    try:
        if filename.endswith(".csv"):
            return pd.read_csv(io.StringIO(content.decode("utf-8")))
        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(content))
        if filename.endswith(".json"):
            return pd.read_json(io.BytesIO(content))
        raise HTTPException(status_code=400, detail="File type not supported.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("read_to_df failed: %s", e)
        raise HTTPException(status_code=400, detail=f"File type not supported: {str(e)}") from e


@api.post("/check")
def check_data(
    file: UploadFile = File(...),
    column: str = Form("text"),
    download_data: bool = Form(False),
):
    # Limit upload size
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large (max {MAX_FILE_SIZE_MB} MB).",
        )

    df = read_to_df(file)

    if column not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found")

    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many rows (max {MAX_ROWS}).",
        )

    texts = df[column].astype(str).tolist()

    with torch.no_grad():
        sentiments = pipe(
            texts,
            batch_size=32,
            truncation=True,
            max_length=256,
        )

    values = pd.DataFrame(sentiments)
    negative_texts = [t for t, s in zip(texts, sentiments) if s["label"] == "NEGATIVE"]
    negative_reasons = []

    if negative_texts:
        with torch.no_grad():
            neg_res = reason_pipe(
                negative_texts,
                candidate_labels=negative_labels,
                batch_size=8,
                truncation=True,
                max_length=256,
            )
        negative_reasons = [n["labels"][0] for n in neg_res]

    label_meaning = []
    neg_i = 0
    for sent in sentiments:
        if sent["label"] == "NEGATIVE":
            label_meaning.append(negative_reasons[neg_i])
            neg_i += 1
        else:
            label_meaning.append(None)

    res = values["label"].value_counts(normalize=True)
    num_of_ratings = values["label"].value_counts().sort_values(ascending=False)
    confidence = values.groupby("label")["score"].mean()
    std_confidence = values.groupby("label")["score"].std()
    reasons = (
        pd.Series([r for r in label_meaning if r is not None])
        .value_counts(normalize=True)
        .sort_values(ascending=False)
    )

    final_df = pd.concat([df, values], axis=1)
    final_df["reason"] = label_meaning

    def to_python(d: dict):
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}

    response = {
        "rating": to_python(res.to_dict()),
        "number_of_ratings": to_python(num_of_ratings.to_dict()),
        "confidence": to_python(confidence.to_dict()),
        "std_confidence": to_python(std_confidence.to_dict()),
        "number_of_samples": int(df.shape[0]),
        "reasons": to_python(reasons.to_dict()),
    }

    if download_data:
        safe_name = f"analysis_{uuid.uuid4().hex}.csv"
        path = DOWNLOAD_DIR / safe_name
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.StringIO()
        final_df.to_csv(buffer, index=False)
        path.write_text(buffer.getvalue(), encoding="utf-8")
        response["download_url"] = f"{BASE_URL.rstrip('/')}/download/{safe_name}"

    logger.info("check: rows=%s", len(df))
    return response


@api.get("/download/{filename}")
def download(filename: str):
    if not DOWNLOAD_FILENAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = DOWNLOAD_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="text/csv", filename=filename)


@api.post("/convert")
async def convert_data(
    file: UploadFile = File(...),
    file_type: str = Form("excel"),
):
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        logger.exception("convert read failed: %s", e)
        raise HTTPException(status_code=400, detail="Cannot read file (expected CSV).") from e

    if file_type == "excel":
        output = io.BytesIO()
        df.to_excel(output, index=False)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        out_filename = "converted.xlsx"
    elif file_type == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        output = io.BytesIO(buf.getvalue().encode("utf-8"))
        media_type = "text/csv"
        out_filename = "converted.csv"
    elif file_type == "parquet":
        output = io.BytesIO()
        df.to_parquet(output, index=False)
        media_type = "application/octet-stream"
        out_filename = "converted.parquet"
    else:
        raise HTTPException(status_code=400, detail="Unsupported format.")

    output.seek(0)
    return StreamingResponse(
        output,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={out_filename}"},
    )

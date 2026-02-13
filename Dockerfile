FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY templates/ ./templates/
COPY sentiment_model/ ./sentiment_model/
COPY front_logic.py .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "front_logic:api", "--host", "0.0.0.0", "--port", "8000"]

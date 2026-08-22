# Minimal Flask front-end for the housing API. Unlike trainer/api, this
# image doesn't need the housing_ml package or the repo root -- it's a
# standalone HTTP client, so its build context is just this directory.
FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY templates ./templates

CMD ["python", "app.py"]

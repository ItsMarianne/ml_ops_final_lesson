# API image: serves predictions from whichever model is registered under
# the alias in MODEL_ALIAS (see api/main.py).
#
# Plain pip only -- see trainer/Dockerfile's header comment for why uv isn't
# used here even though it manages the lockfile this requirements.txt was
# exported from.
#
# Build context is the repo root (see docker-compose.yml's `context: .`)
# because this image needs files from src/ as well as api/.
FROM python:3.11-slim
WORKDIR /app

COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY api ./api
COPY config ./config
RUN pip install --no-cache-dir --no-deps -e .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

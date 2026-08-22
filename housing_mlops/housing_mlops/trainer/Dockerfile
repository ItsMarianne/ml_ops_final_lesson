# Trainer image: runs the training pipeline once (src/housing_ml/train.py)
# and, on demand, the feature-selection analysis (scripts/select_features.py).
#
# Plain pip only -- uv is a local/dev-time tool used to resolve and lock
# dependencies (see uv.lock at the repo root); it has no role at build or
# run time. trainer/requirements.txt below is a pinned export of that lock
# (`make lock` regenerates it), so this image is still fully reproducible
# without needing uv installed in it.
#
# Build context is the repo root (see docker-compose.yml's `context: .`)
# because this image needs files from src/, scripts/, and config/, not just
# this directory.
FROM python:3.11-slim
WORKDIR /app

# Dependencies first, as their own layer -- Docker only re-runs this slow
# step when requirements.txt changes, not on every code edit.
COPY trainer/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Now the actual source. pyproject.toml + src/ are the local `housing_ml`
# package; `pip install --no-deps -e .` just registers it against the
# dependencies already installed above (no re-resolving, so this stays fast
# even though it reruns on every code change).
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY config ./config
RUN pip install --no-cache-dir --no-deps -e .

CMD ["python", "-m", "housing_ml.train"]

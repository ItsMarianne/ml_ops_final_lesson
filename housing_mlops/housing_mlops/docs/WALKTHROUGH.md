# Walkthrough for first-time readers

This is a guided tour, not a reference — [README.md](../README.md) is the
reference (what each piece does and why it was built that way). This
document is for orienting yourself the *first* time you open this project:
what the jargon means, and which file to read first, second, third.

## What this project actually is

It predicts California housing prices from 8 numbers (median income, house
age, etc.). There are three moving pieces:

1. **A trainer** that reads a CSV, tries a few machine learning models, and
   picks the best one.
2. **A registry** (MLflow) that remembers every model you've ever trained
   and which one is the "current" one to use.
3. **An API + a web form** that let you send in 8 numbers and get a price
   prediction back, using whichever model the registry says is current.

Everything runs in Docker containers so "it works on my machine" also means
"it works on your machine" — you don't need Python, scikit-learn, or MLflow
installed directly, Docker handles that.

## Concepts you'll see everywhere (plain-language glossary)

- **Docker / container / image**: a container is a small, isolated
  computer-within-your-computer that runs one program with its own copy of
  Python and libraries, so it can't be broken by whatever else is installed
  on your real machine. An *image* is the recipe (a `Dockerfile`); a
  *container* is that recipe actually running. `docker-compose.yml` starts
  several containers together and lets them talk to each other by name
  (e.g. one container can reach another at `http://mlflow:5000`).
- **Virtual environment / `uv`**: a private folder of Python packages for
  just this project, so installing something here doesn't affect other
  Python projects on your machine. `uv` is the tool that manages it; you
  only need it if you're running code *outside* Docker (tests, linting).
- **Train/test split**: you don't judge a model on the same data it
  studied — that's like grading an exam using the answer key the student
  copied from. So the data is split: the model learns from ~80% (the
  *training set*) and gets scored on the other ~20% it never saw (the
  *test set*).
- **Cross-validation / GridSearchCV**: rather than trying one set of model
  settings once, GridSearchCV tries several combinations (e.g. different
  "how complex should this model be" settings) and — using cross-validation
  (repeatedly re-splitting the training set to test each combination
  fairly) — picks whichever combination generalizes best.
- **RMSE / MAE / R²**: three ways to score "how wrong were the
  predictions." RMSE and MAE are in the same units as the price (lower is
  better); R² is a 0–1 "how much of the variation did the model explain"
  score (higher is better, 1.0 is perfect).
- **Feature / feature selection**: a "feature" is one input column (e.g.
  `MedInc`). Feature selection is deciding which columns are actually worth
  feeding the model — some columns barely affect the prediction and just
  add noise.
- **MLflow tracking / registry / alias**: MLflow is a separate service
  (its own Docker container) that remembers every training run — what
  settings were used, how accurate it was, and the trained model file
  itself. The *registry* is the subset of runs you've explicitly promoted
  to be usable; an *alias* (like `prod`) is a movable label pointing at one
  specific registered version, so "the model currently in production" can
  change without anyone needing to know a version number.
- **FastAPI / Flask**: two different Python web frameworks. FastAPI here
  serves a JSON API (`/predict`); Flask here serves an HTML form for
  humans. The Flask app doesn't do any prediction itself — it just calls
  the FastAPI service over HTTP, the same way your browser would.
- **YAML config**: `config/*.yaml` files hold settings (names, numbers,
  lists) that Python code reads at startup, so you can change *what* the
  system does without editing *how* it does it (the code).

## Suggested reading order

Read in this order — each step assumes only what came before it.

1. **[`config/train.yaml`](../config/train.yaml)** — no code, just settings.
   Skim it to see what's configurable: names, the train/test split ratio,
   which models get tried, promotion rules. This is "the knobs."
2. **[`config/features.yaml`](../config/features.yaml)** — which of the 8
   input columns actually get used, and (once you've run the analysis) why
   any were dropped.
3. **[`src/housing_ml/config.py`](../src/housing_ml/config.py)** — how the
   two files above get loaded into Python and validated. Short, and it'll
   make every other file's `config.something` make sense.
4. **[`src/housing_ml/data.py`](../src/housing_ml/data.py)** — loads the
   CSV, checks it looks sane, splits it into train/test. The most
   "beginner ML" file here — read it slowly if train/test splitting is new.
5. **[`src/housing_ml/features.py`](../src/housing_ml/features.py)** — very
   short: turns `features.yaml`'s list into the actual columns used.
6. **[`src/housing_ml/models.py`](../src/housing_ml/models.py)** — defines
   the 3 candidate model types and how each is wrapped into a
   scale-then-predict pipeline.
7. **[`src/housing_ml/evaluate.py`](../src/housing_ml/evaluate.py)** —
   RMSE/MAE/R² computed from predictions. One small function, easy to
   fully understand.
8. **[`src/housing_ml/registry.py`](../src/housing_ml/registry.py)** — the
   MLflow-facing code: logging a run, and the "only promote if better"
   safety check. The most conceptually dense file — come back to it after
   reading `train.py`, if it doesn't click the first time.
9. **[`src/housing_ml/train.py`](../src/housing_ml/train.py)** — ties
   everything above together: load data → try each model → score it → log
   it → promote the best one. Read this *after* the files above, not
   before — it'll read like a summary instead of a mystery.
10. **[`src/housing_ml/prediction_log.py`](../src/housing_ml/prediction_log.py)**
    — tiny: writes one line to a file per prediction served.
11. **[`api/main.py`](../api/main.py)** — the FastAPI service. By now
    `TrainConfig` and `log_prediction` are familiar imports, not mysteries.
12. **[`webapp/app.py`](../webapp/app.py)** — the Flask form. Notice it
    never imports anything from `housing_ml` — it only ever talks to the
    API over plain HTTP, the same as `curl` would.
13. **[`scripts/select_features.py`](../scripts/select_features.py)** and
    **[`scripts/smoke_test.py`](../scripts/smoke_test.py)** — standalone
    tools, run on demand rather than as part of every training/serving run.
14. **[`tests/`](../tests/)** — one test file per module above; reading a
    module's test file alongside it is often the fastest way to see
    "oh, *that's* what this function is supposed to do."
15. **[`Makefile`](../Makefile)** and
    **[`docker-compose.yml`](../docker-compose.yml)** — last, once you know
    what each piece does, these show how they're actually run and wired
    together as containers.

## Try it yourself

```bash
make sync          # set up a local Python environment (for tests/linting)
make up             # start everything: mlflow, train a model, start api+webapp
```

Then open http://localhost:5001 in a browser and submit the form — that
request travels webapp → api → the registered model → back. Open
http://localhost:5000 alongside it to see the training runs MLflow
recorded. `make help` lists every other command available.

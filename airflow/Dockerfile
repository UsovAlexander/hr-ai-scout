FROM apache/airflow:2.9.1-python3.11

# Все зависимости устанавливаются при сборке образа — один раз,
# а не при каждом старте контейнера.

# ── 0. Системные зависимости ───────────────────────────────────────────────────
# libgomp1 нужен пакету implicit (ALS) — OpenMP runtime
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# ── 1. Основные зависимости ────────────────────────────────────────────────────
COPY requirements_airflow.txt /tmp/requirements_airflow.txt
RUN pip install --no-cache-dir -r /tmp/requirements_airflow.txt

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
# На arm64 нет официального CPU-only wheel: ставим полный torch.
# NVIDIA-пакеты войдут в образ (~1.5 ГБ), но без GPU torch автоматически
# использует CPU. Образ собирается один раз и кешируется локально.
RUN pip install --no-cache-dir torch==2.11.0

# ── 3. Transformers ────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir transformers==5.7.0

# ── 4. NLTK data (stopwords + тэггер) ─────────────────────────────────────────
RUN python -c "\
import nltk; \
nltk.download('stopwords', quiet=True); \
nltk.download('averaged_perceptron_tagger_ru', quiet=True); \
nltk.download('wordnet', quiet=True)"

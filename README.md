# HR-AI Scout: Веб-сервис для анализа и скоринга кандидатов

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Machine-Learning-orange?logo=scikitlearn" alt="ML">
  <img src="https://img.shields.io/badge/Deep-Learning-red?logo=pytorch" alt="DL">
  <img src="https://img.shields.io/badge/FastAPI-green?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow" alt="Status">
</p>

## О проекте

**HR-AI Scout** — интеллектуальный сервис автоматического скоринга резюме. По описанию вакансии возвращает топ-10 наиболее подходящих кандидатов из базы, ранжированных моделью машинного обучения.

Система собирает резюме с Habr Career и вакансии с HH.ru / SuperJob, строит 17 признаков (категориальные, числовые, текстовое сходство TF-IDF и LaBSE), обучает CatBoost-пайплайн и обслуживает запросы через FastAPI REST API.

---

## Технологический стек

| Категория | Технологии |
| :--- | :--- |
| **Бэкенд & API** | Python 3.11, FastAPI, SQLAlchemy + aiosqlite |
| **ML-пайплайн** | CatBoost, Scikit-learn, implicit (ALS) |
| **Текстовые признаки** | TF-IDF + pymorphy3, LaBSE (`cointegrated/LaBSE-en-ru`) |
| **Фреймворки DL** | PyTorch, Transformers (HuggingFace) |
| **Фронтенд** | Streamlit |
| **Хранилище** | ClickHouse (резюме, вакансии, эмбеддинги), SQLite (история запросов) |
| **Парсинг** | BeautifulSoup, Requests |
| **Оркестрация** | Apache Airflow 2.9 (LocalExecutor, Docker Compose) |
| **Эксперименты** | MLflow |

---

## Установка и запуск

### 1. Клонирование и зависимости

```bash
git clone https://github.com/UsovAlexander/hr-ai-scout.git
cd hr-ai-scout

python3 -m venv venv_hr_ai_scout
source ./venv_hr_ai_scout/bin/activate
pip install -r requirements.txt
```

### 2. Переменные окружения

Создайте `.env` в корне проекта:

```env
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your_password
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=9000
CLICKHOUSE_DATABASE=default
```

### 3. ClickHouse

```bash
docker run -d \
  --name clickhouse \
  --restart always \
  -p 9000:9000 \
  -p 8123:8123 \
  -e CLICKHOUSE_PASSWORD=your_password \
  clickhouse/clickhouse-server:latest
```

> Веб-интерфейс — `http://localhost:8123`, нативный клиент (TCP) — порт `9000`.

Создайте таблицы:

```bash
docker exec -it clickhouse clickhouse-client --password your_password
```

```sql
CREATE TABLE IF NOT EXISTS hh_resumes (
    id                             String,
    title                          Nullable(String),
    url                            Nullable(String),
    specialization                 Array(Nullable(String)),
    last_company                   Nullable(String),
    last_position                  Nullable(String),
    last_experience_description    Nullable(String),
    last_company_experience_period Nullable(String),
    skills                         Array(String),
    education                      Array(String),
    courses                        Array(String),
    salary                         Nullable(String),
    age                            Nullable(Int64),
    total_experience               Nullable(String),
    experience_months              Nullable(Int64),
    location                       Nullable(String),
    gender                         Nullable(String),
    applicant_status               Nullable(String),
    search_query                   Nullable(String),
    source                         Nullable(String),
    parsed_date                    DateTime
) ENGINE = MergeTree()
ORDER BY (parsed_date, id);

CREATE TABLE IF NOT EXISTS hh_vacancies (
    id               String,
    name             Nullable(String),
    area             Nullable(String),
    url              Nullable(String),
    alternate_url    Nullable(String),
    requirement      Nullable(String),
    responsibility   Nullable(String),
    description      Nullable(String),
    employer         Nullable(String),
    experience       Nullable(String),
    employment       Nullable(String),
    schedule         Nullable(String),
    published_at     Nullable(DateTime),
    created_at       Nullable(DateTime),
    salary_from      Nullable(Float64),
    salary_to        Nullable(Float64),
    salary_currency  Nullable(String),
    salary_gross     Nullable(Bool),
    search_query     Nullable(String),
    area_id          Nullable(Int64),
    source           Nullable(String),
    parsed_date      DateTime
) ENGINE = MergeTree()
ORDER BY (parsed_date, id);

CREATE TABLE IF NOT EXISTS resume_embeddings (
    resume_id    String,
    model_name   String,
    embedding    Array(Float32),
    created_at   DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (model_name, resume_id);

CREATE TABLE IF NOT EXISTS vacancy_embeddings (
    vacancy_id   String,
    model_name   String,
    embedding    Array(Float32),
    created_at   DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (model_name, vacancy_id);

CREATE TABLE IF NOT EXISTS recruiter_decisions (
    vacancy_id                          String,
    vacancy_area                        Nullable(String),
    vacancy_experience                  Nullable(String),
    vacancy_employment                  Nullable(String),
    vacancy_schedule                    Nullable(String),
    vacancy_description                 Nullable(String),
    resume_id                           String,
    resume_skills                       Nullable(String),
    resume_last_position                Nullable(String),
    resume_last_experience_description  Nullable(String),
    resume_last_company_experience_period Nullable(String),
    resume_salary                       Nullable(String),
    resume_age                          Nullable(Int64),
    resume_experience_months            Nullable(Int64),
    resume_location                     Nullable(String),
    resume_gender                       Nullable(String),
    resume_applicant_status             Nullable(String),
    target                              Int8,
    decided_at                          DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (decided_at, vacancy_id, resume_id);
```

### 4. Apache Airflow

> Требование: ClickHouse должен быть запущен до старта Airflow.

```bash
cd airflow
docker compose up airflow-init   # первый запуск: инициализация БД и admin-пользователя
docker compose up -d
```

Веб-интерфейс: **http://localhost:8080** — логин `admin`, пароль `admin`.

**Расписание DAG-ов:**

| DAG | Расписание | Описание |
| :--- | :--- | :--- |
| `habr_resume_daily` | пн–сб, 03:00 МСК | Сбор резюме с Habr Career |
| `hh_vacancy_daily` | по триггеру | Вакансии с HH.ru после сбора резюме |
| `sj_vacancy_daily` | каждый день, 04:00 МСК | Вакансии с SuperJob |
| `labse_embeddings` | по триггеру | LaBSE-эмбеддинги новых резюме и вакансий |
| `retrain_pipeline` | пн–сб, 06:00 МСК | Дообучение CatBoost на решениях рекрутеров |

Активация DAG-ов (по умолчанию приостановлены):

```bash
cd airflow
docker compose exec airflow-scheduler airflow dags unpause habr_resume_daily
docker compose exec airflow-scheduler airflow dags unpause sj_vacancy_daily
docker compose exec airflow-scheduler airflow dags unpause retrain_pipeline
```

**Остановка:**
```bash
docker compose down
```

### 5. MLflow

```bash
chmod +x mlflow_run.sh
./mlflow_run.sh start      # foreground, порт 5000
./mlflow_run.sh start-d    # background
./mlflow_run.sh stop
./mlflow_run.sh status
```

### 6. FastAPI backend

Поместите файл `df_resumes.csv` в папку `app/` (тестовый набор: https://disk.yandex.kz/d/Ybs4dTfwh1me5g).

```bash
cd app
uvicorn main:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
```

JWT-аутентификация: POST `/token` с `username=test&password=test` → bearer-токен (30 мин).

### 7. Streamlit frontend

```bash
streamlit run steamlit_app/streamlit_app.py
# http://localhost:8501
```

---

## Архитектура

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Источники данных                            │
│   Habr Career (резюме)          HH.ru / SuperJob (вакансии)        │
└────────────┬────────────────────────────────┬───────────────────────┘
             │                                │
             ▼                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Apache Airflow (Docker)                           │
│                                                                     │
│  habr_resume_daily ──► labse_embeddings ──► retrain_pipeline        │
│       03:00 МСК          (по триггеру)        06:00 МСК             │
│                                                                     │
│  hh_vacancy_daily ──────────────────────────────────────────────►  │
│  sj_vacancy_daily  (по триггеру / 04:00 МСК)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │ clickhouse-driver
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ClickHouse                                  │
│                                                                     │
│  hh_resumes          hh_vacancies        recruiter_decisions        │
│  resume_embeddings   vacancy_embeddings                             │
│        (LaBSE cointegrated/LaBSE-en-ru, 768-dim)                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ML-пайплайн                                   │
│                                                                     │
│  17 признаков:                                                      │
│  ├── Категориальные: area, experience, employment, schedule,        │
│  │                   location, gender, applicant_status             │
│  ├── Числовые: salary, age, experience_months,                      │
│  │             last_company_exp_months, skill_count,                │
│  │             last_position_match, location_matching               │
│  ├── TF-IDF cosine similarity  (pymorphy3 лемматизация, рус/англ)  │
│  ├── LaBSE cosine similarity   (семантическое сходство, рус/англ)  │
│  └── ALS score                 (коллаборативная фильтрация)         │
│                                                                     │
│  Модель: CatBoost + OrdinalEncoder (pipeline_cb_als_sim_labse.pkl) │
│  Дообучение: recruiter_decisions → retrain_pipeline DAG             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI  (app/main.py)                            │
│                                                                     │
│  POST /token          — JWT-аутентификация                          │
│  POST /forward        — топ-10 резюме по вакансии                   │
│  GET  /history        — история предсказаний  [admin]               │
│  GET  /stats          — агрегированная статистика  [admin]          │
│  DELETE /history      — очистка истории  [admin]                    │
│  GET  /health         — health check                                │
│                                                                     │
│  История → SQLite (hr_scout_history.db, async SQLAlchemy)           │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ HTTP (localhost:8000)
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Streamlit  (steamlit_app/)                          │
│                                                                     │
│  Режим одного резюме — ввод параметров вручную, скор + топ-10       │
│  Пакетный режим      — загрузка CSV, массовое предсказание,         │
│                        сводная статистика, экспорт                  │
└─────────────────────────────────────────────────────────────────────┘
```

# HR-AI Scout: Веб-сервис для анализа и скоринга кандидатов

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Machine-Learning-orange?logo=scikitlearn" alt="ML">
  <img src="https://img.shields.io/badge/Deep-Learning-red?logo=pytorch" alt="DL">
  <img src="https://img.shields.io/badge/Web-Framework-green?logo=fastapi" alt="Web Framework">
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow" alt="Status">
</p>

## 🎯 О проекте

**HR-AI Scout** — это интеллектуальный веб-сервис, предназначенный для автоматизации процесса первичного отбора кандидатов. Система анализирует резюме (в форматах PDF, DOCX) и агрегированные данные из различных источников (например, LinkedIn, HH.ru, superjob.ru), чтобы предсказать успешность кандидата на целевой позиции.

**Ключевая цель:** Сократить время и ресурсы HR-специалистов на рутинный скрининг, предоставив им инструмент для объективного и основанного на данных первичного отбора.

---

## 🚀 Возможности

*   **📄 Автопарсинг резюме:** Извлечение ключевой информации из загружаемых файлов.
*   **🔍 Агрегация данных:** Получение и анализ дополнительных данных из открытых источников.
*   **🧠 ML/DL-скоринг:** Оценка кандидата с помощью ансамбля машинного и глубокого обучения.
*   **📊 Интуитивный Dashboard:** Визуализация результатов скоринга, ключевых навыков и сравнение кандидатов.

---

## 🗺 Дорожная карта проекта (Этапы разработки)

### Этап 1: Инициация и планирование
*   **Знакомство с куратором, постановка и уточнение задачи.**
*   Формирование требований (SRS).
*   Разработка детального плана разработки и прототипа на год.

### Этап 2: Исследование данных
*   **Сбор данных и EDA.**
*   Поиск и сбор датасетов (резюме, вакансии, успешность кандидатов).
*   Проведение разведочного анализа данных (EDA), очистка и первичная обработка.

### Этап 3: Базовые модели машинного обучения
*   **Линейные модели ML.**
*   Разработка и обучение базовых моделей (Линейная/Логистическая регрессия).
*   Создание бейзлайна для оценки эффективности последующих моделей.

### Этап 4: Усложнение моделирования
*   **Нелинейные модели ML, feature-engineering и выбор лучшего ML-решения.**
*   Активный feature engineering и отбор признаков.
*   Эксперименты с нелинейными моделями (Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost)).
*   Сравнение, валидация и выбор чемпиона среди ML-моделей.

### Этап 5: Воплощение в жизнь
*   **Создание сервиса с имплементацией лучшего ML-решения.**
*   Разработка backend-логики на FastAPI.
*   Создание простого фронтенда (Streamlit) для взаимодействия с моделью.
*   Деплой первой рабочей версии сервиса.

### Этап 6: Глубокое обучение
*   **DL-модели.**
*   Исследование и применение моделей глубокого обучения (BERT, Transformer) для анализа текстов резюме.
*   Сравнение качества DL и чемпионской ML-модели.

### Этап 7: Финальный рывок и оптимизация
*   **Доработка сервиса и финальный выбор модели.**
*   Улучшение сервисной части по обратной связи от команды (UI/UX, производительность, документация).
*   Гипертюнинг DL-моделей и финальное сравнение всех решений.
*   **Выбор лучшего решения overall** и его финальная интеграция в продакшен-сервис.

---

## 🛠 Технологический стек

| Категория | Технологии |
| :--- | :--- |
| **Бэкенд & API** | Python, FastAPI|
| **Машинное обучение** | Scikit-learn, Pandas, NumPy, SciPy |
| **Глубокое обучение** | PyTorch / TensorFlow, Transformers, BERT |
| **Обработка текста** | NLTK, SpaCy|
| **Фронтенд** | Streamlit |
| **Базы данных** | PostgreSQL, ClickHouse |
| **Парсинг данных** | BeautifulSoup, Requests |
| **Оркестрация** | Apache Airflow 2.9 |
| **Деплой & DevOps** | Docker, Docker Compose |

---

## 📦 Установка и запуск

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/UsovAlexander/hr-ai-scout.git
    cd hr-ai-scout
    ```

2.  **Создайте виртуальное окружение и установите зависимости:**
    ```bash
    python3 -m venv venv_hr_ai_scout

    source ./venv_hr_ai_scout/bin/activate
    
    pip install -r requirements.txt
    ```

3.  **Настройте переменные окружения:**

    Создайте файл `.env` в корне проекта:
    ```env
    CLICKHOUSE_USER=default
    CLICKHOUSE_PASSWORD=your_password
    CLICKHOUSE_HOST=localhost
    CLICKHOUSE_PORT=9000
    CLICKHOUSE_DATABASE=default
    ```

4.  **Для работы с MLFLOW используйте следующие команды**

    Сделайте файл исполняемым
    ```
    chmod +x mlflow_run.sh
    ```
    Запуск
    ```
    ./mlflow_run.sh start
    ```

    Запуск в фоновом режиме
    ```
    ./mlflow_run.sh start-d
    ```

    Остановка  
    ```
    ./mlflow_run.sh stop
    ```
    Принудительная остановка всех процессов mlflow
    ```
    ./mlflow_run.sh stop-all
    ```
    Статус
    ```
    ./mlflow_run.sh status
    ```

5. **Развёртывание ClickHouse**

ClickHouse используется как основное хранилище резюме и вакансий. Проще всего запустить его через Docker:

```bash
docker run -d \
  --name clickhouse \
  --restart always \
  -p 9000:9000 \
  -p 8123:8123 \
  -e CLICKHOUSE_PASSWORD=your_password \
  clickhouse/clickhouse-server:latest
```

> Веб-интерфейс (HTTP) — `http://localhost:8123`, нативный клиент (TCP) — порт `9000`.

После запуска создайте таблицы. Подключитесь к ClickHouse:

```bash
docker exec -it clickhouse clickhouse-client --password your_password
```

И выполните DDL:

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
```

---

6. **Развёртывание Apache Airflow и настройка сбора данных**

Парсинг резюме и вакансий с HH.ru автоматизирован через Apache Airflow. Все данные сохраняются в ClickHouse по расписанию.

> **Требование:** ClickHouse должен быть запущен до старта Airflow (шаг 5).

**Запуск:**
```bash
cd airflow
docker compose up airflow-init   # первый запуск: инициализация БД и создание admin-пользователя
docker compose up -d             # запуск планировщика и веб-сервера
```

Веб-интерфейс: **http://localhost:8080** — логин `admin`, пароль `admin`.

В процессе `airflow-init` автоматически создаётся коннектор `clickhouse_default`, ссылающийся на хост-машину (`host.docker.internal`). Убедитесь, что параметры совпадают с вашим `.env`.

**Расписание DAG-ов:**

| DAG | Расписание | Описание |
| :--- | :--- | :--- |
| `resume_daily` | пн–сб в 01:00 МСК | 5 стр. × 20 резюме по каждой IT-профессии |
| `resume_weekly` | сб в 23:00 МСК | 250 стр. × 20 резюме — полное обновление базы |
| `vacancy_daily` | по триггеру | 5 стр. × 20 вакансий, запускается автоматически после `resume_daily` / `resume_weekly` |

Резюме и вакансии парсятся **последовательно**: сначала завершается сбор резюме, затем автоматически стартует сбор вакансий — это снижает нагрузку на HH.ru и уменьшает риск блокировки IP.

Поле `parsed_date` сохраняется в часовом поясе **Europe/Moscow**.

**Активация DAG-ов** (по умолчанию DAG-и приостановлены):
```bash
docker compose exec airflow-scheduler airflow dags unpause resume_daily
docker compose exec airflow-scheduler airflow dags unpause resume_weekly
docker compose exec airflow-scheduler airflow dags unpause vacancy_daily
```

**Остановка Airflow:**
```bash
docker compose down
```

---

7. **Для запуска сервиса с FASTAPI используйте следующие команды**

Перед запуском сервиса необходимо подготовить данные с резюме.

### Подготовка данных с резюме

Сервис использует датасет `df_resumes`, содержащий резюме, с которыми будет сравниваться текст и параметры вакансии.

Тестовый набор `df_resumes` можно скачать по ссылке:  
https://disk.yandex.kz/d/Ybs4dTfwh1me5g

После скачивания:
- распакуйте архив (если применимо);
- поместите файл с `df_resumes` в "app/"


Перейдите в папку с сервисом FASTAPI
```
cd app
```

Запустите сервис 
```
uvicorn main:app --reload
```


---

## 🏗 Архитектура


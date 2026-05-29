"""
DAG 3: LaBSE эмбеддинги резюме и вакансий.

Модель: cointegrated/LaBSE-en-ru — двуязычная (EN/RU) BERT-модель,
предобученная для семантического сходства.

Кодирование идентично ноутбуку (BERT_TFIDF_Experiments.ipynb, In[27]):
  mean pooling по токенам, взвешенный attention_mask, L2-нормализация.
Нормализованные векторы → косинусное сходство = скалярное произведение.

Логика DAG:
  1. Проверяет наличие новых резюме в hh_resumes И новых вакансий в hh_vacancies
     без LaBSE-эмбеддингов. Если нет ничего нового — encode-таски завершаются штатно.
  2. Загружает модель, кодирует last_experience_description новых резюме → resume_embeddings.
  3. Загружает модель, кодирует description новых вакансий → vacancy_embeddings.
     Вакансии могут добавляться рекрутером через Streamlit — DAG обрабатывает их
     независимо от того, были ли новые резюме.

Запускается вручную или триггером из resume_daily / resume_weekly.
"""
import os
import warnings

import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

warnings.simplefilter('ignore', FutureWarning)

# ── Константы ─────────────────────────────────────────────────────────────────
MODEL_NAME = 'cointegrated/LaBSE-en-ru'
BATCH_SIZE = 64
MAX_LENGTH = 512
CHUNK_SIZE = 5_000   # резюме/вакансий за один мегабатч с проверкой идемпотентности

MSK = pendulum.timezone('Europe/Moscow')
DEFAULT_ARGS = {
    'owner': 'hr-ai-scout',
    'retries': 1,
    'retry_delay': pendulum.duration(minutes=5),
}


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _get_clickhouse_client():
    from clickhouse_driver import Client
    return Client(
        host=os.getenv('CLICKHOUSE_HOST', 'localhost'),
        port=int(os.getenv('CLICKHOUSE_PORT', 9000)),
        user=os.getenv('CLICKHOUSE_USER', 'default'),
        password=os.getenv('CLICKHOUSE_PASSWORD', ''),
        database=os.getenv('CLICKHOUSE_DATABASE', 'default'),
    )


def _get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    # MPS недоступен в Linux Docker-контейнере
    return torch.device('cpu')


def _load_model(device):
    """Загружает токенайзер и модель LaBSE на указанное устройство."""
    from transformers import AutoTokenizer, AutoModel
    print(f"Загружаю {MODEL_NAME} на {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    return tokenizer, model


def _encode_texts(texts: list, tokenizer, model, device) -> 'np.ndarray':
    """
    Батчевое кодирование: mean pooling + L2-нормализация (из ноутбука, In[27]).
    Возвращает float32 массив (N, hidden_size).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            out = model(**encoded)

        token_emb = out.last_hidden_state                          # (B, T, H)
        mask = encoded['attention_mask'].unsqueeze(-1).float()     # (B, T, 1)
        pooled = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=1)
        all_embeddings.append(pooled.cpu().numpy())

        done = min(i + BATCH_SIZE, len(texts))
        if done % (BATCH_SIZE * 10) == 0 or done == len(texts):
            print(f"  Закодировано: {done}/{len(texts)} ({100 * done // len(texts)}%)")

    return np.vstack(all_embeddings).astype(np.float32)


def _get_missing_ids(ids_needed: list, table: str, id_col: str,
                     clickhouse) -> list:
    """Возвращает id из ids_needed, для которых нет эмбеддингов в ClickHouse."""
    if not ids_needed:
        return []
    str_ids = [str(i) for i in ids_needed]
    rows = clickhouse.execute(
        f"SELECT {id_col} FROM {table} "
        f"WHERE model_name = %(m)s AND {id_col} IN %(ids)s",
        {'m': MODEL_NAME, 'ids': str_ids},
    )
    existing = {row[0] for row in rows}
    missing = [i for i in str_ids if i not in existing]
    print(f"  {table}: {len(existing)} в кеше, {len(missing)} новых из {len(str_ids)}")
    return missing


def _save_embeddings(emb_map: dict, id_col: str, table: str, clickhouse):
    """Дописывает новые эмбеддинги в ClickHouse."""
    rows = [(str(k), MODEL_NAME, v.tolist()) for k, v in emb_map.items()]
    clickhouse.execute(
        f"INSERT INTO {table} ({id_col}, model_name, embedding) VALUES",
        rows,
    )
    print(f"  Сохранено {len(rows)} эмбеддингов → {table}")


# ── Задачи DAG ────────────────────────────────────────────────────────────────

def check_new_items(**context):
    """
    Проверяет наличие новых резюме и вакансий без LaBSE-эмбеддингов.
    Результаты записываются в XCom; encode-таски обрабатывают пустые списки самостоятельно.
    Вакансии могут добавляться рекрутером через Streamlit независимо от парсера.
    """
    ch = _get_clickhouse_client()

    resume_rows = ch.execute(
        """
        SELECT id FROM hh_resumes
        WHERE id NOT IN (
            SELECT resume_id FROM resume_embeddings WHERE model_name = %(m)s
        )
        AND last_experience_description IS NOT NULL
        AND last_experience_description != ''
        """,
        {'m': MODEL_NAME},
    )
    new_resume_ids = [r[0] for r in resume_rows]

    no_text_rows = ch.execute(
        """
        SELECT count() FROM hh_resumes
        WHERE id NOT IN (
            SELECT resume_id FROM resume_embeddings WHERE model_name = %(m)s
        )
        AND (last_experience_description IS NULL OR last_experience_description = '')
        """,
        {'m': MODEL_NAME},
    )
    print(f"Новых резюме без LaBSE-эмбеддингов: {len(new_resume_ids)} "
          f"(пропущено без описания: {no_text_rows[0][0]})")

    vacancy_rows = ch.execute(
        """
        SELECT id FROM hh_vacancies
        WHERE id NOT IN (
            SELECT vacancy_id FROM vacancy_embeddings WHERE model_name = %(m)s
        )
        AND description IS NOT NULL
        AND description != ''
        """,
        {'m': MODEL_NAME},
    )
    new_vacancy_ids = [r[0] for r in vacancy_rows]
    print(f"Новых вакансий без LaBSE-эмбеддингов: {len(new_vacancy_ids)}")

    ti = context['task_instance']
    ti.xcom_push(key='new_resume_ids', value=new_resume_ids)
    ti.xcom_push(key='new_vacancy_ids', value=new_vacancy_ids)

    return len(new_resume_ids) > 0 or len(new_vacancy_ids) > 0


def encode_resume_embeddings(**context):
    """
    Загружает LaBSE, кодирует новые резюме батчами, сохраняет в resume_embeddings.
    Идемпотентен: перепроверяет пропуски перед каждым мегабатчем CHUNK_SIZE.
    """
    ti = context['task_instance']
    new_ids = ti.xcom_pull(task_ids='check_new_items', key='new_resume_ids')
    if not new_ids:
        print("Нет новых резюме — пропускаю.")
        return

    device = _get_device()
    tokenizer, model = _load_model(device)
    ch = _get_clickhouse_client()
    total_saved = 0

    for start in range(0, len(new_ids), CHUNK_SIZE):
        chunk_ids = new_ids[start: start + CHUNK_SIZE]

        missing = _get_missing_ids(chunk_ids, 'resume_embeddings', 'resume_id', ch)
        if not missing:
            continue

        rows = ch.execute(
            "SELECT id, last_experience_description FROM hh_resumes "
            "WHERE id IN %(ids)s "
            "AND last_experience_description IS NOT NULL "
            "AND last_experience_description != ''",
            {'ids': missing},
        )
        if not rows:
            continue

        ids_batch = [r[0] for r in rows]
        texts = [str(r[1]) for r in rows]

        print(f"Кодирую {len(texts)} резюме (chunk {start // CHUNK_SIZE + 1})...")
        embeddings = _encode_texts(texts, tokenizer, model, device)

        emb_map = {rid: emb for rid, emb in zip(ids_batch, embeddings)}
        _save_embeddings(emb_map, 'resume_id', 'resume_embeddings', ch)
        total_saved += len(emb_map)

    print(f"=== LaBSE резюме завершены: добавлено {total_saved} записей ===")


def encode_vacancy_embeddings(**context):
    """
    Кодирует новые вакансии из hh_vacancies, список получен из XCom (check_new_items).
    Если список пуст — завершается без загрузки модели.
    """
    ti = context['task_instance']
    new_vac_ids = ti.xcom_pull(task_ids='check_new_items', key='new_vacancy_ids') or []
    print(f"Новых вакансий для кодирования: {len(new_vac_ids)}")

    if not new_vac_ids:
        print("Нет новых вакансий — пропускаю.")
        return

    device = _get_device()
    tokenizer, model = _load_model(device)
    ch = _get_clickhouse_client()
    total_saved = 0

    for start in range(0, len(new_vac_ids), CHUNK_SIZE):
        chunk_ids = new_vac_ids[start: start + CHUNK_SIZE]

        missing = _get_missing_ids(chunk_ids, 'vacancy_embeddings', 'vacancy_id', ch)
        if not missing:
            continue

        rows = ch.execute(
            "SELECT id, description FROM hh_vacancies "
            "WHERE id IN %(ids)s "
            "AND description IS NOT NULL "
            "AND description != ''",
            {'ids': missing},
        )
        if not rows:
            continue

        ids_batch = [r[0] for r in rows]
        texts = [str(r[1]) for r in rows]

        print(f"Кодирую {len(texts)} вакансий (chunk {start // CHUNK_SIZE + 1})...")
        embeddings = _encode_texts(texts, tokenizer, model, device)

        emb_map = {vid: emb for vid, emb in zip(ids_batch, embeddings)}
        _save_embeddings(emb_map, 'vacancy_id', 'vacancy_embeddings', ch)
        total_saved += len(emb_map)

    print(f"=== LaBSE вакансии завершены: добавлено {total_saved} записей ===")


# ── Определение DAG ───────────────────────────────────────────────────────────

with DAG(
    dag_id='labse_embeddings',
    description='LaBSE эмбеддинги для новых резюме (resume_embeddings) и вакансий (vacancy_embeddings).',
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz=MSK),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=['embeddings', 'labse', 'bert'],
) as dag:

    check_task = PythonOperator(
        task_id='check_new_items',
        python_callable=check_new_items,
    )

    resume_task = PythonOperator(
        task_id='encode_resume_embeddings',
        python_callable=encode_resume_embeddings,
        execution_timeout=pendulum.duration(hours=4),
    )

    vacancy_task = PythonOperator(
        task_id='encode_vacancy_embeddings',
        python_callable=encode_vacancy_embeddings,
        execution_timeout=pendulum.duration(hours=2),
    )

    check_task >> resume_task >> vacancy_task

"""
DAG для сбора резюме с Habr Career (career.habr.com).

Профили публичны и рендерятся через Vue SSR — авторизация не нужна.
Данные сохраняются в таблицу hh_resumes с полем source='career.habr.com'.

habr_resume_daily  — пн-сб в 03:00 МСК, 3 стр. × ~9 резюме на профессию.
                     Запускается ПОСЛЕ hh_resume_daily (позже по времени),
                     чтобы не смешивать нагрузку на разные сайты.
"""
import sys
import os

sys.path.insert(0, '/opt/airflow/parsers')

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

from professions_list import PROFESSIONS_LIST
from habr_resume_parser_db import HabrResumeParser

MSK = pendulum.timezone('Europe/Moscow')
IT_PROFESSIONS = sorted(PROFESSIONS_LIST['it_tech'])

DEFAULT_ARGS = {
    'owner': 'hr-ai-scout',
    'retries': 1,
    'retry_delay': pendulum.duration(minutes=10),
}


def _get_clickhouse_client():
    from clickhouse_driver import Client
    return Client(
        host=os.getenv('CLICKHOUSE_HOST', 'localhost'),
        port=int(os.getenv('CLICKHOUSE_PORT', 9000)),
        user=os.getenv('CLICKHOUSE_USER', 'default'),
        password=os.getenv('CLICKHOUSE_PASSWORD', ''),
        database=os.getenv('CLICKHOUSE_DATABASE', 'default'),
        settings={'use_numpy': True},
    )


def parse_habr_resumes(pages: int = 3, **context):
    """
    Собирает резюме с Habr Career по каждой IT-профессии.
    Habr показывает ~9 профилей на страницу поиска.
    Задержка 3с между профилями для соблюдения лимитов.
    """
    clickhouse = _get_clickhouse_client()
    parser = HabrResumeParser(timeout=45, max_retries=2)

    print(f"Habr Career: профессий={len(IT_PROFESSIONS)}, стр.={pages}")

    for prof in IT_PROFESSIONS:
        print(f">>> Habr: {prof}")
        df = parser.load_resumes(
            search_terms=[prof],
            pages=pages,
            delay=5,
        )
        parser.save_to_clickhouse(df, clickhouse)

    print("=== Habr Career: парсинг завершён ===")


with DAG(
    dag_id='habr_resume_daily',
    description='Ежедневный сбор резюме Habr Career: 3 стр. × ~9 профилей. Данные в hh_resumes с source=career.habr.com.',
    schedule='0 3 * * 1-6',         # пн-сб в 03:00 МСК
    start_date=pendulum.datetime(2024, 1, 1, tz=MSK),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=['resumes', 'habr', 'daily'],
) as dag:

    PythonOperator(
        task_id='parse_habr_resumes_daily',
        python_callable=parse_habr_resumes,
        op_kwargs={'pages': 3},
    )

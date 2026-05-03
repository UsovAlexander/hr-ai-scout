"""
DAG для сбора вакансий с SuperJob.ru.

Примечание: резюме SuperJob доступны только авторизованным работодателям
и рендерятся на клиенте — парсинг резюме без авторизации невозможен.
Данные сохраняются в таблицу hh_vacancies с полем source='superjob.ru'.

sj_vacancy_daily — каждый день в 04:00 МСК, 5 стр. на профессию.
                   Запускается ПОСЛЕ vacancy_daily (hh.ru), чтобы не создавать
                   одновременную нагрузку на разные сайты.
"""
import sys
import os

sys.path.insert(0, '/opt/airflow/parsers')

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

from professions_list import PROFESSIONS_LIST
from sj_vacancy_parser_db import SJVacancyParser

MSK = pendulum.timezone('Europe/Moscow')
IT_PROFESSIONS = sorted(PROFESSIONS_LIST['it_tech'])

DEFAULT_ARGS = {
    'owner': 'hr-ai-scout',
    'retries': 1,
    'retry_delay': pendulum.duration(minutes=5),
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


def parse_sj_vacancies(pages: int = 5, items_on_page: int = 20, **context):
    clickhouse = _get_clickhouse_client()
    parser = SJVacancyParser(timeout=60, max_retries=3)

    print(f"SuperJob: профессий={len(IT_PROFESSIONS)}, стр.={pages}")

    for prof in IT_PROFESSIONS:
        print(f">>> SJ: {prof}")
        df = parser.load_vacancies(
            search_terms=[prof],
            areas=['Москва'],
            pages=pages,
            items_on_page=items_on_page,
            delay=2,
        )
        parser.save_to_clickhouse(df, clickhouse)

    print("=== SuperJob: парсинг вакансий завершён ===")


with DAG(
    dag_id='sj_vacancy_daily',
    description='Ежедневный сбор вакансий SuperJob.ru: 5 стр. × 20. Данные в hh_vacancies с source=superjob.ru.',
    schedule='0 4 * * *',           # каждый день в 04:00 МСК (после HH.ru DAG-ов)
    start_date=pendulum.datetime(2024, 1, 1, tz=MSK),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=['vacancies', 'superjob', 'daily'],
) as dag:

    PythonOperator(
        task_id='parse_sj_vacancies_daily',
        python_callable=parse_sj_vacancies,
        op_kwargs={'pages': 5, 'items_on_page': 20},
    )

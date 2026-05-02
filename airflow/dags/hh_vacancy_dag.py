"""
DAG для сбора вакансий с HH.ru.

vacancy_daily — запускается только по триггеру от resume_daily или resume_weekly
                (schedule=None). 5 стр. × 20 вакансий на профессию.
"""
import sys
import os

sys.path.insert(0, '/opt/airflow/parsers')

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

from professions_list import PROFESSIONS_LIST
from hh_vacancy_parser_db import HHVacancyParser

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


def parse_vacancies(pages: int = 5, items_on_page: int = 20, **context):
    clickhouse = _get_clickhouse_client()
    parser = HHVacancyParser(timeout=60, max_retries=5)

    print(f"Профессий: {len(IT_PROFESSIONS)}, стр.: {pages}, вакансий на стр.: {items_on_page}")

    for prof in IT_PROFESSIONS:
        print(f">>> {prof}")
        df = parser.load_vacancies(
            search_terms=[prof],
            areas=['Москва'],
            pages=pages,
            items_on_page=items_on_page,
            delay=2,
        )
        parser.save_to_clickhouse(df, clickhouse)

    print("=== Парсинг вакансий завершён ===")


with DAG(
    dag_id='vacancy_daily',
    description='Сбор вакансий: 5 стр. × 20. Запускается триггером от resume DAG-ов.',
    schedule=None,                   # только по триггеру от resume_daily / resume_weekly
    start_date=pendulum.datetime(2024, 1, 1, tz=MSK),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=['vacancies', 'daily'],
) as dag:

    PythonOperator(
        task_id='parse_vacancies_daily',
        python_callable=parse_vacancies,
        op_kwargs={'pages': 5, 'items_on_page': 20},
    )

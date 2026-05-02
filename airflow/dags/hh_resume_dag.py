"""
DAG для сбора резюме с HH.ru.

resume_daily  — пн-сб в 01:00 МСК, 5 стр. × 20 резюме.
               По завершении триггерит vacancy_daily.

resume_weekly — сб в 23:00 МСК, 250 стр. × 20 резюме (полное обновление).
               По завершении также триггерит vacancy_daily.
"""
import sys
import os

sys.path.insert(0, '/opt/airflow/parsers')

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from professions_list import PROFESSIONS_LIST
from hh_resume_parser_db import HHResumeParser

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


def parse_resumes(pages: int, items_on_page: int = 20, **context):
    clickhouse = _get_clickhouse_client()
    parser = HHResumeParser(timeout=60, max_retries=5)

    for prof in IT_PROFESSIONS:
        print(f">>> {prof} ({pages} стр.)")
        df = parser.load_resumes(
            search_terms=[prof],
            areas=['Москва'],
            pages=pages,
            items_on_page=items_on_page,
            delay=2,
        )
        parser.save_to_clickhouse(df, clickhouse)

    print("=== Парсинг резюме завершён ===")


# ── DAG 1: Ежедневный (пн-сб в 01:00 МСК) ────────────────────────────────────

with DAG(
    dag_id='resume_daily',
    description='Ежедневный сбор резюме: 5 стр. × 20. После завершения запускает vacancy_daily.',
    schedule='0 1 * * 1-6',
    start_date=pendulum.datetime(2024, 1, 1, tz=MSK),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=['resumes', 'daily'],
) as daily_dag:

    parse_task = PythonOperator(
        task_id='parse_resumes_daily',
        python_callable=parse_resumes,
        op_kwargs={'pages': 5, 'items_on_page': 20},
    )

    trigger_vacancies = TriggerDagRunOperator(
        task_id='trigger_vacancy_daily',
        trigger_dag_id='vacancy_daily',
        wait_for_completion=False,  # не ждём завершения vacancy, просто запускаем
    )

    parse_task >> trigger_vacancies


# ── DAG 2: Еженедельный (сб в 23:00 МСК) ─────────────────────────────────────

with DAG(
    dag_id='resume_weekly',
    description='Еженедельный полный сбор резюме: 250 стр. × 20. После завершения запускает vacancy_daily.',
    schedule='0 23 * * 6',
    start_date=pendulum.datetime(2024, 1, 1, tz=MSK),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=['resumes', 'weekly'],
) as weekly_dag:

    parse_task_weekly = PythonOperator(
        task_id='parse_resumes_weekly',
        python_callable=parse_resumes,
        op_kwargs={'pages': 250, 'items_on_page': 20},
    )

    trigger_vacancies_weekly = TriggerDagRunOperator(
        task_id='trigger_vacancy_daily',
        trigger_dag_id='vacancy_daily',
        wait_for_completion=False,
    )

    parse_task_weekly >> trigger_vacancies_weekly

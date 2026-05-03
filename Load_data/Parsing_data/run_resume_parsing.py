import os
import sys
from dotenv import load_dotenv
from tqdm import tqdm
from clickhouse_driver import Client

from professions_list import PROFESSIONS_LIST
from hh_resume_parser_db import HHResumeParser

load_dotenv('/Users/user/Documents/Magistracy/year_project/hr-ai-scout/.env')

clickhouse = Client(
    host=os.getenv('CLICKHOUSE_HOST'),
    port='9000',
    user=os.getenv('CLICKHOUSE_USER'),
    password=os.getenv('CLICKHOUSE_PASSWORD'),
    database=os.getenv('CLICKHOUSE_DATABASE'),
    settings={'use_numpy': True}
)

def get_unloaded_positions():
    try:
        loaded = clickhouse.query_dataframe('SELECT DISTINCT search_query FROM hh_resumes')
        loaded_set = set(loaded['search_query'].tolist())
    except Exception:
        loaded_set = set()
    all_positions = set(PROFESSIONS_LIST['it_tech'])
    unloaded = sorted(all_positions - loaded_set)
    print(f"Уже загружено: {len(loaded_set)}, осталось: {len(unloaded)}")
    return unloaded

positions = get_unloaded_positions()
parser = HHResumeParser(timeout=60, max_retries=5)

for prof in tqdm(positions, desc='Профессии'):
    print(f'\n>>> {prof}', flush=True)
    df = parser.load_resumes(
        search_terms=[prof],
        areas=['Москва'],
        pages=250,
        items_on_page=20,
        delay=2
    )
    parser.save_to_clickhouse(df, clickhouse)

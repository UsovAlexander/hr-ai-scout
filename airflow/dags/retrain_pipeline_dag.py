"""
DAG 4: Дообучение пайплайна на решениях рекрутеров.

Логика:
  1. Проверяет наличие новых записей в recruiter_decisions за последние 24 ч.
     Если новых нет — досрочный выход (ShortCircuit).
  2. Загружает ВСЕ recruiter_decisions и строит признаки как в ноутбуке
     BERT_TFIDF_Experiments.ipynb:
       - Парсинг salary / last_company_experience_period
       - location_matching, resume_skill_count_in_vacancy, last_position_in_vacancy
       - similarity_score_tfidf  — TF-IDF фит на резюме + transform вакансий; векторайзер сохраняется в app/
       - sim_labse_en_ru         — LaBSE эмбеддинги из ClickHouse (resume/vacancy_embeddings)
       - als_score               — ALS (factors=64, reg=0.1, iter=30)
  3. Переобучает CatBoost-пайплайн с теми же гиперпараметрами, что в pkl.
     Сохраняет обновлённый пайплайн → app/pipeline_cb_als_sim_labse_en_ru.pkl
     и ALS-факторы                  → app/als_model.pkl

Запускается вручную или как триггер после labse_embeddings.
"""
import os
import pickle
import re
import warnings

import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

warnings.simplefilter('ignore', FutureWarning)

# ── Константы ─────────────────────────────────────────────────────────────────
PIPELINE_PATH   = '/opt/airflow/app/pipeline_cb_als_sim_labse_en_ru.pkl'
VECTORIZER_PATH = '/opt/airflow/app/tfidf_vectorizer.pkl'
ALS_PATH        = '/opt/airflow/app/als_model.pkl'
MODEL_NAME_LABSE = 'cointegrated/LaBSE-en-ru'
RANDOM_STATE    = 42

BASE_FEATURES = [
    'vacancy_area', 'vacancy_experience', 'vacancy_employment', 'vacancy_schedule',
    'resume_salary', 'resume_age', 'resume_experience_months', 'resume_location',
    'resume_gender', 'resume_applicant_status', 'resume_last_company_experience_months',
    'location_matching', 'resume_skill_count_in_vacancy', 'last_position_in_vacancy',
    'similarity_score_tfidf',
]
FEATURE_COLS = BASE_FEATURES + ['als_score', 'sim_labse_en_ru']
CAT_FEATURES = [
    'vacancy_area', 'vacancy_experience', 'vacancy_employment', 'vacancy_schedule',
    'resume_location', 'resume_gender', 'resume_applicant_status',
]

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


def _parse_salary(text: str) -> float:
    """Конвертирует строку зарплаты в рубли (логика из ноутбука, In[5])."""
    if not text or text in ('NDT', ''):
        return 0.0
    parts = str(text).split()
    num_str = ''.join(p for p in parts if re.fullmatch(r'\d+', p))
    if not num_str:
        return 0.0
    num = int(num_str)
    rates_rub = {'₽': 1.0, '$': 80.85, '€': 94.14, '₴': 1.94, '₸': 0.150,
                 '₼': 47.8, '₾': 33.5, 'Br': 28.7, "so'm": 0.0068}
    rate = next((rates_rub[s] for s in parts if s in rates_rub), 1.0)
    return float(num) * rate


def _period_to_months(text: str) -> float:
    """Парсит период опыта в месяцы (логика из ноутбука, In[7])."""
    if not text or text in ('NDT', ''):
        return 0.0
    months = 0
    for pat in [r'(\d+)\s*год', r'(\d+)\s*лет']:
        m = re.search(pat, str(text))
        if m:
            months += int(m.group(1)) * 12
    m = re.search(r'(\d+)\s*месяц', str(text))
    if m:
        months += int(m.group(1))
    return float(months)


def _ensure_nltk():
    import nltk
    for resource, name in [
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger_ru', 'averaged_perceptron_tagger_ru'),
        ('corpora/wordnet', 'wordnet'),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(name, quiet=True)


_morph = None
_stop_words = None


def _get_morph():
    global _morph
    if _morph is None:
        import pymorphy3
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


def _get_stop_words():
    global _stop_words
    if _stop_words is None:
        from nltk.corpus import stopwords
        _stop_words = set(stopwords.words('russian') + stopwords.words('english'))
    return _stop_words


def _preprocess_text_for_tfidf(text: str) -> str:
    """Лемматизация текста для TF-IDF (из tfidf_embeddings_dag)."""
    from gensim.utils import simple_preprocess
    morph = _get_morph()
    stop_words = _get_stop_words()
    tokens = simple_preprocess(str(text), deacc=True, min_len=2)
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(morph.parse(t)[0].normal_form for t in tokens)


def _load_labse_embeddings(table: str, id_col: str, ids: list, ch) -> dict:
    """Загружает LaBSE-эмбеддинги из ClickHouse для указанных id."""
    import numpy as np
    if not ids:
        return {}
    rows = ch.execute(
        f"SELECT {id_col}, embedding FROM {table} "
        f"WHERE model_name = %(m)s AND {id_col} IN %(ids)s",
        {'m': MODEL_NAME_LABSE, 'ids': [str(i) for i in ids]},
    )
    return {row[0]: np.array(row[1], dtype=np.float32) for row in rows}


# ── Задачи DAG ────────────────────────────────────────────────────────────────

def check_new_decisions(**context) -> bool:
    """
    ShortCircuit: пропускает DAG если за последние 24 ч нет новых решений рекрутеров.
    """
    ch = _get_clickhouse_client()
    rows = ch.execute(
        "SELECT count() FROM recruiter_decisions "
        "WHERE decided_at >= now() - INTERVAL 1 DAY"
    )
    count = rows[0][0]
    print(f"Новых решений рекрутеров за 24 ч: {count}")
    return count > 0


def retrain_pipeline(**context):
    """
    Полный пайплайн дообучения:
      загрузка данных → препроцессинг → признаки → CatBoost → сохранение.
    """
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    from catboost import CatBoostClassifier
    from implicit.als import AlternatingLeastSquares

    ch = _get_clickhouse_client()

    # ── 1. Загрузка данных ─────────────────────────────────────────────────────
    print("Загружаю recruiter_decisions...")
    cols = [
        'vacancy_id', 'vacancy_area', 'vacancy_experience',
        'vacancy_employment', 'vacancy_schedule', 'vacancy_description',
        'resume_id', 'resume_skills', 'resume_last_position',
        'resume_last_experience_description', 'resume_last_company_experience_period',
        'resume_salary', 'resume_age', 'resume_experience_months',
        'resume_location', 'resume_gender', 'resume_applicant_status', 'target',
    ]
    rows = ch.execute(f"SELECT {', '.join(cols)} FROM recruiter_decisions")
    df = pd.DataFrame(rows, columns=cols)
    print(f"Загружено {len(df):,} записей")

    # ── 2. Препроцессинг ───────────────────────────────────────────────────────
    df['resume_salary'] = df['resume_salary'].fillna('').apply(_parse_salary)
    df['resume_last_company_experience_months'] = (
        df['resume_last_company_experience_period'].fillna('').apply(_period_to_months)
    )

    for col in CAT_FEATURES:
        df[col] = df[col].fillna('NDT')

    df['resume_age'] = df['resume_age'].fillna(df['resume_age'].mean())
    df['resume_experience_months'] = df['resume_experience_months'].fillna(0)
    df['vacancy_description'] = df['vacancy_description'].fillna('').astype(str)
    df['resume_last_experience_description'] = (
        df['resume_last_experience_description'].fillna('').astype(str)
    )
    df['resume_last_position'] = df['resume_last_position'].fillna('').astype(str)
    df['resume_skills'] = df['resume_skills'].fillna('').astype(str)

    gender_map = {
        'Мужчина': 'Мужчина', 'Male': 'Мужчина',
        'Женщина': 'Женщина', 'Female': 'Женщина',
    }
    df['resume_gender'] = df['resume_gender'].apply(
        lambda x: gender_map.get(x, 'Неизвестно')
    )

    # Ограничения из ноутбука (In[8])
    df = df[df['resume_salary'] <= 1e7]
    df.loc[df['resume_experience_months'] > 720, 'resume_experience_months'] = 720
    df.loc[df['resume_last_company_experience_months'] > 720,
           'resume_last_company_experience_months'] = 720

    print(f"После очистки: {len(df):,} записей")

    # ── 3. Ручные признаки ─────────────────────────────────────────────────────
    df['location_matching'] = (df['vacancy_area'] == df['resume_location']).astype(int)

    def _skill_count(row):
        skills = (row['resume_skills']
                  .replace('[', '').replace(']', '').replace("'", '').split(', '))
        return sum(1 for s in skills if s and s in row['vacancy_description'])

    def _position_in_vacancy(row):
        pos = row['resume_last_position']
        if not pos:
            return 0.0
        bow = list({w for sep in (' ', '-', '_') for w in pos.split(sep) if w})
        if not bow:
            return 0.0
        return sum(1 for w in bow if w in row['vacancy_description']) / len(bow)

    print("Вычисляю ручные признаки...")
    df['resume_skill_count_in_vacancy'] = df.apply(_skill_count, axis=1)
    df['last_position_in_vacancy'] = df.apply(_position_in_vacancy, axis=1)

    # ── 4. TF-IDF: фит векторайзера + cosine similarity ───────────────────────
    # Обучаем здесь на лету на текстах резюме из recruiter_decisions
    # (параметры идентичны ноутбуку, In[18]).
    # Сохраняем в app/ — FastAPI забирает оттуда на проде.
    # Sparse-матрицы: ~8 МБ для 20k резюме × 5000 фичей.
    print("Обучаю TF-IDF векторайзер и вычисляю косинусное сходство...")
    _ensure_nltk()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as sk_normalize

    unique_res = (df[['resume_id', 'resume_last_experience_description']]
                  .drop_duplicates('resume_id'))
    unique_vac = df[['vacancy_id', 'vacancy_description']].drop_duplicates('vacancy_id')

    print(f"  Лемматизирую {len(unique_res)} уникальных резюме для фита...")
    res_processed = [_preprocess_text_for_tfidf(t)
                     for t in unique_res['resume_last_experience_description']]

    vectorizer = TfidfVectorizer(
        max_features=5000, min_df=2, max_df=0.8,
        ngram_range=(1, 2), lowercase=False,
    )
    res_tfidf = sk_normalize(vectorizer.fit_transform(res_processed), norm='l2')
    res_idx_map = {rid: i for i, rid in enumerate(unique_res['resume_id'])}

    os.makedirs(os.path.dirname(VECTORIZER_PATH), exist_ok=True)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  Векторайзер сохранён: {VECTORIZER_PATH} "
          f"(словарь: {len(vectorizer.vocabulary_):,} токенов)")

    print(f"  Лемматизирую {len(unique_vac)} уникальных вакансий...")
    vac_tfidf = sk_normalize(
        vectorizer.transform([_preprocess_text_for_tfidf(t)
                               for t in unique_vac['vacancy_description']]),
        norm='l2',
    )
    vac_idx_map = {vid: i for i, vid in enumerate(unique_vac['vacancy_id'])}

    vi = [vac_idx_map[v] for v in df['vacancy_id']]
    ri = [res_idx_map[r] for r in df['resume_id']]
    df['similarity_score_tfidf'] = (
        vac_tfidf[vi].multiply(res_tfidf[ri]).sum(axis=1).A1.tolist()
    )

    # ── 5. LaBSE cosine similarity ─────────────────────────────────────────────
    print("Загружаю LaBSE-эмбеддинги из ClickHouse...")
    all_vac_ids = df['vacancy_id'].unique().tolist()
    all_res_ids = df['resume_id'].unique().tolist()

    vac_emb_map = _load_labse_embeddings('vacancy_embeddings', 'vacancy_id',
                                         all_vac_ids, ch)
    res_emb_map = _load_labse_embeddings('resume_embeddings', 'resume_id',
                                         all_res_ids, ch)
    print(f"  Вакансий с эмбеддингами: {len(vac_emb_map)}, "
          f"резюме: {len(res_emb_map)}")

    df['sim_labse_en_ru'] = [
        float(np.dot(vac_emb_map[str(row.vacancy_id)],
                     res_emb_map[str(row.resume_id)]))
        if str(row.vacancy_id) in vac_emb_map and str(row.resume_id) in res_emb_map
        else 0.0
        for row in df.itertuples()
    ]

    # ── 6. ALS score ───────────────────────────────────────────────────────────
    print("Обучаю ALS (factors=64, reg=0.1, iter=30)...")
    unique_vacancies = df['vacancy_id'].unique().tolist()
    unique_resumes = df['resume_id'].unique().tolist()
    vac2id = {v: i for i, v in enumerate(unique_vacancies)}
    res2id = {r: i for i, r in enumerate(unique_resumes)}

    interaction_matrix = csr_matrix(
        (df['target'].values.astype(np.float32),
         ([vac2id[v] for v in df['vacancy_id']],
          [res2id[r] for r in df['resume_id']])),
        shape=(len(unique_vacancies), len(unique_resumes)),
    )

    als = AlternatingLeastSquares(
        factors=64, regularization=0.1, iterations=30,
        random_state=RANDOM_STATE, num_threads=0,
    )
    als.fit(interaction_matrix.T)

    vac_factors = als.item_factors
    res_factors = als.user_factors

    df['als_score'] = [
        float(np.dot(vac_factors[vac2id[v]], res_factors[res2id[r]]))
        for v, r in zip(df['vacancy_id'], df['resume_id'])
    ]

    # ── 7. Переобучение CatBoost-пайплайна ────────────────────────────────────
    X = df[FEATURE_COLS].copy()
    y = df['target'].copy()

    print(f"Тренирую пайплайн: X={X.shape}, positives={int(y.sum())}")

    if os.path.exists(PIPELINE_PATH):
        with open(PIPELINE_PATH, 'rb') as f:
            existing = pickle.load(f)
        cb_params = existing.named_steps['model'].get_all_params()
        kept = {k: cb_params[k] for k in
                ('iterations', 'depth', 'learning_rate', 'l2_leaf_reg',
                 'auto_class_weights', 'random_seed')
                if k in cb_params}
        kept.setdefault('random_seed', RANDOM_STATE)
        kept['verbose'] = 0
        catboost = CatBoostClassifier(**kept)
        print(f"  Загружены гиперпараметры из существующего pkl: {kept}")
    else:
        catboost = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
            auto_class_weights='Balanced', random_seed=RANDOM_STATE, verbose=0,
        )
        print("  Существующий pkl не найден — используются дефолтные гиперпараметры.")

    new_pipeline = Pipeline([
        ('preprocessing', ColumnTransformer([
            ('cat', OrdinalEncoder(
                handle_unknown='use_encoded_value', unknown_value=-1
            ), CAT_FEATURES),
        ], remainder='passthrough')),
        ('model', catboost),
    ])
    new_pipeline.fit(X, y)

    os.makedirs(os.path.dirname(PIPELINE_PATH), exist_ok=True)
    with open(PIPELINE_PATH, 'wb') as f:
        pickle.dump(new_pipeline, f)
    print(f"Пайплайн сохранён: {PIPELINE_PATH}")

    # ── 8. Сохранение ALS-модели ───────────────────────────────────────────────
    als_artifact = {
        'vac2id': vac2id,
        'res2id': res2id,
        'vac_factors': vac_factors,
        'res_factors': res_factors,
    }
    with open(ALS_PATH, 'wb') as f:
        pickle.dump(als_artifact, f)
    print(f"ALS-модель сохранена: {ALS_PATH}")
    print("=== Дообучение завершено ===")


# ── Определение DAG ───────────────────────────────────────────────────────────

with DAG(
    dag_id='retrain_pipeline',
    description='Дообучение CatBoost-пайплайна на новых решениях рекрутеров.',
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz=MSK),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=['training', 'catboost', 'labse'],
) as dag:

    check_task = ShortCircuitOperator(
        task_id='check_new_decisions',
        python_callable=check_new_decisions,
    )

    retrain_task = PythonOperator(
        task_id='retrain_pipeline',
        python_callable=retrain_pipeline,
        execution_timeout=pendulum.duration(hours=3),
    )

    check_task >> retrain_task

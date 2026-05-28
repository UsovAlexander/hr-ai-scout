import re
import nltk
import pymorphy3
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

for _resource, _name in [
    ('corpora/stopwords', 'stopwords'),
    ('taggers/averaged_perceptron_tagger_ru', 'averaged_perceptron_tagger_ru'),
    ('corpora/wordnet', 'wordnet'),
]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_name, quiet=True)

morph = pymorphy3.MorphAnalyzer()


# ── Текстовый препроцессинг ────────────────────────────────────────────────────

def lemmatize_russian(tokens):
    """Лемматизация русских слов"""
    lemmas = []
    for token in tokens:
        parsed = morph.parse(token)[0]  # Берем самый вероятный разбор
        lemmas.append(parsed.normal_form)
    return lemmas


def tokenize_and_lemmatize(text):
    """Токенизация текста с лемматизацией и удалением стоп-слов"""
    tokens = simple_preprocess(text, deacc=True, min_len=2)
    stop_words = set(stopwords.words('russian') + stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = lemmatize_russian(tokens)
    return lemmatized_tokens


# ── Числовой препроцессинг ─────────────────────────────────────────────────────

def experience_to_months(text):
    """Парсит строку периода опыта в месяцы (из ноутбука In[7])."""
    months = 0
    for pat in [r'(\d+)\s*год', r'(\d+)\s*лет']:
        m = re.search(pat, str(text))
        if m:
            months += int(m.group(1)) * 12
    m = re.search(r'(\d+)\s*месяц', str(text))
    if m:
        months += int(m.group(1))
    return months if months > 0 else np.nan


def parse_salary(text: str) -> float:
    """Конвертирует строку зарплаты в рубли (из ноутбука In[6])."""
    if not text or text in ('NDT', ''):
        return 0.0
    parts = str(text).split()
    num_str = ''.join(p for p in parts if re.fullmatch(r'\d+', p))
    if not num_str:
        return 0.0
    num = int(num_str)
    rates_rub = {
        '₽': 1.0, '$': 80.85, '€': 94.14, '₴': 1.94, '₸': 0.150,
        '₼': 47.8, '₾': 33.5, 'Br': 28.7, "so'm": 0.0068,
    }
    rate = next((rates_rub[s] for s in parts if s in rates_rub), 1.0)
    return float(num) * rate


# ── Ручные признаки ────────────────────────────────────────────────────────────

def compute_location_matching(row):
    return 1 if row['vacancy_area'] == row['resume_location'] else 0


def resume_skill_count_in_vacancy(row):
    """Количество навыков резюме в тексте вакансии (из ноутбука In[11])."""
    skills_val = row.get('resume_skills', '')
    if isinstance(skills_val, list):
        skills = [str(s) for s in skills_val]
    else:
        skills = str(skills_val).replace('[', '').replace(']', '').replace("'", "").split(', ')
    return sum(1 for s in skills if s and s in row['vacancy_description'])


def last_position_in_vacancy(row):
    """Доля слов последней должности в описании вакансии (из ноутбука In[11])."""
    pos = row.get('resume_last_position', '')
    if not pos:
        return 0.0
    bow = []
    for sep in [' ', '-', '_']:
        bow += str(pos).split(sep=sep)
    bow = list(set(w for w in bow if w))
    if not bow:
        return 0.0
    return sum(1 for w in bow if w in row['vacancy_description']) / len(bow)


# ── TF-IDF эмбеддинги ─────────────────────────────────────────────────────────

def get_tfidf_embeddings(texts, vectorizer=None, fit=True):
    """Создание TF-IDF эмбеддингов с лемматизацией (из ноутбука In[18])."""
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            tokenizer=tokenize_and_lemmatize,
            token_pattern=None,
            lowercase=False,
        )
        embeddings = vectorizer.fit_transform(texts)
    else:
        embeddings = vectorizer.transform(texts)
    return embeddings, vectorizer


def calculate_cosine_similarity(embeddings1, embeddings2):
    """Попарное косинусное сходство между двумя наборами эмбеддингов (из ноутбука In[17])."""
    similarities = []
    for i in range(embeddings1.shape[0]):
        emb1_row = embeddings1[i]
        emb2_row = embeddings2[i]
        similarity = cosine_similarity(emb1_row, emb2_row)[0][0]
        similarities.append(similarity)
    return similarities


def compute_similarity_features(df, tfidf_vectorizer, experience_embeddings):
    """Вычисляет similarity_score_tfidf для каждой строки df (старый подход — для Streamlit)."""
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    vacancy_description = df['vacancy_description'].unique().tolist()
    vacancy_embedding = tfidf_vectorizer.transform(vacancy_description)

    max_index = min(experience_embeddings.shape[0], len(df))
    exp_subset = experience_embeddings[:max_index]
    similarities = cos_sim(vacancy_embedding, exp_subset).flatten().tolist()
    if len(similarities) < len(df):
        avg = np.mean(similarities) if similarities else 0.0
        similarities.extend([avg] * (len(df) - len(similarities)))

    df['similarity_score_tfidf'] = similarities
    return df


# ── BERT / LaBSE эмбеддинги ───────────────────────────────────────────────────

def encode_texts(texts, tokenizer, model, batch_size=64, prefix='', device=None):
    """
    Батчевое кодирование текстов в L2-нормированные эмбеддинги.
    Mean pooling по токенам, взвешенный attention mask (из ноутбука In[27]).
    """
    import torch
    import torch.nn.functional as F

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    if prefix:
        texts = [prefix + t for t in texts]

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            out = model(**encoded)

        token_emb = out.last_hidden_state                              # (B, T, H)
        mask = encoded['attention_mask'].unsqueeze(-1).float()         # (B, T, 1)
        pooled = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = F.normalize(pooled, p=2, dim=1)
        all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


# ── ALS ───────────────────────────────────────────────────────────────────────

def get_als_score(vacancy_id, resume_id, vacancy2id, resume2id,
                  vacancy_factors, resume_factors):
    """
    ALS-скор для пары (vacancy_id, resume_id).
    Возвращает 0 при cold-start (id не встречался при обучении) (из ноутбука In[32]).
    """
    if vacancy_id not in vacancy2id or resume_id not in resume2id:
        return 0.0
    return float(np.dot(
        vacancy_factors[vacancy2id[vacancy_id]],
        resume_factors[resume2id[resume_id]],
    ))


# ── Список признаков ──────────────────────────────────────────────────────────

FEATURES = [
    'vacancy_area',
    'vacancy_experience',
    'vacancy_employment',
    'vacancy_schedule',
    'resume_salary',
    'resume_age',
    'resume_experience_months',
    'resume_location',
    'resume_gender',
    'resume_applicant_status',
    'resume_last_company_experience_months',
    'location_matching',
    'resume_skill_count_in_vacancy',
    'last_position_in_vacancy',
    'similarity_score_tfidf',
    'als_score',
    'sim_labse_en_ru',
]

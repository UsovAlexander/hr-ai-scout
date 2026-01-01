import re
import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import nltk
    from nltk.corpus import stopwords
    import pymorphy3
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    STOPWORDS = set(stopwords.words('russian') + stopwords.words('english'))
    MORPH = pymorphy3.MorphAnalyzer()
    NLP_AVAILABLE = True
except ImportError:
    STOPWORDS = set()
    MORPH = None
    NLP_AVAILABLE = False
    print("Warning: NLP libraries not available. Install nltk and pymorphy3 for better feature engineering.")


def clean_text(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r'[^а-яА-ЯёЁa-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


def tokenize_simple(text: str) -> List[str]:
    text = clean_text(text)
    tokens = text.split()
    if STOPWORDS:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    if not NLP_AVAILABLE or not MORPH:
        return tokens
    return [MORPH.parse(token)[0].normal_form for token in tokens]


def calculate_location_matching(vacancy_area: str, resume_location: str) -> float:
    if pd.isna(vacancy_area) or pd.isna(resume_location):
        return 0.0

    v_area = clean_text(str(vacancy_area))
    r_loc = clean_text(str(resume_location))

    if v_area == r_loc:
        return 1.0
    
    if v_area in r_loc or r_loc in v_area:
        return 1.0
    
    return 0.0


def calculate_resume_skill_count_in_vacancy(
    vacancy_description: str, 
    resume_skills: str
) -> float:
    if pd.isna(vacancy_description) or pd.isna(resume_skills):
        return 0.0
    
    vacancy_text = clean_text(str(vacancy_description))
    
    if isinstance(resume_skills, str):
        skills_str = re.sub(r'[\[\]\'\"]', '', resume_skills)
        skills = [clean_text(skill) for skill in re.split(r'[,;\n]', skills_str)]
        skills = [s for s in skills if s and len(s) > 2]
    else:
        return 0.0
    
    count = 0
    for skill in skills:
        if skill in vacancy_text:
            count += 1
    
    return float(count)


def calculate_last_position_in_vacancy(
    vacancy_description: str, 
    resume_last_position: str
) -> float:

    if pd.isna(vacancy_description) or pd.isna(resume_last_position):
        return 0.0
    
    vacancy_tokens = set(tokenize_simple(str(vacancy_description)))
    position_tokens = tokenize_simple(str(resume_last_position))
    
    if not position_tokens:
        return 0.0
    
    matching_count = sum(1 for token in position_tokens if token in vacancy_tokens)
    
    return matching_count / len(position_tokens)


def calculate_similarity_score_tfidf(
    vacancy_description: str,
    resume_experience_description: str,
    vectorizer: Optional[TfidfVectorizer] = None
) -> tuple[float, Optional[TfidfVectorizer]]:

    if pd.isna(vacancy_description) or pd.isna(resume_experience_description):
        return 0.0, vectorizer
    
    v_text = clean_text(str(vacancy_description))
    r_text = clean_text(str(resume_experience_description))
    
    if not v_text or not r_text:
        return 0.0, vectorizer
    
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=1,
            ngram_range=(1, 2),
            lowercase=True
        )
        try:
            vectorizer.fit([v_text, r_text])
        except:
            return 0.0, vectorizer
    
    try:
        v_vec = vectorizer.transform([v_text])
        r_vec = vectorizer.transform([r_text])
        
        similarity = cosine_similarity(v_vec, r_vec)[0][0]
        return float(similarity), vectorizer
    except:
        return 0.0, vectorizer


def calculate_resume_age_from_birth_year(birth_year: float, current_year: int = 2026) -> float:
    if pd.isna(birth_year):
        return 30.0 
    
    try:
        if birth_year < 100:
            return float(birth_year)
        age = current_year - int(birth_year)
        if 18 <= age <= 90:
            return float(age)
        return 30.0
    except:
        return 30.0


def parse_experience_period(period_str: str) -> float:
    if pd.isna(period_str) or not isinstance(period_str, str):
        return 0.0
    
    months = 0
    
    years_match = re.search(r'(\d+)\s*(год|лет|года)', period_str, re.IGNORECASE)
    if years_match:
        months += int(years_match.group(1)) * 12
    
    months_match = re.search(r'(\d+)\s*месяц', period_str, re.IGNORECASE)
    if months_match:
        months += int(months_match.group(1))
    
    return float(months)


def process_csv_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    print("Calculating features")
    
    print("  - location_matching")
    df['location_matching'] = df.apply(
        lambda row: calculate_location_matching(row['vacancy_area'], row['resume_location']),
        axis=1
    )
    
    print("  - resume_skill_count_in_vacancy")
    df['resume_skill_count_in_vacancy'] = df.apply(
        lambda row: calculate_resume_skill_count_in_vacancy(
            row['vacancy_description'], 
            row.get('resume_skills', '')
        ),
        axis=1
    )
    print("  - last_position_in_vacancy")
    df['last_position_in_vacancy'] = df.apply(
        lambda row: calculate_last_position_in_vacancy(
            row['vacancy_description'],
            row.get('resume_last_position', '')
        ),
        axis=1
    )
    
    print("  - similarity_score_tfidf")
    vectorizer = None
    similarities = []
    
    for idx, row in df.iterrows():
        similarity, vectorizer = calculate_similarity_score_tfidf(
            row['vacancy_description'],
            row.get('resume_last_experience_description', ''),
            vectorizer
        )
        similarities.append(similarity)
    
    df['similarity_score_tfidf'] = similarities
    
    if 'resume_age' in df.columns:
        df['resume_age'] = df['resume_age'].apply(calculate_resume_age_from_birth_year)
    
    if 'resume_last_company_experience_period' in df.columns:
        if df['resume_last_company_experience_period'].dtype == 'object':
            df['resume_last_company_experience_months'] = df['resume_last_company_experience_period'].apply(
                parse_experience_period
            )
    
    if 'resume_experience_months' not in df.columns and 'resume_total_experience' in df.columns:
        if df['resume_total_experience'].dtype == 'object':
            df['resume_experience_months'] = df['resume_total_experience'].apply(parse_experience_period)
        else:
            df['resume_experience_months'] = df['resume_total_experience']
    
    print("Features calculated successfully!")
    return df

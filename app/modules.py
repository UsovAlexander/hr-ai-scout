import nltk
import pymorphy3
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import numpy as np

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_ru')
except LookupError:
    nltk.download('averaged_perceptron_tagger_ru')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

morph = pymorphy3.MorphAnalyzer()

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

def compute_location_matching(row):
    return 1 if row['vacancy_area'] == row['resume_location'] else 0


def resume_skill_count_in_vacancy(row):
    count = 0
    skill_list = row['resume_skills'].replace('[', '').replace(']', '').replace("'", "").split(', ')
    for i in skill_list:
        if i in row['vacancy_description']:
            count += 1
    return count


def last_position_in_vacancy(row):
    bow = []
    seps = [' ', '-', '_']
    for sep in seps:
        bow += row['resume_last_position'].split(sep=sep)
        bow = list(set(bow))
    
    c = 0
    for word in bow:
        if word in row['vacancy_description']:
            c +=1
    
    return c / len(bow)


def calculate_cosine_similarity(vacancy_embedding, experience_embeddings, num_resumes):

    from sklearn.metrics.pairwise import cosine_similarity
    
    max_index = min(experience_embeddings.shape[0], num_resumes)
    exp_subset = experience_embeddings[:max_index]
    similarities = cosine_similarity(vacancy_embedding, exp_subset).flatten().tolist()
    
    if len(similarities) < num_resumes:
        avg_similarity = np.mean(similarities) if similarities else 0.0
        similarities.extend([avg_similarity] * (num_resumes - len(similarities)))
    
    return similarities


def compute_similarity_features(df, tfidf_vectorizer, experience_embeddings):

    vacancy_description = df['vacancy_description'].unique().tolist()
    vacancy_embedding = tfidf_vectorizer.transform(vacancy_description)
    
    similarity_scores = calculate_cosine_similarity(
        vacancy_embedding, 
        experience_embeddings, 
        num_resumes=len(df)
    )
    
    df['similarity_score_tfidf'] = similarity_scores
    return df


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
    'similarity_score_tfidf'
]

import pandas as pd
import numpy as np
import json


def clean_salary(salary):
    if pd.isna(salary):
        return 0.0
    
    try:
        salary = float(salary)
        
        if salary > 10_000_000:
            return 0.0
        
        if salary < 0:
            return 0.0
        
        return float(salary)
    except:
        return 0.0


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    print("Cleaning data")
    
    if 'resume_salary' in df.columns:
        print("  - cleaning resume_salary")
        df['resume_salary'] = df['resume_salary'].apply(clean_salary)
    
    if 'resume_age' in df.columns:
        print("  - cleaning resume_age")
        df['resume_age'] = df['resume_age'].fillna(30.0).clip(lower=18, upper=90)
    
    if 'resume_experience_months' in df.columns:
        print("  - cleaning resume_experience_months")
        df['resume_experience_months'] = df['resume_experience_months'].fillna(0).clip(lower=0, upper=720)
    
    if 'resume_last_company_experience_months' in df.columns:
        print("  - cleaning resume_last_company_experience_months")
        df['resume_last_company_experience_months'] = df['resume_last_company_experience_months'].fillna(0).clip(lower=0, upper=720)
    
    prob_features = ['location_matching', 'last_position_in_vacancy', 'similarity_score_tfidf']
    for feat in prob_features:
        if feat in df.columns:
            print(f"  - cleaning {feat}")
            df[feat] = df[feat].fillna(0).clip(lower=0, upper=1)
    
    if 'resume_skill_count_in_vacancy' in df.columns:
        print("  - cleaning resume_skill_count_in_vacancy")
        df['resume_skill_count_in_vacancy'] = df['resume_skill_count_in_vacancy'].fillna(0).clip(lower=0)
    
    if 'resume_last_company_experience_months' in df.columns and 'resume_experience_months' in df.columns:
        mask = df['resume_last_company_experience_months'] > df['resume_experience_months']
        if mask.any():
            print("  - fixing resume_last_company_experience_months > resume_experience_months")
            df.loc[mask, 'resume_last_company_experience_months'] = df.loc[mask, 'resume_experience_months']
    
    print("Data cleaned!")
    return df


def export_features_to_json(df: pd.DataFrame, row_index: int = 0) -> dict:

    if row_index >= len(df):
        row_index = 0
    
    row = df.iloc[row_index]
    
    features = {
        'vacancy_area': str(row.get('vacancy_area', '')),
        'vacancy_experience': str(row.get('vacancy_experience', '')),
        'vacancy_employment': str(row.get('vacancy_employment', '')),
        'vacancy_schedule': str(row.get('vacancy_schedule', '')),
        'resume_location': str(row.get('resume_location', '')),
        'resume_gender': str(row.get('resume_gender', '')),
        'resume_applicant_status': str(row.get('resume_applicant_status', '')),
        'resume_salary': float(row.get('resume_salary', 0.0)),
        'resume_age': float(row.get('resume_age', 30.0)),
        'resume_experience_months': float(row.get('resume_experience_months', 0.0)),
        'resume_last_company_experience_months': float(row.get('resume_last_company_experience_months', 0.0)),
        'location_matching': float(row.get('location_matching', 0.0)),
        'resume_skill_count_in_vacancy': float(row.get('resume_skill_count_in_vacancy', 0.0)),
        'last_position_in_vacancy': float(row.get('last_position_in_vacancy', 0.0)),
        'similarity_score_tfidf': float(row.get('similarity_score_tfidf', 0.0))
    }
    
    return features

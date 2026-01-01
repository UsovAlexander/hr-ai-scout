import pandas as pd
import numpy as np
from typing import Dict, Any
import pickle
import os
from feature_engineer import process_csv_features
from data_cleaner import clean_dataframe


def load_model(model_path: str = None):
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__),
            "feature_engineering_with_best_optuna_lr.pkl"
        )
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def prepare_data_for_model(df: pd.DataFrame) -> pd.DataFrame:

    required_cols = [
        'vacancy_area',
        'vacancy_experience',
        'vacancy_employment', 
        'vacancy_schedule',
        'resume_location',
        'resume_gender',
        'resume_applicant_status',
        'resume_salary',
        'resume_age',
        'resume_experience_months',
        'resume_last_company_experience_months',
        'location_matching',
        'resume_skill_count_in_vacancy',
        'last_position_in_vacancy',
        'similarity_score_tfidf'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        for col in missing_cols:
            if col in ['resume_salary', 'resume_age', 'resume_experience_months', 
                      'resume_last_company_experience_months', 'location_matching',
                      'resume_skill_count_in_vacancy', 'last_position_in_vacancy',
                      'similarity_score_tfidf']:
                df[col] = 0.0
            else:
                df[col] = ""
    
    return df[required_cols].copy()


def process_csv_batch(
    csv_path: str = None,
    df: pd.DataFrame = None,
    model_path: str = None
) -> pd.DataFrame:

    if df is None:
        if csv_path is None:
            raise ValueError("Either csv_path or df must be provided")
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        df = df.copy()
    
    print(f"Loaded {len(df)} rows")
    
    print("Performing feature engineering...")
    df = process_csv_features(df)
    
    print("Cleaning data...")
    df = clean_dataframe(df)
    
    print("Preparing data for model")
    model_df = prepare_data_for_model(df)
    
    print("Loading model")
    model = load_model(model_path)
    
    print("Making predictions")
    try:
        predictions = model.predict(model_df)
        probabilities = model.predict_proba(model_df)
        
        prob_class_1 = probabilities[:, 1]
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        predictions = np.zeros(len(df))
        prob_class_1 = np.zeros(len(df))
    
    result_df = pd.DataFrame({
        'resume_id': df.get('resume_id', range(len(df))),
        'vacancy_id': df.get('vacancy_id', ''),
        'resume_title': df.get('resume_title', ''),
        'resume_location': df.get('resume_location', ''),
        'resume_salary': df.get('resume_salary', 0),
        'resume_age': df.get('resume_age', 0),
        'resume_experience_months': df.get('resume_experience_months', 0),
        'prediction': predictions.astype(int),
        'probability': prob_class_1,
        'location_matching': df['location_matching'],
        'resume_skill_count_in_vacancy': df['resume_skill_count_in_vacancy'],
        'last_position_in_vacancy': df['last_position_in_vacancy'],
        'similarity_score_tfidf': df['similarity_score_tfidf']
    })
    
    result_df = result_df.sort_values('probability', ascending=False).reset_index(drop=True)
    
    print(f"Processing complete {sum(predictions == 1)} candidates recommended for interview")
    
    return result_df


def get_summary_statistics(results_df: pd.DataFrame) -> Dict[str, Any]:

    total = len(results_df)
    recommended = sum(results_df['prediction'] == 1)
    not_recommended = sum(results_df['prediction'] == 0)
    
    stats = {
        'total_candidates': total,
        'recommended_count': recommended,
        'not_recommended_count': not_recommended,
        'recommendation_rate': recommended / total if total > 0 else 0,
        'avg_probability': results_df['probability'].mean(),
        'median_probability': results_df['probability'].median(),
        'top_candidate': {
            'resume_id': results_df.iloc[0]['resume_id'] if len(results_df) > 0 else None,
            'probability': results_df.iloc[0]['probability'] if len(results_df) > 0 else 0
        },
        'feature_stats': {
            'avg_location_match': results_df['location_matching'].mean(),
            'avg_skill_count': results_df['resume_skill_count_in_vacancy'].mean(),
            'avg_position_match': results_df['last_position_in_vacancy'].mean(),
            'avg_tfidf_similarity': results_df['similarity_score_tfidf'].mean()
        }
    }
    
    return stats

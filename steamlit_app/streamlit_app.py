import streamlit as st
import requests
import json
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from batch_processor import process_csv_batch, get_summary_statistics

FASTAPI_URL = "http://localhost:8000"

DEFAULT_JSON = {
    "vacancy_area": "Москва",
    "vacancy_experience": "От 1 года до 3 лет",
    "vacancy_employment": "Полная занятость",
    "vacancy_schedule": "Полный день",
    "resume_location": "Москва",
    "resume_gender": "Мужчина",
    "resume_applicant_status": "Активно ищет работу",
    "resume_salary": 80000.0,
    "resume_age": 25.0,
    "resume_experience_months": 24.0,
    "resume_last_company_experience_months": 12.0,
    "location_matching": 1.0,
    "resume_skill_count_in_vacancy": 5.0,
    "last_position_in_vacancy": 0.6,
    "similarity_score_tfidf": 0.75
}

st.title("HR AI Scout")


mode = st.radio(
    "Способ ввода:",
    ["Одиночная оценка - JSON", "Пакетная оценка - CSV"],
    horizontal=True
)

if mode == "Пакетная оценка - CSV":
    
    uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])
    
    if uploaded_file is not None:
        
        if st.button("Обработать", type="primary", use_container_width=True):
            try:
                with st.spinner("Загрузка данных"):
                    df = pd.read_csv(uploaded_file)
                
                with st.spinner("Обработка данных и расчет признаков"):
                    results_df = process_csv_batch(df=df)
                
                st.success("Обработка завершена")
                
                stats = get_summary_statistics(results_df)
                
                st.markdown("### Статистика обработки")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Всего кандидатов", stats['total_candidates'])
                
                with col2:
                    st.metric(
                        "Рекомендовано", stats['recommended_count'])
                
                with col3:
                    st.metric("Не рекомендовано", stats['not_recommended_count'])
                
                with col4:
                    st.metric("Средняя вероятность", f"{stats['avg_probability']:.2%}")
                
                st.markdown("### Результаты оценки")
                
                display_df = results_df[[
                    'resume_id', 'resume_title', 'resume_location', 
                    'resume_age', 'resume_salary', 'resume_experience_months',
                    'prediction', 'probability',
                    'location_matching', 'resume_skill_count_in_vacancy',
                    'last_position_in_vacancy', 'similarity_score_tfidf'
                ]].copy()
                
                display_df['probability'] = display_df['probability'].apply(lambda x: f"{x:.2%}")
                display_df['prediction'] = display_df['prediction'].apply(lambda x: "Подходит" if x == 1 else "Не подходит")
                
                display_df.columns = [
                    'ID', 'Название', 'Локация', 'Возраст', 'Зарплата', 'Опыт (мес)',
                    'Решение', 'Вероятность',
                    'Совп. локации', 'Навыки', 'Совп. позиции', 'TF-IDF'
                ]
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=400
                )
                
            except Exception as e:
                st.error(f"Ошибка при обработке: {str(e)}")
                st.exception(e)

else:
    
    json_input = st.text_area(
        "Введите JSON:",
        value=json.dumps(DEFAULT_JSON, indent=2, ensure_ascii=False),
        height=400
    )
    
    if st.button("Получить оценку", type="primary", use_container_width=True):
        try:
            candidate_data = json.loads(json_input)
            
            with st.spinner("Обработка данных"):
                response = requests.post(
                    f"{FASTAPI_URL}/forward",
                    json=candidate_data,
                    timeout=10
                )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction")
                probability = result.get("probability")
                
                st.markdown("---")
                st.markdown("### Результат оценки")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.success("Решение: Подходит")
                    else:
                        st.error("Решение: Не подходит")
                
                with col2:
                    st.metric(
                        label="Вероятность",
                        value=f"{probability:.2%}"
                    )
            
            else:
                st.error(f"Ошибка {response.status_code}: {response.text}")
        
        except json.JSONDecodeError as e:
            st.error(f"Ошибка парсинга JSON: {str(e)}")
        
        except requests.exceptions.ConnectionError:
            st.error("Ошибка подключения: Не удается связаться с FastAPI сервером")
        
        except requests.exceptions.Timeout:
            st.error("Ошибка: Превышено время ожидания ответа")
        
        except Exception as e:
            st.error(f"Неожиданная ошибка: {str(e)}")

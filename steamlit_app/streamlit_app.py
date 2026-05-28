import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

FASTAPI_URL = "http://localhost:8000"

EXPERIENCE_OPTIONS = ["Нет опыта", "От 1 года до 3 лет", "От 3 до 6 лет", "Более 6 лет"]
EMPLOYMENT_OPTIONS = ["Полная занятость", "Частичная занятость", "Проектная работа"]
SCHEDULE_OPTIONS   = ["Полный день", "Удаленная работа", "Гибкий график", "Сменный график", "Вахтовый метод"]

def _normalize_experience(v: str) -> str:
    v = (v or '').lower().strip()
    if 'нет' in v or 'не треб' in v:                 return EXPERIENCE_OPTIONS[0]
    if '1' in v and ('3' in v or 'год' in v):         return EXPERIENCE_OPTIONS[1]
    if '3' in v and ('6' in v or 'лет' in v):         return EXPERIENCE_OPTIONS[2]
    if '6' in v or 'более' in v or 'свыше' in v:      return EXPERIENCE_OPTIONS[3]
    return EXPERIENCE_OPTIONS[0]

def _normalize_employment(v: str) -> str:
    v = (v or '').lower()
    if 'частич' in v:  return EMPLOYMENT_OPTIONS[1]
    if 'проект' in v or 'разовое' in v: return EMPLOYMENT_OPTIONS[2]
    return EMPLOYMENT_OPTIONS[0]

def _normalize_schedule(v: str) -> str:
    v = (v or '').lower()
    if 'удал' in v:    return SCHEDULE_OPTIONS[1]
    if 'гибк' in v:    return SCHEDULE_OPTIONS[2]
    if 'сменн' in v:   return SCHEDULE_OPTIONS[3]
    if 'вахт' in v:    return SCHEDULE_OPTIONS[4]
    return SCHEDULE_OPTIONS[0]

# ── Конфигурация страницы ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR AI Scout",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Шапка */
  .app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 32px 40px 24px;
    margin-bottom: 28px;
    color: white;
  }
  .app-header h1 { margin: 0; font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px; }
  .app-header p  { margin: 6px 0 0; font-size: 1.05rem; opacity: 0.75; }

  /* Карточка кандидата */
  .candidate-card {
    background: #ffffff;
    border: 1px solid #e8edf3;
    border-radius: 14px;
    padding: 20px 22px 16px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: box-shadow .2s;
  }
  .candidate-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.11); }

  /* Ранг и скор */
  .rank-badge {
    display: inline-block;
    background: #0f3460;
    color: white;
    font-weight: 700;
    border-radius: 50%;
    width: 32px; height: 32px;
    line-height: 32px;
    text-align: center;
    font-size: .85rem;
    margin-right: 10px;
    vertical-align: middle;
  }
  .score-high  { color: #1a7a4a; font-weight: 700; font-size: 1.15rem; }
  .score-mid   { color: #b58a00; font-weight: 700; font-size: 1.15rem; }
  .score-low   { color: #c0392b; font-weight: 700; font-size: 1.15rem; }

  /* Мета-строка */
  .meta { color: #6b7a8d; font-size: .88rem; margin: 6px 0 10px; }
  .meta span { margin-right: 16px; }

  /* Последняя должность */
  .last-position-block {
    background: #eef2ff;
    border-left: 4px solid #4f46e5;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin: 10px 0 10px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .last-position-label {
    font-size: .72rem;
    font-weight: 600;
    color: #4f46e5;
    text-transform: uppercase;
    letter-spacing: .6px;
  }
  .last-position-value {
    font-size: 1rem;
    font-weight: 700;
    color: #1e1b4b;
  }

  /* Навыки */
  .skill-badge {
    display: inline-block;
    background: #eef2ff;
    color: #3730a3;
    border-radius: 6px;
    padding: 2px 9px;
    font-size: .78rem;
    margin: 2px 3px 2px 0;
    font-weight: 500;
  }

  /* Секция формы */
  .form-section {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 24px 28px;
    margin-bottom: 24px;
  }

  /* Убираем лишние отступы Streamlit */
  .block-container { padding-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ── ClickHouse: детали резюме ──────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_resume_details(resume_ids: tuple) -> dict:
    """Загружает детали резюме из ClickHouse по ID."""
    try:
        from clickhouse_driver import Client
        ch = Client(
            host=os.environ.get('CLICKHOUSE_HOST', 'localhost'),
            port=int(os.environ.get('CLICKHOUSE_PORT', 9000)),
            user=os.environ.get('CLICKHOUSE_USER', 'default'),
            password=os.environ.get('CLICKHOUSE_PASSWORD', ''),
            database=os.environ.get('CLICKHOUSE_DATABASE', 'default'),
        )
        rows = ch.execute(
            "SELECT id, title, last_position, last_experience_description, "
            "skills, salary, age, experience_months, location, gender, applicant_status, url "
            "FROM hh_resumes WHERE id IN %(ids)s",
            {'ids': list(resume_ids)},
        )
        return {
            str(row[0]): {
                'title':        row[1] or '',
                'last_position': row[2] or '',
                'last_experience_description': row[3] or '',
                'skills':       row[4] if isinstance(row[4], list) else [],
                'salary':       row[5] or '',
                'age':          row[6],
                'experience_months': row[7] or 0,
                'location':     row[8] or '',
                'gender':       row[9] or '',
                'applicant_status': row[10] or '',
                'url':          row[11] or '',
            }
            for row in rows
        }
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_vacancies_from_clickhouse() -> pd.DataFrame:
    """Загружает список вакансий из ClickHouse для выбора рекрутером."""
    try:
        from clickhouse_driver import Client
        ch = Client(
            host=os.environ.get('CLICKHOUSE_HOST', 'localhost'),
            port=int(os.environ.get('CLICKHOUSE_PORT', 9000)),
            user=os.environ.get('CLICKHOUSE_USER', 'default'),
            password=os.environ.get('CLICKHOUSE_PASSWORD', ''),
            database=os.environ.get('CLICKHOUSE_DATABASE', 'default'),
        )
        rows = ch.execute(
            "SELECT id, name, area, employer, experience, employment, schedule, description "
            "FROM hh_vacancies "
            "WHERE name IS NOT NULL AND description IS NOT NULL "
            "ORDER BY parsed_date DESC "
            "LIMIT 2000"
        )
        df = pd.DataFrame(rows, columns=[
            'id', 'name', 'area', 'employer', 'experience', 'employment', 'schedule', 'description',
        ])
        df = df.fillna('')
        return df
    except Exception:
        return pd.DataFrame()


# ── Вспомогательные функции ───────────────────────────────────────────────────
def _score_class(score: float) -> str:
    if score >= 0.75:
        return "score-high"
    if score >= 0.5:
        return "score-mid"
    return "score-low"


def _months_to_str(months) -> str:
    if not months:
        return "—"
    m = int(months)
    years, rem = divmod(m, 12)
    parts = []
    if years:
        parts.append(f"{years} г.")
    if rem:
        parts.append(f"{rem} мес.")
    return " ".join(parts) or "—"


def _render_candidate_card(rank: int, resume_id: str, score: float, info: dict):
    cls = _score_class(score)
    score_pct = f"{score * 100:.1f}%"

    last_position = info.get('last_position') or '—'
    loc   = info.get('location') or '—'
    exp   = _months_to_str(info.get('experience_months'))
    age   = f"{int(info['age'])} лет" if info.get('age') else '—'
    sal   = info.get('salary') or '—'
    stat  = info.get('applicant_status') or '—'
    url   = info.get('url') or ''
    skills = info.get('skills') or []
    descr  = info.get('last_experience_description') or ''

    skills_html = ''.join(f'<span class="skill-badge">{s}</span>' for s in skills[:7])
    if len(skills) > 7:
        skills_html += f'<span class="skill-badge">+{len(skills) - 7}</span>'

    url_html = (
        f'<a href="{url}" target="_blank" style="color:#4f46e5; font-size:.82rem; '
        f'text-decoration:none; font-weight:500;">🔗 Открыть резюме на HH.ru</a>'
        if url else
        f'<span style="color:#9aa3ae; font-size:.78rem;">ID: {resume_id}</span>'
    )

    card_html = f"""
    <div class="candidate-card">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span class="rank-badge">#{rank}</span>
        <span class="{cls}">{score_pct}</span>
      </div>
      <div class="last-position-block">
        <span class="last-position-label">Последняя должность</span>
        <span class="last-position-value">{last_position}</span>
      </div>
      <div class="meta">
        <span>📍 {loc}</span>
        <span>💼 {exp}</span>
        <span>🎂 {age}</span>
        <span>💰 {sal}</span>
        <span>🔔 {stat}</span>
      </div>
      <div>{skills_html}</div>
      <div style="margin-top:10px;">{url_html}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    if descr:
        with st.expander("Описание последнего места работы", expanded=False):
            st.write(descr[:2000] + ("…" if len(descr) > 2000 else ""))


# ── Шапка ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🔍 HR AI Scout</h1>
  <p>Умный подбор кандидатов с помощью машинного обучения</p>
</div>
""", unsafe_allow_html=True)


# ── Выбор режима ввода вакансии ───────────────────────────────────────────────
st.markdown("### Вакансия")

tab_new, tab_db = st.tabs(["✏️ Новая вакансия", "🗃️ Выбрать из базы"])

# Prefill defaults — заполняются при выборе вакансии из базы
pf = st.session_state.get('prefill', {})

with tab_db:
    vacancies_df = load_vacancies_from_clickhouse()
    if vacancies_df.empty:
        st.warning("Не удалось загрузить вакансии из ClickHouse.")
    else:
        search_q = st.text_input(
            "🔎 Поиск по названию или работодателю",
            placeholder="например: Python, аналитик, Яндекс…",
            key="vac_search",
        )
        if search_q.strip():
            mask = (
                vacancies_df['name'].str.contains(search_q, case=False, na=False) |
                vacancies_df['employer'].str.contains(search_q, case=False, na=False)
            )
            filtered_df = vacancies_df[mask]
        else:
            filtered_df = vacancies_df

        if filtered_df.empty:
            st.info("Вакансии не найдены. Измените запрос.")
        else:
            options = [
                f"{row['name']}  |  {row['employer'] or '—'}  |  {row['area'] or '—'}  [ID: {row['id']}]"
                for _, row in filtered_df.iterrows()
            ]
            selected_idx = st.selectbox(
                f"Найдено вакансий: {len(filtered_df)}",
                range(len(options)),
                format_func=lambda i: options[i],
                key="vac_select",
            )
            if st.button("Использовать эту вакансию →", type="secondary"):
                row = filtered_df.iloc[selected_idx]
                st.session_state['prefill'] = {
                    'name':        row['name'] or '',
                    'area':        row['area'] or 'Москва',
                    'experience':  _normalize_experience(row['experience']),
                    'employment':  _normalize_employment(row['employment']),
                    'schedule':    _normalize_schedule(row['schedule']),
                    'description': row['description'] or '',
                    'id':          str(row['id']),
                }
                pf = st.session_state['prefill']
                st.success(f"Вакансия «{row['name']}» загружена. Перейдите во вкладку **✏️ Новая вакансия**.")

# ── Форма вакансии ────────────────────────────────────────────────────────────
with tab_new:
    if pf:
        st.info(
            f"📋 Заполнено из базы: **{pf.get('name', '')}** (ID: {pf.get('id', '—')}). "
            "Поля можно отредактировать.",
            icon="ℹ️",
        )

    with st.form("vacancy_form", clear_on_submit=False):
        col_left, col_right = st.columns([3, 2])

        with col_left:
            vacancy_name = st.text_input(
                "Название вакансии",
                value=pf.get('name', ''),
                placeholder="например: Python-разработчик",
            )
            vacancy_area = st.text_input(
                "Город *",
                value=pf.get('area', 'Москва'),
                placeholder="Москва",
            )
            vacancy_description = st.text_area(
                "Описание вакансии *",
                value=pf.get('description', ''),
                placeholder="Опишите задачи, требования и условия работы…",
                height=220,
            )

        with col_right:
            exp_default  = EXPERIENCE_OPTIONS.index(pf['experience'])  if pf.get('experience')  in EXPERIENCE_OPTIONS  else 0
            empl_default = EMPLOYMENT_OPTIONS.index(pf['employment']) if pf.get('employment') in EMPLOYMENT_OPTIONS else 0
            sch_default  = SCHEDULE_OPTIONS.index(pf['schedule'])     if pf.get('schedule')    in SCHEDULE_OPTIONS    else 0

            vacancy_experience = st.selectbox("Требуемый опыт *", EXPERIENCE_OPTIONS, index=exp_default)
            vacancy_employment = st.selectbox("Тип занятости *",  EMPLOYMENT_OPTIONS, index=empl_default)
            vacancy_schedule   = st.selectbox("График работы *",  SCHEDULE_OPTIONS,   index=sch_default)
            vacancy_id = st.text_input(
                "ID вакансии на HH.ru",
                value=pf.get('id', ''),
                placeholder="например: 126167948",
                help="Заполняется автоматически при выборе из базы. "
                     "Повышает точность за счёт ALS-модели.",
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Заполните **Город** и **Описание вакансии**, затем нажмите кнопку ниже.")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔍 Подобрать кандидатов",
            type="primary",
            use_container_width=True,
        )


# ── Инициализация session_state ───────────────────────────────────────────────
for _key in ('search_results', 'search_vacancy', 'search_details', 'decisions'):
    if _key not in st.session_state:
        st.session_state[_key] = None if _key != 'decisions' else {}


def _send_decision(rid: str, target: int):
    """Отправляет решение рекрутера на /decision и обновляет session_state."""
    vac = st.session_state['search_vacancy'] or {}
    payload = {
        "vacancy_area":        vac.get('vacancy_area', ''),
        "vacancy_experience":  vac.get('vacancy_experience', ''),
        "vacancy_employment":  vac.get('vacancy_employment', ''),
        "vacancy_schedule":    vac.get('vacancy_schedule', ''),
        "vacancy_description": vac.get('vacancy_description', ''),
        "resume_id":           rid,
        "target":              target,
    }
    if vac.get('vacancy_name'):
        payload['vacancy_name'] = vac['vacancy_name']
    if vac.get('vacancy_id'):
        payload['vacancy_id'] = vac['vacancy_id']
    try:
        r = requests.post(f"{FASTAPI_URL}/decision", json=payload, timeout=10)
        if r.status_code == 200:
            st.session_state['decisions'][rid] = target
        else:
            st.error(f"Ошибка сохранения решения: {r.text}")
    except Exception as e:
        st.error(f"Ошибка подключения: {e}")


# ── Результаты ────────────────────────────────────────────────────────────────
if submitted:
    errors = []
    if not vacancy_area.strip():
        errors.append("Укажите город.")
    elif not all(c.isalpha() or c in ' -()' for c in vacancy_area) or not any('Ѐ' <= c <= 'ӿ' for c in vacancy_area):
        errors.append("Город должен быть написан кириллицей.")
    if not vacancy_description.strip():
        errors.append("Заполните описание вакансии.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        payload = {
            "vacancy_area":        vacancy_area.strip(),
            "vacancy_experience":  vacancy_experience,
            "vacancy_employment":  vacancy_employment,
            "vacancy_schedule":    vacancy_schedule,
            "vacancy_description": vacancy_description.strip(),
        }
        if vacancy_name.strip():
            payload["vacancy_name"] = vacancy_name.strip()
        if vacancy_id.strip():
            payload["vacancy_id"] = vacancy_id.strip()

        with st.spinner("Оцениваем кандидатов из базы резюме…"):
            try:
                resp = requests.post(f"{FASTAPI_URL}/forward", json=payload, timeout=180)
            except requests.exceptions.ConnectionError:
                st.error("Нет подключения к API-серверу (http://localhost:8000).")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Превышено время ожидания ответа от API.")
                st.stop()

        if resp.status_code != 200:
            st.error(f"Ошибка API ({resp.status_code}): {resp.text}")
            st.stop()

        results = resp.json()
        if not results:
            st.warning("Кандидаты не найдены.")
            st.stop()

        with st.spinner("Загружаем профили кандидатов…"):
            details_map = get_resume_details(tuple(r["resume_id"] for r in results))

        # Сохраняем в session_state, сбрасываем решения
        st.session_state['search_results']  = results
        st.session_state['search_vacancy']  = payload
        st.session_state['search_details']  = details_map
        st.session_state['decisions']       = {}


# Отображаем результаты из session_state (сохраняются между кликами кнопок)
if st.session_state['search_results']:
    results     = st.session_state['search_results']
    details_map = st.session_state['search_details'] or {}
    vac         = st.session_state['search_vacancy'] or {}

    st.markdown("---")
    vac_display = f'«{vac.get("vacancy_name", "")}»' if vac.get("vacancy_name") else f'«{vac.get("vacancy_area", "")}»'
    st.markdown(f"### Топ-{len(results)} кандидатов для вакансии {vac_display}")

    scores = [r["y_pred_proba"] for r in results]
    decided = len(st.session_state['decisions'])
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Найдено кандидатов", len(results))
    mc2.metric("Лучший скор",        f"{max(scores) * 100:.1f}%")
    mc3.metric("Средний скор",       f"{np.mean(scores) * 100:.1f}%")
    mc4.metric("Решений принято",    f"{decided}/{len(results)}")

    st.markdown("<br>", unsafe_allow_html=True)

    for rank, item in enumerate(results, start=1):
        rid   = item["resume_id"]
        score = item["y_pred_proba"]
        info  = details_map.get(rid, {})

        # Карточка кандидата
        _render_candidate_card(rank, rid, score, info)

        # Кнопки решения
        decision = st.session_state['decisions'].get(rid)
        if decision is None:
            btn_col1, btn_col2, _ = st.columns([1, 1, 3])
            with btn_col1:
                if st.button("✅ Пригласить", key=f"invite_{rid}", use_container_width=True):
                    _send_decision(rid, 1)
                    st.rerun()
            with btn_col2:
                if st.button("❌ Отклонить", key=f"reject_{rid}", use_container_width=True):
                    _send_decision(rid, 0)
                    st.rerun()
        elif decision == 1:
            st.success("✅ Приглашён на собеседование — решение сохранено")
        else:
            st.error("❌ Отклонён — решение сохранено")

        st.markdown("<br>", unsafe_allow_html=True)

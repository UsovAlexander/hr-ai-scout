import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional
import pickle
import pandas as pd
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel, Field, field_validator, ConfigDict
import scipy.sparse as sp
from sklearn.preprocessing import normalize as sk_normalize

# Загружаем .env из корня проекта (на уровень выше app/)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
from sqlalchemy import DateTime, Float, Integer, String, delete, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from modules import (
    tokenize_and_lemmatize,
    compute_location_matching,
    resume_skill_count_in_vacancy,
    last_position_in_vacancy,
    encode_texts,
    experience_to_months,
    parse_salary,
    FEATURES,
)

SECRET_KEY = os.environ.get("KEY", "test")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = f"sqlite+aiosqlite:///{BASE_DIR}/hr_scout_history.db"

PIPELINE_PATH    = BASE_DIR / "pipeline_cb_als_sim_labse_en_ru.pkl"
VECTORIZER_PATH  = BASE_DIR / "tfidf_vectorizer.pkl"
ALS_PATH         = BASE_DIR / "als_model.pkl"
TFIDF_CACHE_PATH = BASE_DIR / "resume_tfidf_cache.npz"
TFIDF_META_PATH  = BASE_DIR / "resume_tfidf_cache_meta.pkl"

LABSE_MODEL_NAME = "cointegrated/LaBSE-en-ru"

engine: AsyncEngine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False,
    autocommit=False, autoflush=False,
)


class Base(DeclarativeBase):
    pass


class PredictionHistory(Base):
    __tablename__ = "predictHist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    resume_id: Mapped[str] = mapped_column(String(100), nullable=False)
    vacancy_area: Mapped[str] = mapped_column(String(200), nullable=False)
    vacancy_experience: Mapped[str] = mapped_column(String(100), nullable=False)
    vacancy_employment: Mapped[str] = mapped_column(String(100), nullable=False)
    vacancy_schedule: Mapped[str] = mapped_column(String(100), nullable=False)
    resume_location: Mapped[str] = mapped_column(String(200), nullable=False)
    resume_gender: Mapped[str] = mapped_column(String(50), nullable=False)
    resume_applicant_status: Mapped[str] = mapped_column(String(100), nullable=False)
    resume_salary: Mapped[float] = mapped_column(Float, nullable=False)
    resume_age: Mapped[float] = mapped_column(Float, nullable=False)
    resume_experience_months: Mapped[float] = mapped_column(Float, nullable=False)
    resume_last_company_experience_months: Mapped[float] = mapped_column(Float, nullable=False)
    location_matching: Mapped[float] = mapped_column(Float, nullable=False)
    resume_skill_count_in_vacancy: Mapped[float] = mapped_column(Float, nullable=False)
    last_position_in_vacancy: Mapped[float] = mapped_column(Float, nullable=False)
    similarity_score_tfidf: Mapped[float] = mapped_column(Float, nullable=False)
    als_score: Mapped[float] = mapped_column(Float, nullable=False)
    sim_labse_en_ru: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    processing_time: Mapped[float] = mapped_column(Float, nullable=False)

    def __repr__(self) -> str:
        return f"<PredictionHistory(id={self.id}, prediction={self.prediction})>"


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ── Вспомогательные функции для загрузки данных ───────────────────────────────

def _get_clickhouse_client():
    from clickhouse_driver import Client
    return Client(
        host=os.environ.get('CLICKHOUSE_HOST', 'localhost'),
        port=int(os.environ.get('CLICKHOUSE_PORT', 9000)),
        user=os.environ.get('CLICKHOUSE_USER', 'default'),
        password=os.environ.get('CLICKHOUSE_PASSWORD', ''),
        database=os.environ.get('CLICKHOUSE_DATABASE', 'default'),
    )


def _load_resumes_from_clickhouse() -> pd.DataFrame:
    """Загружает резюме из ClickHouse и возвращает DataFrame с нужными колонками."""
    ch = _get_clickhouse_client()
    print("Загружаю резюме из ClickHouse...")
    rows = ch.execute(
        "SELECT id, last_position, last_experience_description, "
        "last_company_experience_period, skills, salary, age, "
        "experience_months, location, gender, applicant_status "
        "FROM hh_resumes ORDER BY id"
    )
    df = pd.DataFrame(rows, columns=[
        'resume_id', 'resume_last_position', 'last_experience_description',
        'last_company_experience_period', 'resume_skills', 'resume_salary_raw',
        'resume_age', 'resume_experience_months', 'resume_location',
        'resume_gender', 'resume_applicant_status',
    ])
    print(f"  Загружено {len(df):,} резюме")

    df['resume_salary'] = df['resume_salary_raw'].fillna('').apply(parse_salary)
    df['resume_last_company_experience_months'] = (
        df['last_company_experience_period'].fillna('').apply(
            lambda x: experience_to_months(x) if x else 0.0
        ).fillna(0.0)
    )
    df['resume_age'] = df['resume_age'].fillna(df['resume_age'].mean())
    df['resume_experience_months'] = df['resume_experience_months'].fillna(0)
    df['resume_location'] = df['resume_location'].fillna('NDT')
    df['resume_applicant_status'] = df['resume_applicant_status'].fillna('NDT')
    df['resume_last_position'] = df['resume_last_position'].fillna('')
    df['last_experience_description'] = df['last_experience_description'].fillna('')

    gender_map = {
        'Мужчина': 'Мужчина', 'Male': 'Мужчина',
        'Женщина': 'Женщина', 'Female': 'Женщина',
    }
    df['resume_gender'] = df['resume_gender'].apply(lambda x: gender_map.get(x, 'Неизвестно'))

    df['resume_skills'] = df['resume_skills'].apply(
        lambda x: x if isinstance(x, list) else str(x) if x else ''
    )
    return df.drop(columns=['resume_salary_raw', 'last_company_experience_period'])


def _build_tfidf_matrix(df_resumes: pd.DataFrame, vectorizer):
    """
    Строит нормализованную TF-IDF матрицу для резюме.
    Использует кеш чтобы не пересчитывать при перезапуске.
    """
    resume_ids = df_resumes['resume_id'].tolist()
    ids_hash = hash(tuple(sorted(resume_ids)))

    if TFIDF_CACHE_PATH.exists() and TFIDF_META_PATH.exists():
        with open(TFIDF_META_PATH, 'rb') as f:
            meta = pickle.load(f)
        if meta.get('ids_hash') == ids_hash:
            print("  TF-IDF матрица загружена из кеша")
            return sp.load_npz(str(TFIDF_CACHE_PATH))

    print(f"  Вычисляю TF-IDF матрицу для {len(resume_ids):,} резюме...")
    texts = df_resumes['last_experience_description'].tolist()

    if getattr(vectorizer, 'tokenizer', None) is not None:
        # Векторайзер сам лемматизирует через tokenize_and_lemmatize
        matrix = vectorizer.transform(texts)
    else:
        # Векторайзер без tokenizer — нужно предварительно лемматизировать
        preprocessed = [' '.join(tokenize_and_lemmatize(t)) for t in texts]
        matrix = vectorizer.transform(preprocessed)

    matrix = sk_normalize(matrix, norm='l2')
    sp.save_npz(str(TFIDF_CACHE_PATH), matrix)
    with open(TFIDF_META_PATH, 'wb') as f:
        pickle.dump({'ids_hash': ids_hash}, f)
    print(f"  TF-IDF матрица сохранена в кеш: {TFIDF_CACHE_PATH}")
    return matrix


def _load_labse_embeddings_from_clickhouse(resume_ids: list) -> dict:
    """Загружает LaBSE-эмбеддинги из ClickHouse батчами (обход лимита max_query_size)."""
    ch = _get_clickhouse_client()
    print(f"Загружаю LaBSE-эмбеддинги из ClickHouse ({len(resume_ids):,} резюме)...")
    str_ids = [str(i) for i in resume_ids]
    result = {}
    batch_size = 5000
    for i in range(0, len(str_ids), batch_size):
        batch = str_ids[i: i + batch_size]
        rows = ch.execute(
            "SELECT resume_id, embedding FROM resume_embeddings "
            "WHERE model_name = %(m)s AND resume_id IN %(ids)s",
            {'m': LABSE_MODEL_NAME, 'ids': batch},
        )
        for row in rows:
            result[row[0]] = np.array(row[1], dtype=np.float32)
    print(f"  Загружено {len(result):,} LaBSE-эмбеддингов")
    return result


def _build_labse_matrix(df_resumes: pd.DataFrame, labse_map: dict):
    """Строит упорядоченную матрицу LaBSE-эмбеддингов (N, dim) и индексы."""
    resume_ids = df_resumes['resume_id'].tolist()
    indices, embeddings = [], []
    for i, rid in enumerate(resume_ids):
        emb = labse_map.get(str(rid))
        if emb is not None:
            indices.append(i)
            embeddings.append(emb)
    if embeddings:
        matrix = np.stack(embeddings, axis=0)  # (M, dim)
    else:
        matrix = None
    return matrix, indices


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    import torch
    from transformers import AutoTokenizer, AutoModel

    # ── Модель и артефакты ─────────────────────────────────────────────────────
    print(f"Загружаю пайплайн: {PIPELINE_PATH.name}")
    with open(PIPELINE_PATH, 'rb') as f:
        app.state.model = pickle.load(f)

    print(f"Загружаю TF-IDF векторайзер: {VECTORIZER_PATH.name}")
    with open(VECTORIZER_PATH, 'rb') as f:
        app.state.tfidf_vectorizer = pickle.load(f)

    print(f"Загружаю ALS-модель: {ALS_PATH.name}")
    with open(ALS_PATH, 'rb') as f:
        als_raw = pickle.load(f)
    # Нормализуем ключи к строкам — в pkl они int, в ClickHouse string
    app.state.als_artifact = {
        'vac2id':       {str(k): v for k, v in als_raw['vac2id'].items()},
        'res2id':       {str(k): v for k, v in als_raw['res2id'].items()},
        'vac_factors':  als_raw['vac_factors'],
        'res_factors':  als_raw['res_factors'],
    }

    # ── Данные из ClickHouse ───────────────────────────────────────────────────
    app.state.df_resumes = _load_resumes_from_clickhouse()

    # ── TF-IDF матрица резюме ──────────────────────────────────────────────────
    app.state.resume_tfidf_matrix = _build_tfidf_matrix(
        app.state.df_resumes, app.state.tfidf_vectorizer
    )

    # ── LaBSE эмбеддинги ──────────────────────────────────────────────────────
    resume_ids = app.state.df_resumes['resume_id'].tolist()
    labse_map = _load_labse_embeddings_from_clickhouse(resume_ids)
    app.state.resume_labse_matrix, app.state.resume_labse_indices = \
        _build_labse_matrix(app.state.df_resumes, labse_map)

    # ── ALS матрица резюме (предвычисляем для быстрого инференса) ────────────
    print("Строю ALS-матрицу резюме...")
    als = app.state.als_artifact
    resume_ids_list = app.state.df_resumes['resume_id'].tolist()
    n_factors = als['res_factors'].shape[1]
    als_res_matrix = np.zeros((len(resume_ids_list), n_factors), dtype=np.float32)
    for i, rid in enumerate(resume_ids_list):
        idx = als['res2id'].get(str(rid))
        if idx is not None:
            als_res_matrix[i] = als['res_factors'][idx]
    app.state.als_res_matrix = als_res_matrix
    known = int(np.any(als_res_matrix != 0, axis=1).sum())
    print(f"  ALS: {known:,} из {len(resume_ids_list):,} резюме имеют факторы")

    # ── LaBSE модель ──────────────────────────────────────────────────────────
    print(f"Загружаю LaBSE модель: {LABSE_MODEL_NAME}")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    app.state.labse_device = device
    app.state.labse_tokenizer = AutoTokenizer.from_pretrained(LABSE_MODEL_NAME)
    app.state.labse_model = AutoModel.from_pretrained(LABSE_MODEL_NAME).to(device).eval()
    print(f"  LaBSE загружена на {device}")

    # ── SQLite БД ──────────────────────────────────────────────────────────────
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Миграция: добавляем новые колонки если их нет (для существующих БД)
        for col, dtype in [('als_score', 'REAL'), ('sim_labse_en_ru', 'REAL')]:
            try:
                await conn.execute(
                    text(f"ALTER TABLE predictHist ADD COLUMN {col} {dtype} NOT NULL DEFAULT 0.0")
                )
            except Exception:
                pass  # колонка уже существует

    print("=== Сервис готов ===")
    yield
    await engine.dispose()


app = FastAPI(
    title="HR AI Scout",
    description="Веб-сервис для автоматического анализа и скоринга кандидатов по резюме и вакансиям",
    version="0.2",
    lifespan=lifespan,
)


# ── Enums / Pydantic схемы ────────────────────────────────────────────────────

class ResumeGender(str, Enum):
    male = "Мужчина"
    female = "Женщина"
    unknown = "Неизвестно"


class ResumeApplicantStatus(str, Enum):
    active = "Активно ищет работу"
    passive = "Рассматривает предложения"


class VacancyEmployment(str, Enum):
    full_time = "Полная занятость"
    part_time = "Частичная занятость"
    project = "Проектная работа"


class VacancyExperience(str, Enum):
    no_exp = "Нет опыта"
    from_1_to_3 = "От 1 года до 3 лет"
    from_3_to_6 = "От 3 до 6 лет"
    more_than_6 = "Более 6 лет"


class VacancySchedule(str, Enum):
    full_day = "Полный день"
    remote = "Удаленная работа"
    flex = "Гибкий график"
    shift = "Сменный график"
    vaht = "Вахтовый метод"


class VacancyInput(BaseModel):
    vacancy_name: Optional[str] = Field(None, description="Название вакансии")
    vacancy_area: str = Field(..., description="Город вакансии")
    vacancy_experience: VacancyExperience = Field(..., description="Требуемый опыт")
    vacancy_employment: VacancyEmployment = Field(..., description="Тип занятости")
    vacancy_schedule: VacancySchedule = Field(..., description="График работы")
    vacancy_description: str = Field(..., description="Описание вакансии")
    vacancy_id: Optional[str] = Field(None, description="ID вакансии с HH.ru (для ALS-скора)")

    @field_validator('vacancy_area')
    @classmethod
    def check_cyrillic_only(cls, v: str, info) -> str:
        if not re.match(r'^[а-яА-ЯёЁ\s\-()]+$', v):
            raise ValueError(f'{info.field_name} можно написать только на кириллице')
        return v

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "vacancy_name": "Аналитик данных/data analyst",
            "vacancy_area": "Москва",
            "vacancy_experience": "От 1 года до 3 лет",
            "vacancy_employment": "Полная занятость",
            "vacancy_schedule": "Полный день",
            "vacancy_description": "Твои задачи: Формирование баз данных, аналитика, SQL, Python",
        }
    })


class TopResume(BaseModel):
    resume_id: str
    y_pred_proba: float

    model_config = ConfigDict(json_schema_extra={"example": {"resume_id": "116651504", "y_pred_proba": 0.999}})


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    is_admin: Optional[bool] = False


class User(BaseModel):
    username: str
    is_admin: bool = False


class UserInDB(User):
    password: str


fake_users_db: Dict[str, Dict] = {
    "test": {"username": "test", "password": "test", "is_admin": True}
}


def verify_password(plain_password, stored_password):
    return plain_password == stored_password


def get_user(db, username: str):
    user = db.get(username)
    if not user:
        return None
    return UserInDB(**user)


def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user or not verify_password(password, user.password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        is_admin = payload.get("is_admin", False)
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, is_admin=is_admin)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_admin_user(current_user: User = Depends(get_current_user)):
    if not getattr(current_user, "is_admin", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    return current_user


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.post("/token", response_model=Token, tags=["Auth"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username, "is_admin": user.is_admin},
        expires_delta=timedelta(minutes=30),
    )
    return {"access_token": access_token, "token_type": "bearer"}


# ── Pydantic схема истории ─────────────────────────────────────────────────────

class HistoryResponse(BaseModel):
    id: int
    created_at: datetime
    resume_id: str
    vacancy_area: str
    vacancy_experience: str
    vacancy_employment: str
    vacancy_schedule: str
    resume_location: str
    resume_gender: str
    resume_applicant_status: str
    resume_salary: float
    resume_age: float
    resume_experience_months: float
    resume_last_company_experience_months: float
    location_matching: float
    resume_skill_count_in_vacancy: float
    last_position_in_vacancy: float
    similarity_score_tfidf: float
    als_score: float
    sim_labse_en_ru: float
    prediction: int
    probability: float

    model_config = ConfigDict(from_attributes=True)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request, _exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "модель не смогла обработать данные"},
    )


# ── Эндпоинты ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "HR AI Scout",
        "version": "0.2",
        "status": "running",
        "model": "pipeline_cb_als_sim_labse_en_ru",
        "documentation": "/docs",
    }


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    model_loaded = hasattr(app.state, 'model')
    return {"status": "healthy" if model_loaded else "unhealthy", "model_loaded": model_loaded}


@app.get("/validation/demo", tags=["Documentation"])
async def validation_demo():
    return {
        "message": "Правила отправки значений для POST /forward",
        "description": "API принимает описание вакансии и возвращает топ-10 подходящих резюме из базы",
        "input_fields": {
            "vacancy_name": {"type": "string", "required": False},
            "vacancy_area": {"type": "string", "required": True, "constraint": "Только кириллица"},
            "vacancy_experience": {
                "type": "enum", "required": True,
                "allowed_values": ["Нет опыта", "От 1 года до 3 лет", "От 3 до 6 лет", "Более 6 лет"],
            },
            "vacancy_employment": {
                "type": "enum", "required": True,
                "allowed_values": ["Полная занятость", "Частичная занятость", "Проектная работа"],
            },
            "vacancy_schedule": {
                "type": "enum", "required": True,
                "allowed_values": ["Полный день", "Удаленная работа", "Гибкий график", "Сменный график", "Вахтовый метод"],
            },
            "vacancy_description": {"type": "string", "required": True},
        },
        "output_format": {
            "type": "array",
            "items": {"resume_id": "integer", "y_pred_proba": "float 0-1"},
            "description": "Топ-10 резюме, отсортированных по убыванию вероятности",
        },
    }


@app.post("/forward", response_model=List[TopResume], tags=["Prediction"])
async def forward(data: VacancyInput, db: AsyncSession = Depends(get_db)):
    try:
        start_time = time.time()

        if not hasattr(app.state, 'model') or app.state.model is None:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail="Модель не загружена")

        # ── Формируем датафрейм вакансия × резюме ─────────────────────────────
        input_dict = {
            'vacancy_name':        data.vacancy_name,
            'vacancy_area':        data.vacancy_area,
            'vacancy_experience':  data.vacancy_experience.value,
            'vacancy_employment':  data.vacancy_employment.value,
            'vacancy_schedule':    data.vacancy_schedule.value,
            'vacancy_description': data.vacancy_description,
        }
        df_vacancy = pd.DataFrame([input_dict])
        df = df_vacancy.merge(app.state.df_resumes, how='cross')

        # ── Ручные признаки ───────────────────────────────────────────────────
        df['location_matching'] = df.apply(compute_location_matching, axis=1)
        df['resume_skill_count_in_vacancy'] = df.apply(resume_skill_count_in_vacancy, axis=1)
        df['last_position_in_vacancy'] = df.apply(last_position_in_vacancy, axis=1)

        # ── TF-IDF сходство ───────────────────────────────────────────────────
        vac_text = data.vacancy_description
        vectorizer = app.state.tfidf_vectorizer
        if getattr(vectorizer, 'tokenizer', None) is not None:
            vacancy_tfidf = sk_normalize(vectorizer.transform([vac_text]), norm='l2')
        else:
            preprocessed_vac = ' '.join(tokenize_and_lemmatize(vac_text))
            vacancy_tfidf = sk_normalize(vectorizer.transform([preprocessed_vac]), norm='l2')

        tfidf_scores = vacancy_tfidf.dot(app.state.resume_tfidf_matrix.T).toarray().flatten()
        df['similarity_score_tfidf'] = tfidf_scores

        # ── LaBSE сходство ────────────────────────────────────────────────────
        vacancy_labse = encode_texts(
            [vac_text],
            tokenizer=app.state.labse_tokenizer,
            model=app.state.labse_model,
            batch_size=1,
            device=app.state.labse_device,
        )[0]  # (dim,)

        labse_scores = np.zeros(len(df), dtype=np.float32)
        if app.state.resume_labse_matrix is not None:
            scores = app.state.resume_labse_matrix @ vacancy_labse  # (M,)
            for i, idx in enumerate(app.state.resume_labse_indices):
                labse_scores[idx] = scores[i]
        df['sim_labse_en_ru'] = labse_scores

        # ── ALS-скор ──────────────────────────────────────────────────────────
        als = app.state.als_artifact
        v_id = str(data.vacancy_id) if data.vacancy_id else None
        if v_id and v_id in als['vac2id']:
            vac_factor = als['vac_factors'][als['vac2id'][v_id]]  # (n_factors,)
            als_scores = app.state.als_res_matrix @ vac_factor    # (N,)
        else:
            als_scores = np.zeros(len(df), dtype=np.float32)
        df['als_score'] = als_scores

        # ── Предсказание ──────────────────────────────────────────────────────
        X = df[FEATURES]
        y_pred = app.state.model.predict(X)
        y_pred_proba = app.state.model.predict_proba(X)[:, 1]

        df['y_pred'] = y_pred
        df['y_pred_proba'] = y_pred_proba

        processing_time = time.time() - start_time
        top = df.sort_values('y_pred_proba', ascending=False).head(10)

        # ── Сохранение в историю ──────────────────────────────────────────────
        for _, row in top.iterrows():
            db.add(PredictionHistory(
                resume_id=str(row.get('resume_id', '')),
                vacancy_area=row['vacancy_area'],
                vacancy_experience=row['vacancy_experience'],
                vacancy_employment=row['vacancy_employment'],
                vacancy_schedule=row['vacancy_schedule'],
                resume_location=row.get('resume_location', ''),
                resume_gender=row.get('resume_gender', ''),
                resume_applicant_status=row.get('resume_applicant_status', ''),
                resume_salary=float(row.get('resume_salary') or 0.0),
                resume_age=float(row.get('resume_age') or 0.0),
                resume_experience_months=float(row.get('resume_experience_months') or 0.0),
                resume_last_company_experience_months=float(row.get('resume_last_company_experience_months') or 0.0),
                location_matching=float(row.get('location_matching') or 0.0),
                resume_skill_count_in_vacancy=float(row.get('resume_skill_count_in_vacancy') or 0.0),
                last_position_in_vacancy=float(row.get('last_position_in_vacancy') or 0.0),
                similarity_score_tfidf=float(row.get('similarity_score_tfidf') or 0.0),
                als_score=float(row.get('als_score') or 0.0),
                sim_labse_en_ru=float(row.get('sim_labse_en_ru') or 0.0),
                prediction=int(row.get('y_pred') or 0),
                probability=float(row.get('y_pred_proba') or 0.0),
                processing_time=processing_time,
            ))
        await db.commit()

        return [TopResume(resume_id=str(r.resume_id), y_pred_proba=float(r.y_pred_proba))
                for _, r in top.iterrows()]

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"модель не смогла обработать данные: {type(exc).__name__}: {exc}",
        )


@app.get("/history", response_model=List[HistoryResponse], tags=["History"])
async def get_history(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_active_admin_user),
):
    result = await db.execute(
        select(PredictionHistory)
        .order_by(PredictionHistory.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all())


@app.delete("/history", status_code=status.HTTP_204_NO_CONTENT, tags=["History"])
async def delete_history(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_active_admin_user),
):
    await db.execute(delete(PredictionHistory))
    await db.commit()
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)


@app.get("/stats", tags=["Statistics"])
async def get_stats(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_active_admin_user),
):
    result = await db.execute(select(PredictionHistory))
    records = result.scalars().all()

    if not records:
        return {"total_requests": 0, "message": "No data"}

    processing_times = [r.processing_time for r in records]
    return {
        "total_requests": len(records),
        "processing_time_seconds": {
            "mean": float(np.mean(processing_times)),
            "median_p50": float(np.percentile(processing_times, 50)),
            "p95": float(np.percentile(processing_times, 95)),
            "p99": float(np.percentile(processing_times, 99)),
            "min": float(np.min(processing_times)),
            "max": float(np.max(processing_times)),
        },
        "input_characteristics": {
            "resume_age": {
                "mean": float(np.mean([r.resume_age for r in records])),
                "median": float(np.median([r.resume_age for r in records])),
            },
            "resume_salary": {
                "mean": float(np.mean([r.resume_salary for r in records])),
                "median": float(np.median([r.resume_salary for r in records])),
            },
            "resume_experience_months": {
                "mean": float(np.mean([r.resume_experience_months for r in records])),
                "median": float(np.median([r.resume_experience_months for r in records])),
            },
            "similarity_score_tfidf": {
                "mean": float(np.mean([r.similarity_score_tfidf for r in records])),
                "median": float(np.median([r.similarity_score_tfidf for r in records])),
            },
            "sim_labse_en_ru": {
                "mean": float(np.mean([r.sim_labse_en_ru for r in records])),
                "median": float(np.median([r.sim_labse_en_ru for r in records])),
            },
        },
        "predictions": {
            "total_predictions": len(records),
            "positive_predictions": sum(1 for r in records if r.prediction == 1),
            "negative_predictions": sum(1 for r in records if r.prediction == 0),
        },
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")

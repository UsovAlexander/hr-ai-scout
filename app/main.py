import os
import re
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional
import pickle
import pandas as pd
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import ast
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from modules import (
    tokenize_and_lemmatize,
    compute_location_matching,
    resume_skill_count_in_vacancy,
    last_position_in_vacancy,
    compute_similarity_features,
    FEATURES
)
from sqlalchemy import DateTime, Float, Integer, String, delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SECRET_KEY = os.environ.get("KEY", "test")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = f"sqlite+aiosqlite:///{BASE_DIR}/hr_scout_history.db"

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

class Base(DeclarativeBase):
    pass

class PredictionHistory(Base):
    __tablename__ = "predictHist"
    
    id: Mapped[int] = mapped_column(Integer,primary_key=True,index=True,autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime,default=datetime.now,nullable=False)
    resume_id: Mapped[int] = mapped_column(Integer, nullable=False)
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
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    processing_time: Mapped[float] = mapped_column(Float, nullable=False)


    def __repr__(self) -> str:
        return f"<PredictionHistory(id={self.id}, prediction={self.prediction}, created_at={self.created_at})>"

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    base_dir = Path(__file__).resolve().parent

    model_path = base_dir / "feature_engineering_with_best_optuna_lr.pkl"
    tfidf_path = base_dir / "tfidf_vectorizer.pkl"
    embeddings_path = base_dir / "experience_embeddings.npz"
    resumes_path = base_dir / "df_resumes.csv"

    with open(model_path, 'rb') as f:
        app.state.model = pickle.load(f)

    with open(tfidf_path, 'rb') as f:
        app.state.tfidf_vectorizer = pickle.load(f)

    app.state.experience_embeddings = sp.load_npz(str(embeddings_path))

    app.state.df_resumes = pd.read_csv(resumes_path)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    await engine.dispose()

app = FastAPI(
    title="HR AI Scout",
    description="Веб-сервис для автоматического анализа и скоринга кандидатов на работу по резюме и агрегированным данным из различных источников",
    version="0.1",
    lifespan=lifespan
)

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
    full_dya = "Полный день"
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

    @field_validator('vacancy_area')
    @classmethod
    def check_cyrillic_only(cls, v: str, info) -> str:
        if not re.match(r'^[а-яА-ЯёЁ\s\-()]+$', v):
            raise ValueError(f'{info.field_name} можно написать только на кириллице')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "vacancy_name": "Аналитик данных/data analyst",
                "vacancy_area": "Москва",
                "vacancy_experience": "От 1 года до 3 лет",
                "vacancy_employment": "Полная занятость",
                "vacancy_schedule": "Полный день",
                "vacancy_description": "Твои задачи: Формирование баз данных, аналитика, SQL, Python"
            }
        }


class TopResume(BaseModel):
    resume_id: int
    y_pred_proba: float

    class Config:
        schema_extra = {"example": {"resume_id": 116651504, "y_pred_proba": 0.999}}


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Подойдет ли на собеседование (1) или нет (0)")
    probability: float = Field(..., description="Вероятность")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.85
            }
        }


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
    "test": {
        "username": "test",
        "password": "test",
        "is_admin": True
    }
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
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt


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


@app.post("/token", response_model=Token, tags=["Auth"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username, "is_admin": user.is_admin},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


class HistoryResponse(BaseModel):
    id: int
    created_at: datetime
    resume_id: int
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
    prediction: int
    probability: float

    model_config = ConfigDict(from_attributes=True)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "модель не смогла обработать данные"}
    )


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "HR AI Scout",
        "version": "0.1",
        "status": "running",
        "model": "feature_engineering_with_best_optuna_lr",
        "documentation": "/docs"
    }


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():

    model_loaded = hasattr(app.state, 'model')
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }


@app.get("/validation/demo", tags=["Documentation"])
async def validation_demo():
    return {
        "message": "Правила отправки значений для POST /forward",
        "description": "API принимает описание вакансии и возвращает топ-10 подходящих резюме из базы",
        "pydantic_version": "2.5.0",
        "input_fields": {
            "vacancy_name": {
                "type": "string",
                "required": False,
                "description": "Название вакансии"
            },
            "vacancy_area": {
                "type": "string",
                "required": True,
                "constraint": "Можно написать только на кириллице",
                "description": "Город вакансии"
            },
            "vacancy_experience": {
                "type": "enum",
                "required": True,
                "allowed_values": ["Нет опыта", "От 1 года до 3 лет", "От 3 до 6 лет", "Более 6 лет"],
                "description": "Требуемый опыт работы"
            },
            "vacancy_employment": {
                "type": "enum",
                "required": True,
                "allowed_values": ["Полная занятость", "Частичная занятость", "Проектная работа"],
                "description": "Тип занятости"
            },
            "vacancy_schedule": {
                "type": "enum",
                "required": True,
                "allowed_values": ["Полный день", "Удаленная работа", "Гибкий график", "Сменный график", "Вахтовый метод"],
                "description": "График работы"
            },
            "vacancy_description": {
                "type": "string",
                "required": True,
                "description": "Полное описание вакансии (требования, задачи, условия)"
            }
        },
        "output_format": {
            "type": "array",
            "items": {
                "resume_id": "integer - ID резюме",
                "y_pred_proba": "float - вероятность подходящести (0-1)"
            },
            "description": "Массив из топ-10 резюме, отсортированных по убыванию вероятности"
        }
    }


@app.post("/forward", response_model=List[TopResume], tags=["Prediction"])
async def forward(
    data: VacancyInput,
    db: AsyncSession = Depends(get_db)
):
    try:
        start_time = time.time()

        if not hasattr(app.state, 'model') or app.state.model is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="модель не смогла обработать данные"
            )

        if not hasattr(app.state, 'df_resumes'):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="модель не смогла обработать данные"
            )

        input_dict = {
            'vacancy_name': data.vacancy_name,
            'vacancy_area': data.vacancy_area,
            'vacancy_experience': data.vacancy_experience.value,
            'vacancy_employment': data.vacancy_employment.value,
            'vacancy_schedule': data.vacancy_schedule.value,
            'vacancy_description': data.vacancy_description
        }

        df_vacancy = pd.DataFrame([input_dict])
        df = df_vacancy.merge(app.state.df_resumes, how='cross')

        df['location_matching'] = df.apply(compute_location_matching, axis=1)
        df['resume_skill_count_in_vacancy'] = df.apply(resume_skill_count_in_vacancy, axis=1)
        df['last_position_in_vacancy'] = df.apply(last_position_in_vacancy, axis=1)

        df = compute_similarity_features(df, app.state.tfidf_vectorizer, app.state.experience_embeddings)

        X = df[FEATURES]
        y_pred = app.state.model.predict(X)
        y_pred_proba = app.state.model.predict_proba(X)[:, 1]

        df['y_pred'] = y_pred
        df['y_pred_proba'] = y_pred_proba

        processing_time = time.time() - start_time

        top = df.sort_values('y_pred_proba', ascending=False).head(10)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        for _, row in top.iterrows():
            history_record = PredictionHistory(
                resume_id=int(row.get('resume_id')),
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
                prediction=int(row.get('y_pred') or 0),
                probability=float(row.get('y_pred_proba') or 0.0),
                processing_time=processing_time
            )
            db.add(history_record)
            await db.commit()
            await db.refresh(history_record)

        result = [TopResume(resume_id=int(r.resume_id), y_pred_proba=float(r.y_pred_proba)) for _, r in top.iterrows()]

        return result

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="модель не смогла обработать данные"
        )


@app.get("/history", response_model=List[HistoryResponse], tags=["History"])
async def get_history(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin_user)
):
    result = await db.execute(
        select(PredictionHistory)
        .order_by(PredictionHistory.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    
    history = result.scalars().all()
    return list(history)


@app.delete("/history", status_code=status.HTTP_204_NO_CONTENT, tags=["History"])
async def delete_history(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_admin_user)
):
    await db.execute(delete(PredictionHistory))
    await db.commit()
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)

@app.get("/stats", tags=["Statistics"])
async def get_stats(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_active_admin_user)):
    result = await db.execute(select(PredictionHistory))
    records = result.scalars().all()
    
    if not records:
        return {
            "total_requests": 0,
            "message": "No data"
        }
    
    processing_times = [r.processing_time for r in records]
    
    processing_stats = {
        "mean": float(np.mean(processing_times)),
        "median_p50": float(np.percentile(processing_times, 50)),
        "p95": float(np.percentile(processing_times, 95)),
        "p99": float(np.percentile(processing_times, 99)),
        "min": float(np.min(processing_times)),
        "max": float(np.max(processing_times))
    }
    
    input_stats = {
        "resume_age": {
            "mean": float(np.mean([r.resume_age for r in records])),
            "median": float(np.median([r.resume_age for r in records]))
        },
        "resume_salary": {
            "mean": float(np.mean([r.resume_salary for r in records])),
            "median": float(np.median([r.resume_salary for r in records]))
        },
        "resume_experience_months": {
            "mean": float(np.mean([r.resume_experience_months for r in records])),
            "median": float(np.median([r.resume_experience_months for r in records]))
        },
        "similarity_score_tfidf": {
            "mean": float(np.mean([r.similarity_score_tfidf for r in records])),
            "median": float(np.median([r.similarity_score_tfidf for r in records]))
        }
    }
    
    prediction_distribution = {
        "total_predictions": len(records),
        "positive_predictions": sum(1 for r in records if r.prediction == 1),
        "negative_predictions": sum(1 for r in records if r.prediction == 0)
    }
    
    return {
        "total_requests": len(records),
        "processing_time_seconds": processing_stats,
        "input_characteristics": input_stats,
        "predictions": prediction_distribution
    }

if __name__ == "__main__":
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

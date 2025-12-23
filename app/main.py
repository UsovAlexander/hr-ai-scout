import os
import re
import time
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
from sqlalchemy import DateTime, Float, Integer, String, delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SECRET_KEY = os.environ.get("KEY", "test")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

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
    model_path = os.path.join(
        os.path.dirname(__file__),
        "feature_engineering_with_best_optuna_lr.pkl"
    )
    try:
        with open(model_path, 'rb') as f:
            app.state.model = pickle.load(f)
    except FileNotFoundError:
        print("Ошибка файл модели не найден")
        raise
    except Exception:
        print("Ошибка при загрузке модели")
        raise
    
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

class CandidateInput(BaseModel):
    vacancy_area: str = Field(..., description="Город вакансии")
    vacancy_experience: VacancyExperience = Field(..., description="Требуемый опыт")
    vacancy_employment: VacancyEmployment = Field(..., description="Тип занятости")
    vacancy_schedule: VacancySchedule = Field(..., description="График работы")
    resume_location: str = Field(..., description="Город кандидата")
    resume_gender: ResumeGender = Field(..., description="Пол кандидата")
    resume_applicant_status: ResumeApplicantStatus = Field(..., description="Статус поиска работы")
    resume_salary: float = Field(..., ge=0, le=1e7, description="Ожидаемая зарплата")
    resume_age: float = Field(..., ge=0, le=90, description="Возраст кандидата")
    resume_experience_months: float = Field(..., ge=0, le=720, description="Общий опыт в месяцах")
    resume_last_company_experience_months: float = Field(..., ge=0, le=720, description="Опыт в последней компании")
    location_matching: float = Field(..., ge=0, le=1, description="Совпадение локации (0 или 1)")
    resume_skill_count_in_vacancy: float = Field(..., ge=0, le=10000, description="Количество навыков, которые совпадают между резюме и вакансией")
    last_position_in_vacancy: float = Field(..., ge=0, le=1, description="Доля слов, которые совпадают с последней позиции с вакансией")
    similarity_score_tfidf: float = Field(..., ge=0, le=1, description="Косинусное сходство TF-IDF")

    @field_validator('vacancy_area', 'resume_location')
    @classmethod
    def check_cyrillic_only(cls, v: str, info) -> str:
        if not re.match(r'^[а-яА-ЯёЁ\s\-()]+$', v):
            raise ValueError(f'{info.field_name} можно написать только на кириллице')
        return v

    @model_validator(mode='after')
    def check_experience_consistency(self) -> 'CandidateInput':

        if self.resume_last_company_experience_months > self.resume_experience_months:
            raise ValueError(
                'Последний опыт не может превышать общий опыт'
            )
        return self

    class Config:
        json_schema_extra = {
            "example": {
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
        }


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
    "admin": {
        "user": "test",
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
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
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
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "is_admin": user.is_admin},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


class HistoryResponse(BaseModel):
    id: int
    created_at: datetime
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
        content={"detail": "bad request"}
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
        "message": "Правила отправки значений",
        "pydantic_version": "2.5.0",
        "validation_rules": {
            "vacancy_area": {
                "type": "string",
                "constraint": "Можно написать только на кириллице"
            },
            "vacancy_experience": {
                "type": "enum",
                "allowed_values": ["Нет опыта", "От 1 года до 3 лет", "От 3 до 6 лет", "Более 6 лет"]
            },
            "vacancy_employment": {
                "type": "enum",
                "allowed_values": ["Полная занятость", "Проектная работа", "Частичная занятость"]
            },
            "vacancy_schedule": {
                "type": "enum",
                "allowed_values": ["Удаленная работа", "Полный день", "Гибкий график", "Сменный график", "Вахтовый метод"]
            },
            "resume_location": {
                "type": "string",
                "constraint": "Можно написать только на кириллице"
            },
            "resume_gender": {
                "type": "enum",
                "allowed_values": ["Мужчина", "Женщина", "Неизвестно"]
            },
            "resume_applicant_status": {
                "type": "enum",
                "allowed_values": ["Активно ищет работу", "Рассматривает предложения"]
            },
            "resume_salary": {
                "type": "float",
                "constraint": "Значения от 0 до 10 000 000"
            },
            "resume_age": {
                "type": "float",
                "constraint": "Значения от 0 до 90"
            },
            "resume_experience_months": {
                "type": "float",
                "constraint": "Значения от 0 до 720"
            },
            "resume_last_company_experience_months": {
                "type": "float",
                "constraint": "Значения от 0 до 720 и не может превышать resume_experience_months"
            },
            "location_matching": {
                "type": "float",
                "constraint": "Бинарные значения - 1 или 0"
            },
            "resume_skill_count_in_vacancy": {
                "type": "float",
                "constraint": "Значения от 0 до 10000"
            },
            "last_position_in_vacancy": {
                "type": "float",
                "constraint": "Значения от 0 до 1"
            },
            "similarity_score_tfidf": {
                "type": "float",
                "constraint": "Значения от 0 до 1"
            }
        }
    }


@app.post("/forward", response_model=PredictionResponse, tags=["Prediction"])
async def forward(
    data: CandidateInput,
    db: AsyncSession = Depends(get_db)
):
    try:
        start_time = time.time()

        if not hasattr(app.state, 'model') or app.state.model is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="модель не смогла обработать данные"
            )

        input_dict = {
            'vacancy_area': data.vacancy_area,
            'vacancy_experience': data.vacancy_experience,
            'vacancy_employment': data.vacancy_employment,
            'vacancy_schedule': data.vacancy_schedule,
            'resume_salary': data.resume_salary,
            'resume_age': data.resume_age,
            'resume_experience_months': data.resume_experience_months,
            'resume_location': data.resume_location,
            'resume_gender': data.resume_gender,
            'resume_applicant_status': data.resume_applicant_status,
            'resume_last_company_experience_months': data.resume_last_company_experience_months,
            'location_matching': data.location_matching,
            'resume_skill_count_in_vacancy': data.resume_skill_count_in_vacancy,
            'last_position_in_vacancy': data.last_position_in_vacancy,
            'similarity_score_tfidf': data.similarity_score_tfidf
        }
        
        input_df = pd.DataFrame([input_dict])

        prediction = app.state.model.predict(input_df)[0]
        probability = app.state.model.predict_proba(input_df)[0]

        processing_time = time.time() - start_time

        history_record = PredictionHistory(
            vacancy_area=data.vacancy_area,
            vacancy_experience=data.vacancy_experience.value,
            vacancy_employment=data.vacancy_employment.value,
            vacancy_schedule=data.vacancy_schedule.value,
            resume_location=data.resume_location,
            resume_gender=data.resume_gender.value,
            resume_applicant_status=data.resume_applicant_status.value,
            resume_salary=data.resume_salary,
            resume_age=data.resume_age,
            resume_experience_months=data.resume_experience_months,
            resume_last_company_experience_months=data.resume_last_company_experience_months,
            location_matching=data.location_matching,
            resume_skill_count_in_vacancy=data.resume_skill_count_in_vacancy,
            last_position_in_vacancy=data.last_position_in_vacancy,
            similarity_score_tfidf=data.similarity_score_tfidf,
            prediction=int(prediction),
            probability=float(probability[1]),
            processing_time=processing_time
        )
        
        db.add(history_record)
        await db.commit()
        await db.refresh(history_record)

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability[1])
        )
    
    except Exception as e:
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

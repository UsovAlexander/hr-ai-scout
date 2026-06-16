"""
Нагрузочный тест FastAPI-сервиса HR-AI Scout на Locust.

Запуск (из корня проекта, сервис должен быть поднят на http://localhost:8000):

    # веб-интерфейс (графики, ручной разгон):
    locust -f load_testing/locustfile.py --host http://localhost:8000

    # headless — 50 пользователей, разгон 5 польз./сек, 2 минуты:
    locust -f load_testing/locustfile.py --host http://localhost:8000 \
           --headless -u 50 -r 5 -t 2m --csv load_testing/report

Учётные данные берутся из env (по умолчанию test/test — как в fake_users_db):

    LOCUST_USERNAME=test LOCUST_PASSWORD=test
"""
import os
import random

from locust import HttpUser, between, task

USERNAME = os.environ.get("LOCUST_USERNAME", "test")
PASSWORD = os.environ.get("LOCUST_PASSWORD", "test")

# Допустимые значения enum-полей вакансии (см. app/main.py)
EXPERIENCE = ["Нет опыта", "От 1 года до 3 лет", "От 3 до 6 лет", "Более 6 лет"]
EMPLOYMENT = ["Полная занятость", "Частичная занятость", "Проектная работа"]
SCHEDULE = ["Полный день", "Удаленная работа", "Гибкий график",
            "Сменный график", "Вахтовый метод"]

# Несколько реалистичных вакансий для разнообразия нагрузки
VACANCIES = [
    {
        "vacancy_name": "Аналитик данных/data analyst",
        "vacancy_area": "Москва",
        "vacancy_description": "Твои задачи: формирование баз данных, аналитика, SQL, Python",
    },
    {
        "vacancy_name": "Python-разработчик",
        "vacancy_area": "Санкт-Петербург",
        "vacancy_description": "Backend на FastAPI, PostgreSQL, Docker, очереди задач",
    },
    {
        "vacancy_name": "Менеджер по продажам",
        "vacancy_area": "Казань",
        "vacancy_description": "Работа с клиентами, ведение CRM, выполнение плана продаж",
    },
]


def random_vacancy() -> dict:
    """Собирает валидный payload для POST /forward."""
    base = random.choice(VACANCIES)
    return {
        **base,
        "vacancy_experience": random.choice(EXPERIENCE),
        "vacancy_employment": random.choice(EMPLOYMENT),
        "vacancy_schedule": random.choice(SCHEDULE),
    }


class HRScoutUser(HttpUser):
    """Имитирует рекрутёра: авторизуется и матчит резюме под вакансии."""

    # Пауза между запросами одного пользователя
    wait_time = between(1, 3)

    def on_start(self) -> None:
        """Получаем JWT-токен один раз при старте пользователя."""
        self.token = None
        with self.client.post(
            "/token",
            data={"username": USERNAME, "password": PASSWORD},
            name="POST /token",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                self.token = resp.json().get("access_token")
                resp.success()
            else:
                resp.failure(f"Не удалось авторизоваться: {resp.status_code} {resp.text}")

    @property
    def auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @task(10)
    def forward(self) -> None:
        """Основная нагрузка — подбор топ-10 резюме под вакансию."""
        if not self.token:
            return
        with self.client.post(
            "/forward",
            json=random_vacancy(),
            headers=self.auth_headers,
            name="POST /forward",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"{resp.status_code}: {resp.text[:200]}")

    @task(2)
    def health(self) -> None:
        """Лёгкая проверка доступности."""
        self.client.get("/health", name="GET /health")

    @task(1)
    def stats(self) -> None:
        """Админская агрегированная статистика (test — админ)."""
        if not self.token:
            return
        self.client.get("/stats", headers=self.auth_headers, name="GET /stats")

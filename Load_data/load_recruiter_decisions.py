"""
Load total_df.csv into ClickHouse table `recruiter_decisions`.

Table stores vacancy-resume pairs with recruiter decisions:
  target = 1 → invited to interview
  target = 0 → rejected

Run from repo root:
    python load_data/load_recruiter_decisions.py
"""
import os
import math
import pandas as pd
from clickhouse_driver import Client

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", 9000))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "1234")
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "default")

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "total_df.csv")
BATCH_SIZE = 10_000

DDL = """
CREATE TABLE IF NOT EXISTS recruiter_decisions (
    vacancy_id                        Int64,
    vacancy_name                      Nullable(String),
    vacancy_area                      Nullable(String),
    vacancy_experience                Nullable(String),
    vacancy_employment                Nullable(String),
    vacancy_schedule                  Nullable(String),
    vacancy_salary_from               Nullable(Float64),
    vacancy_salary_to                 Nullable(Float64),
    vacancy_salary_currency           Nullable(String),
    vacancy_salary_gross              Nullable(String),
    vacancy_description               Nullable(String),
    resume_id                         Int64,
    resume_title                      Nullable(String),
    resume_specialization             Nullable(String),
    resume_last_position              Nullable(String),
    resume_last_experience_description Nullable(String),
    resume_last_company_experience_period Nullable(String),
    resume_skills                     Nullable(String),
    resume_education                  Nullable(String),
    resume_courses                    Nullable(String),
    resume_salary                     Nullable(String),
    resume_age                        Nullable(Float64),
    resume_total_experience           Nullable(String),
    resume_experience_months          Nullable(Float64),
    resume_location                   Nullable(String),
    resume_gender                     Nullable(String),
    resume_applicant_status           Nullable(String),
    target                            Int8,
    decided_at                        DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY (vacancy_id, resume_id)
COMMENT 'Recruiter decisions: target=1 invited, target=0 rejected'
"""

NULLABLE_STR_COLS = [
    "vacancy_name", "vacancy_area", "vacancy_experience", "vacancy_employment",
    "vacancy_schedule", "vacancy_salary_currency", "vacancy_salary_gross",
    "vacancy_description", "resume_title", "resume_specialization",
    "resume_last_position", "resume_last_experience_description",
    "resume_last_company_experience_period", "resume_skills", "resume_education",
    "resume_courses", "resume_salary", "resume_total_experience",
    "resume_location", "resume_gender", "resume_applicant_status",
]
NULLABLE_FLOAT_COLS = ["vacancy_salary_from", "vacancy_salary_to", "resume_age", "resume_experience_months"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # vacancy_salary_gross can be Python bool (True/False) — cast to str first
    if "vacancy_salary_gross" in df.columns:
        df["vacancy_salary_gross"] = df["vacancy_salary_gross"].apply(
            lambda x: None if pd.isna(x) else str(x)
        )
    for col in NULLABLE_STR_COLS:
        if col == "vacancy_salary_gross":
            continue  # already handled above
        df[col] = df[col].where(df[col].notna(), None)
    for col in NULLABLE_FLOAT_COLS:
        df[col] = df[col].where(df[col].notna(), None)
        df[col] = df[col].apply(lambda x: None if x is not None and math.isnan(x) else x)
    return df


def main():
    client = Client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
    )

    print("Creating table recruiter_decisions …")
    client.execute(DDL)

    existing = client.execute("SELECT count() FROM recruiter_decisions")[0][0]
    if existing > 0:
        print(f"Table already has {existing:,} rows. Skipping load.")
        return

    print(f"Reading {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH)
    df = clean(df)

    col_order = [
        "vacancy_id", "vacancy_name", "vacancy_area", "vacancy_experience",
        "vacancy_employment", "vacancy_schedule", "vacancy_salary_from",
        "vacancy_salary_to", "vacancy_salary_currency", "vacancy_salary_gross",
        "vacancy_description", "resume_id", "resume_title", "resume_specialization",
        "resume_last_position", "resume_last_experience_description",
        "resume_last_company_experience_period", "resume_skills", "resume_education",
        "resume_courses", "resume_salary", "resume_age", "resume_total_experience",
        "resume_experience_months", "resume_location", "resume_gender",
        "resume_applicant_status", "target",
    ]
    df = df[col_order]

    total = len(df)
    n_batches = math.ceil(total / BATCH_SIZE)
    print(f"Inserting {total:,} rows in {n_batches} batches of {BATCH_SIZE:,} …")

    for i in range(n_batches):
        batch = df.iloc[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        rows = [tuple(row) for row in batch.itertuples(index=False, name=None)]
        client.execute(
            f"INSERT INTO recruiter_decisions ({', '.join(col_order)}) VALUES",
            rows,
        )
        print(f"  batch {i + 1}/{n_batches} done ({min((i + 1) * BATCH_SIZE, total):,} rows)")

    count = client.execute("SELECT count() FROM recruiter_decisions")[0][0]
    print(f"Done. Total rows in recruiter_decisions: {count:,}")


if __name__ == "__main__":
    main()

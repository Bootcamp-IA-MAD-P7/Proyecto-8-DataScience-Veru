"""Adaptador de base de datos para la API (D11).

Guarda el historial de predicciones en PostgreSQL (producción/Docker) o en
SQLite (desarrollo local y tests) según la variable de entorno DATABASE_URL.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "predictions.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB}")

FEATURE_COLUMNS = [
    "age", "avg_glucose_level", "bmi", "gender", "ever_married",
    "work_type", "Residence_type", "smoking_status",
    "hypertension", "heart_disease",
]

META_COLUMNS = ["probabilidad", "clase", "riesgo", "created_at"]
ALL_COLUMNS = ["id"] + FEATURE_COLUMNS + META_COLUMNS


def _is_sqlite():
    return DATABASE_URL.startswith("sqlite")


def _connect():
    if _is_sqlite():
        path = DATABASE_URL.replace("sqlite:///", "")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def _placeholder(n):
    return ", ".join("?" if _is_sqlite() else "%s" for _ in range(n))


def init_db():
    with _connect() as conn:
        if _is_sqlite():
            conn.execute(
                "CREATE TABLE IF NOT EXISTS predictions ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " age REAL, avg_glucose_level REAL, bmi REAL,"
                " gender TEXT, ever_married TEXT, work_type TEXT,"
                " residence_type TEXT, smoking_status TEXT,"
                " hypertension INTEGER, heart_disease INTEGER,"
                " probabilidad REAL, clase INTEGER, riesgo TEXT,"
                " created_at TEXT)"
            )
        else:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS predictions ("
                " id SERIAL PRIMARY KEY,"
                " age DOUBLE PRECISION, avg_glucose_level DOUBLE PRECISION, bmi DOUBLE PRECISION,"
                " gender TEXT, ever_married TEXT, work_type TEXT,"
                " residence_type TEXT, smoking_status TEXT,"
                " hypertension INTEGER, heart_disease INTEGER,"
                " probabilidad DOUBLE PRECISION, clase INTEGER, riesgo TEXT,"
                " created_at TIMESTAMP DEFAULT NOW())"
            )
        conn.commit()


def save_prediction(features: dict, probabilidad: float, clase: int, riesgo: str):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cols = FEATURE_COLUMNS + ["probabilidad", "clase", "riesgo"]
    values = [features.get(c) for c in FEATURE_COLUMNS] + [probabilidad, clase, riesgo]
    if _is_sqlite():
        cols = cols + ["created_at"]
        values = values + [ts]
    sql = f"INSERT INTO predictions ({', '.join(cols)}) VALUES ({_placeholder(len(values))})"
    with _connect() as conn:
        conn.execute(sql, values)
        conn.commit()


def list_predictions(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?" if _is_sqlite()
            else "SELECT * FROM predictions ORDER BY id DESC LIMIT %s",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
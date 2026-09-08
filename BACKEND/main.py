"""API REST de predicción de riesgo de ictus (productivización, D10).

Expone el modelo final (`models/catboost_final.pkl`) mediante un endpoint
`POST /predict` que recibe los datos del paciente y devuelve la probabilidad,
la clase y el veredicto de riesgo.

Ejecución local:
    uvicorn BACKEND.main:app --reload
    # o bien: make api
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "catboost_final.pkl"

THRESHOLD = 0.5

NUMERIC = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
BINARY = ["hypertension", "heart_disease"]
FEATURES = NUMERIC + CATEGORICAL + BINARY

GENDERS = ["Male", "Female", "Other"]
WORK_TYPES = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
SMOKING_STATUSES = ["never smoked", "formerly smoked", "smokes", "Unknown"]


class PatientData(BaseModel):
    age: float
    avg_glucose_level: float
    bmi: float
    gender: str
    ever_married: str
    work_type: str
    residence_type: str
    smoking_status: str
    hypertension: int
    heart_disease: int


class PredictionResponse(BaseModel):
    probabilidad_ictus: float
    clase: int
    riesgo: str


def _load_model():
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"No se encuentra el modelo final en {MODEL_PATH}. "
                   "Ejecuta `make train-cat` primero.",
        )
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error cargando el modelo: {exc}") from exc


model = _load_model()

app = FastAPI(
    title="API de riesgo de ictus",
    description="Productiviza el modelo CatBoost final para predecir riesgo de ictus.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "modelo": MODEL_PATH.name}


@app.post("/predict", response_model=PredictionResponse)
def predict(paciente: PatientData):
    for campo, validos, nombre in [
        ("gender", GENDERS, "gender"),
        ("work_type", WORK_TYPES, "work_type"),
        ("smoking_status", SMOKING_STATUSES, "smoking_status"),
    ]:
        if getattr(paciente, campo) not in validos:
            raise HTTPException(
                status_code=422,
                detail=f"{nombre} inválido: '{getattr(paciente, campo)}'. "
                       f"Valores válidos: {validos}",
            )
    if paciente.ever_married not in ("Yes", "No"):
        raise HTTPException(status_code=422, detail="ever_married debe ser 'Yes' o 'No'")
    if paciente.residence_type not in ("Urban", "Rural"):
        raise HTTPException(status_code=422, detail="residence_type debe ser 'Urban' o 'Rural'")
    if paciente.hypertension not in (0, 1) or paciente.heart_disease not in (0, 1):
        raise HTTPException(status_code=422, detail="hypertension y heart_disease deben ser 0 o 1")

    fila = pd.DataFrame([{
        "age": paciente.age,
        "avg_glucose_level": paciente.avg_glucose_level,
        "bmi": paciente.bmi,
        "gender": paciente.gender,
        "ever_married": paciente.ever_married,
        "work_type": paciente.work_type,
        "Residence_type": paciente.residence_type,
        "smoking_status": paciente.smoking_status,
        "hypertension": paciente.hypertension,
        "heart_disease": paciente.heart_disease,
    }])[FEATURES]

    proba = float(model.predict_proba(fila)[0, 1])
    clase = int(proba >= THRESHOLD)
    riesgo = "ALTO" if clase == 1 else "BAJO"

    return PredictionResponse(
        probabilidad_ictus=round(proba, 4),
        clase=clase,
        riesgo=riesgo,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
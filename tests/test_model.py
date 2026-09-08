"""Tests de los requisitos ya implementados del proyecto.

Cubre (requisitos del trabajo):
- Dataset: carga correcta (shape, sin nulos, proporción de ictus ~5 %).
- Preprocesado: columnas de features correctas y target binario.
- Modelo final (CatBoost regularizado): existe, es cargable, predice y CUMPLE
  el requisito de overfitting (|train - test| <= 5 puntos porcentuales).
- Presence/nombre del informe de rendimiento.
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "stroke_dataset.csv"
MODEL_PATH = ROOT / "models" / "catboost_final.pkl"
REPORT_PATH = ROOT / "models" / "INFORME_MODELOS.md"

TARGET = "stroke"
RANDOM_STATE = 42
MAX_DIFF = 0.05  # requisito: <= 5 puntos porcentuales
THRESHOLD = 0.5

NUMERIC = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
BINARY = ["hypertension", "heart_disease"]
FEATURES = NUMERIC + CATEGORICAL + BINARY


@pytest.fixture(scope="module")
def dataset():
    df = pd.read_csv(DATA_PATH)
    return df


@pytest.fixture(scope="module")
def final_model():
    return joblib.load(MODEL_PATH)


# ---------- 1. DATASET ----------
def test_dataset_shape(dataset):
    assert dataset.shape == (4981, 11)


def test_dataset_no_nulls(dataset):
    assert not dataset.isnull().any().any()


def test_dataset_no_duplicates(dataset):
    assert dataset.duplicated().sum() == 0


def test_target_in_dataset(dataset):
    assert TARGET in dataset.columns


def test_stroke_proportion(dataset):
    y = dataset[TARGET]
    assert set(y.unique()) <= {0, 1}
    # Proporción de ictus ~5 % (requisito: dataset desbalanceado)
    assert abs(y.mean() - 0.05) < 0.02


# ---------- 2. PREPROCESADO ----------
def test_preprocess_features(dataset):
    X = dataset[FEATURES]
    assert list(X.columns) == FEATURES
    assert len(X) == 4981


# ---------- 3. MODELO FINAL ----------
def test_final_model_exists():
    assert MODEL_PATH.exists(), "Falta models/catboost_final.pkl"


def test_final_model_can_predict(final_model):
    # Un paciente sintético: debe devolver probabilidades sin error
    paciente = pd.DataFrame([{
        "age": 75.0, "avg_glucose_level": 200.0, "bmi": 30.0,
        "gender": "Male", "ever_married": "Yes", "work_type": "Private",
        "Residence_type": "Urban", "smoking_status": "never smoked",
        "hypertension": 1, "heart_disease": 1,
    }])
    proba = final_model.predict_proba(paciente)[0]
    assert proba.shape == (2,)
    assert 0.0 <= proba[1] <= 1.0


def _train_test_split():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def test_final_model_meets_overfitting_requirement(final_model):
    """Requisito clave: |train - test| <= 5 puntos en recall/precision/F1."""
    X_train, X_test, y_train, y_test = _train_test_split()

    def metrics(Xs, ys):
        proba = final_model.predict_proba(Xs)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        return {
            "recall": recall_score(ys, pred),
            "precision": precision_score(ys, pred),
            "f1": f1_score(ys, pred),
        }

    tr = metrics(X_train, y_train)
    te = metrics(X_test, y_test)

    for m in ["recall", "precision", "f1"]:
        diff = abs(tr[m] - te[m])
        assert diff <= MAX_DIFF, (
            f"Requisito incumplido en {m}: |train-test| = {diff:.3f} > {MAX_DIFF}"
        )


# ---------- 4. INFORME ----------
def test_report_exists():
    assert REPORT_PATH.exists(), "Falta models/INFORME_MODELOS.md"


@pytest.mark.parametrize("keyword", [
    "Recall", "Precision", "F1", "AUC-ROC", "CatBoost", "overfitting",
])
def test_report_mentions_required_metrics(keyword):
    text = REPORT_PATH.read_text(encoding="utf-8")
    assert keyword.lower() in text.lower(), f"El informe no menciona {keyword}"

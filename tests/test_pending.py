"""Tests de requisitos de productivización y CLI.

Requisitos del trabajo cubiertos:
- Aplicación en línea de comandos (CLI) que ingresa datos y devuelve la predicción.
- Solución que productivice el modelo (API REST FastAPI).
- Productivización: la API carga el modelo final, predice y valida la entrada.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent

VALIDACIÓN_DATOS = {
    "age": 75.0, "avg_glucose_level": 200.0, "bmi": 30.0,
    "gender": "Male", "ever_married": "Yes", "work_type": "Private",
    "residence_type": "Urban", "smoking_status": "never smoked",
    "hypertension": 1, "heart_disease": 1,
}


# ---------- 3. CLI (implementado) ----------
CLI_SCRIPT = ROOT / "scripts" / "predict_cli.py"


def test_cli_script_exists():
    assert CLI_SCRIPT.exists(), "Falta scripts/predict_cli.py"


def test_cli_returns_prediction():
    """El CLI debe aceptar datos del paciente y devolver la predicción."""
    cmd = [sys.executable, str(CLI_SCRIPT)]
    for k, v in VALIDACIÓN_DATOS.items():
        cmd += [f"--{k}", str(v)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = result.stdout.lower()
    assert ("riesgo" in output) or ("ictus" in output) or ("probabilidad" in output)


# ---------- 4. Productivización (implementada: API FastAPI) ----------
API_MODULE = ROOT / "BACKEND" / "main.py"


def test_api_module_exists():
    assert API_MODULE.exists(), "Falta BACKEND/main.py (API FastAPI)"


def test_api_loads():
    spec = importlib.util.spec_from_file_location("backend_main", API_MODULE)
    assert spec is not None and spec.loader is not None


def test_api_imports_final_model():
    """La API debe cargar el modelo final (catboost_final.pkl)."""
    spec = importlib.util.spec_from_file_location("backend_main", API_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.MODEL_PATH.name == "catboost_final.pkl"
    assert module.model is not None


@pytest.fixture(scope="module")
def client():
    spec = importlib.util.spec_from_file_location("backend_main", API_MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return TestClient(module.app)


def test_api_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["modelo"] == "catboost_final.pkl"


def test_api_predict_ok(client):
    r = client.post("/predict", json=VALIDACIÓN_DATOS)
    assert r.status_code == 200
    body = r.json()
    assert body["clase"] in (0, 1)
    assert body["riesgo"] in ("ALTO", "BAJO")
    assert 0.0 <= body["probabilidad_ictus"] <= 1.0


def test_api_predict_validation_error(client):
    datos = dict(VALIDACIÓN_DATOS)
    datos["work_type"] = "ValorInvalido"
    r = client.post("/predict", json=datos)
    assert r.status_code == 422

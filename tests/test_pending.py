"""Tests de requisitos PENDIENTES de implementar.

Estos tests se mantienen saltados (skip) hasta que exista la funcionalidad
correspondiente. Cuando se implemente el CLI y la productivización (p. ej.
FastAPI/Streamlit/Gradio), descomentar el skip y ajustar el contrato.

Requisitos pendientes del trabajo:
- CLI en línea de comandos que ingrese datos y devuelva la predicción.
- Solución que productivice el modelo (API / Streamlit / Gradio / Dash).
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


# ---------- 3. CLI (implementado) ----------
CLI_SCRIPT = ROOT / "scripts" / "predict_cli.py"


def test_cli_script_exists():
    assert CLI_SCRIPT.exists(), "Falta scripts/predict_cli.py"


def test_cli_returns_prediction():
    """El CLI debe aceptar datos del paciente y devolver la predicción."""
    cmd = [
        sys.executable, str(CLI_SCRIPT),
        "--age", "75.0", "--gender", "Male", "--heart_disease", "1",
        "--hypertension", "1", "--smoking_status", "never smoked",
        "--work_type", "Private", "--residence_type", "Urban",
        "--ever_married", "Yes", "--bmi", "30.0", "--avg_glucose_level", "200.0",
    ]
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

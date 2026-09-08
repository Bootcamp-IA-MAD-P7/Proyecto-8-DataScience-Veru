"""Aplicación en línea de comandos (CLI) de predicción de riesgo de ictus.

Carga el modelo final (`models/catboost_final.pkl`, D7) y, a partir de los
datos del paciente pasados por argumentos, devuelve la probabilidad y el
veredicto de riesgo.

Uso:
    python scripts/predict_cli.py --age 75 --gender Male --hypertension 1 \
        --heart_disease 1 --ever_married Yes --work_type Private \
        --residence_type Urban --smoking_status "never smoked" \
        --bmi 30 --avg_glucose_level 200
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "catboost_final.pkl"

THRESHOLD = 0.5

GENDERS = ["Male", "Female", "Other"]
WORK_TYPES = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
SMOKING_STATUSES = ["never smoked", "formerly smoked", "smokes", "Unknown"]

NUMERIC = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
BINARY = ["hypertension", "heart_disease"]
FEATURES = NUMERIC + CATEGORICAL + BINARY


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Predice el riesgo de ictus de un paciente con el modelo final."
    )
    p.add_argument("--age", required=True, type=float, help="Edad (años)")
    p.add_argument("--avg_glucose_level", required=True, type=float,
                   help="Nivel medio de glucosa (mg/dL)")
    p.add_argument("--bmi", required=True, type=float, help="Índice de masa corporal")
    p.add_argument("--gender", required=True, choices=GENDERS, help="Género")
    p.add_argument("--ever_married", required=True, choices=["Yes", "No"],
                   help="¿Ha estado casado/a?")
    p.add_argument("--work_type", required=True, choices=WORK_TYPES, help="Tipo de trabajo")
    p.add_argument("--residence_type", required=True, choices=["Urban", "Rural"],
                   help="Tipo de residencia")
    p.add_argument("--smoking_status", required=True, choices=SMOKING_STATUSES,
                   help="Estado de fumador")
    p.add_argument("--hypertension", required=True, choices=["0", "1"],
                   help="Hipertensión (0/1)")
    p.add_argument("--heart_disease", required=True, choices=["0", "1"],
                   help="Enfermedad cardíaca (0/1)")
    p.add_argument("--threshold", type=float, default=THRESHOLD,
                   help=f"Umbral de clasificación (default {THRESHOLD})")
    return p.parse_args(argv)


def build_patient_dataframe(args):
    return pd.DataFrame([{
        "age": args.age,
        "avg_glucose_level": args.avg_glucose_level,
        "bmi": args.bmi,
        "gender": args.gender,
        "ever_married": args.ever_married,
        "work_type": args.work_type,
        "Residence_type": args.residence_type,
        "smoking_status": args.smoking_status,
        "hypertension": int(args.hypertension),
        "heart_disease": int(args.heart_disease),
    }])[FEATURES]


def main(argv=None):
    args = parse_args(argv)
    if not MODEL_PATH.exists():
        print(f"Error: no se encuentra el modelo final en {MODEL_PATH}. "
              f"Ejecuta primero `make train-cat`.", file=sys.stderr)
        return 2

    pipe = joblib.load(MODEL_PATH)
    paciente = build_patient_dataframe(args)

    proba = float(pipe.predict_proba(paciente)[0, 1])
    pred = int(proba >= args.threshold)

    print("=" * 50)
    print("Predicción de riesgo de ictus")
    print("=" * 50)
    print(f"Probabilidad de ictus : {proba:.4f} ({proba*100:.1f}%)")
    print(f"Clasificación (umbral {args.threshold}):", end=" ")
    if pred == 1:
        print("RIESGO ALTO — se recomienda revisión médica.")
    else:
        print("Riesgo bajo.")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())

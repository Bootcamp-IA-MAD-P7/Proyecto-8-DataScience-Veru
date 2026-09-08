"""Comparativa train vs test para control de overfitting.

Criterio del trabajo: la diferencia entre métricas de training y de test
debe ser <= 5 puntos porcentuales (|train - test| <= 0.05).

Evalúa los modelos entrenados en recall, precisión, F1 y AUC-ROC,
tanto en train como en test, y marca cuáles cumplen el criterio.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "stroke_dataset.csv"
MODEL_DIR = ROOT / "models"

TARGET = "stroke"
RANDOM_STATE = 42
MAX_DIFF = 0.05  # 5 puntos porcentuales

NUMERIC = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
BINARY = ["hypertension", "heart_disease"]
FEATURES = NUMERIC + CATEGORICAL + BINARY

MODELS = [
    ("Random Forest baseline", "random_forest_baseline.pkl"),
    ("Random Forest + SMOTE", "random_forest_smote.pkl"),
    ("XGBoost baseline", "xgboost_baseline.pkl"),
    ("CatBoost final (regularizado)", "catboost_final.pkl"),
]

# Umbral por defecto 0.5 (el criterio de overfitting se evalúa en las
# probabilidades; se usa 0.5 para clasificar en recall/precision/F1).
THRESHOLD = 0.5


def evaluate(clf, X, y):
    proba = clf.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    return {
        "recall": recall_score(y, pred),
        "precision": precision_score(y, pred),
        "f1": f1_score(y, pred),
        "auc_roc": roc_auc_score(y, proba),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    metrics = ["recall", "precision", "f1", "auc_roc"]
    print(f"Criterio: |train - test| <= {MAX_DIFF*100:.0f} puntos porcentuales (umbral {THRESHOLD})")
    print("=" * 90)

    for name, fname in MODELS:
        clf = joblib.load(MODEL_DIR / fname)
        tr = evaluate(clf, X_train, y_train)
        te = evaluate(clf, X_test, y_test)

        print(f"\n### {name}")
        print(f"{'Métrica':<10} | {'Train':>7} | {'Test':>7} | {'|Diff|':>8} | {'Cumple?':>8}")
        print("-" * 60)
        all_ok = True
        for m in metrics:
            diff = abs(tr[m] - te[m])
            ok = diff <= MAX_DIFF
            all_ok = all_ok and ok
            print(f"{m:<10} | {tr[m]:>7.3f} | {te[m]:>7.3f} | {diff:>8.3f} | {'SI' if ok else 'NO':>8}")
        # Nota: AUC-ROC es una medida de ranking; el criterio de 5 puntos
        # suele aplicarse a métricas de clasificación (recall/prec/F1).
        print(f"  -> ¿Cumple en todas las métricas de clasificación (recall/prec/F1)? "
              f"{'SI' if all_ok else 'NO'}")


if __name__ == "__main__":
    main()

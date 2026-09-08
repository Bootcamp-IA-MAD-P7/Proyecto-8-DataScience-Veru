"""Entrenamiento Random Forest (class_weight='balanced') — para comparativa.

Permite reproducir las métricas del Random Forest baseline que se usan en la
comparativa de models/INFORME_MODELOS.md frente a CatBoost (modelo final).
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "stroke_dataset.csv"
MODEL_DIR = ROOT / "models"

TARGET = "stroke"
RANDOM_STATE = 42

NUMERIC = ["age", "avg_glucose_level", "bmi"]
CATEGORICAL = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
BINARY = ["hypertension", "heart_disease"]
FEATURES = NUMERIC + CATEGORICAL + BINARY

THRESHOLD = 0.5


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("bin", "passthrough", BINARY),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=RANDOM_STATE,
        n_jobs=-1, class_weight="balanced",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    print(f"Datos: {df.shape[0]} filas | ictus: {y.sum()} ({y.mean()*100:.2f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)

    print(f"\n=== RANDOM FOREST (class_weight='balanced', umbral {THRESHOLD}) ===")
    print(f"F1        : {f1_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC   : {roc_auc_score(y_test, y_proba):.4f}")
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    print(f"AUC-PR    : {auc(rec, prec):.4f}")
    print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification report:\n", classification_report(y_test, y_pred))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for train_idx, val_idx in cv.split(X, y):
        p = build_pipeline()
        p.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = p.predict_proba(X.iloc[val_idx])[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        yv = y.iloc[val_idx]
        folds.append({
            "f1": f1_score(yv, pred),
            "recall": recall_score(yv, pred),
            "precision": precision_score(yv, pred),
            "auc_roc": roc_auc_score(yv, proba),
        })
    print("\n=== VALIDACIÓN CRUZADA (5 folds, umbral 0.5) ===")
    for m in ["f1", "recall", "precision", "auc_roc"]:
        print(f"{m:10s}: {sum(f[m] for f in folds)/len(folds):.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "random_forest_baseline.pkl")
    print(f"\nModelo guardado en: {MODEL_DIR / 'random_forest_baseline.pkl'}")


if __name__ == "__main__":
    main()

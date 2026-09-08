"""Control de overfitting sobre CatBoost.

Objetivo del trabajo: |train - test| <= 5 puntos porcentuales en las métricas
de clasificación (recall, precisión, F1).

Se prueban varias configuraciones de regularización (menos iteraciones,
profundidad limitada, min_child_weight, l2_leaf_reg, bagging) y se mide la
diferencia train vs test para elegir la que cumple el criterio.
"""

from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

THRESHOLD = 0.5

# Configuraciones de regularización de menor a mayor control del overfitting
# Nota: en CatBoost el equivalente a min_child_weight es min_data_in_leaf.
CONFIGS = [
    {"name": "light (iter=100, depth=4)", "iterations": 100, "depth": 4,
     "min_data_in_leaf": 1, "l2_leaf_reg": 3, "bagging_temperature": 0},
    {"name": "med (iter=100, depth=3)", "iterations": 100, "depth": 3,
     "min_data_in_leaf": 8, "l2_leaf_reg": 8, "bagging_temperature": 1},
    {"name": "strong (iter=60, depth=2)", "iterations": 60, "depth": 2,
     "min_data_in_leaf": 20, "l2_leaf_reg": 15, "bagging_temperature": 2},
    {"name": "very-strong (iter=40, depth=2)", "iterations": 40, "depth": 2,
     "min_data_in_leaf": 40, "l2_leaf_reg": 30, "bagging_temperature": 3},
]


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("bin", "passthrough", BINARY),
        ]
    )


def build_pipeline(scale_pos_weight, cfg):
    kwargs = dict(cfg)
    name = kwargs.pop("name")
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", CatBoostClassifier(
                iterations=kwargs["iterations"],
                depth=kwargs["depth"],
                learning_rate=0.1,
                l2_leaf_reg=kwargs["l2_leaf_reg"],
                min_data_in_leaf=kwargs["min_data_in_leaf"],
                bagging_temperature=kwargs["bagging_temperature"],
                scale_pos_weight=scale_pos_weight,
                eval_metric="Logloss",
                random_seed=RANDOM_STATE,
                verbose=False,
                thread_count=-1,
            )),
        ]
    ), name


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    print(f"Criterio: |train - test| <= {MAX_DIFF*100:.0f} puntos (umbral {THRESHOLD})")
    print("=" * 90)

    # Mejor configuración que cumple el criterio (por F1 en test)
    best = {"f1": -1, "pipe": None, "name": None}

    for cfg in CONFIGS:
        pipe, name = build_pipeline(spw, cfg)
        pipe.fit(X_train, y_train)

        def metrics(Xs, ys):
            proba = pipe.predict_proba(Xs)[:, 1]
            pred = (proba >= THRESHOLD).astype(int)
            return {
                "recall": recall_score(ys, pred),
                "precision": precision_score(ys, pred),
                "f1": f1_score(ys, pred),
            }

        tr = metrics(X_train, y_train)
        te = metrics(X_test, y_test)

        print(f"\n### CatBoost {name}")
        print(f"{'Métrica':<10} | {'Train':>7} | {'Test':>7} | {'|Diff|':>8} | {'Cumple?':>8}")
        print("-" * 60)
        all_ok = True
        for m in ["recall", "precision", "f1"]:
            diff = abs(tr[m] - te[m])
            ok = diff <= MAX_DIFF
            all_ok = all_ok and ok
            print(f"{m:<10} | {tr[m]:>7.3f} | {te[m]:>7.3f} | {diff:>8.3f} | {'SI' if ok else 'NO':>8}")
        verdict = "CUMPLE" if all_ok else "no cumple"
        print(f"  -> {name}: {verdict} el criterio de 5 puntos en recall/precisión/F1")

        MODEL_DIR.mkdir(exist_ok=True)
        safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        joblib.dump(pipe, MODEL_DIR / f"catboost_reg_{safe}.pkl")

        # Guardar como final la mejor config que cumpla el criterio (por F1 test)
        if all_ok and te["f1"] > best["f1"]:
            best = {"f1": te["f1"], "pipe": pipe, "name": name}

    print("\nNota: se reclasifica con umbral 0.5. Si ninguna configuración cumple, "
          "la próxima vía es ajustar el umbral con validación cruzada u OOF.")

    if best["pipe"] is not None:
        joblib.dump(best["pipe"], MODEL_DIR / "catboost_final.pkl")
        print(f"\nMODELO FINAL guardado en: {MODEL_DIR / 'catboost_final.pkl'}"
              f"  ({best['name']}, F1 test {best['f1']:.3f})")
    else:
        print("\nNinguna configuración cumplió el criterio; no se sobreescribe catboost_final.pkl.")


if __name__ == "__main__":
    main()

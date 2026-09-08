# PROYECTO DATA SCIENTIST / AI DEVELOPER: Aprendizaje Supervisado

**Predicción del riesgo de ictus** — modelo de Machine Learning funcional que, a partir de los datos clínicos de un paciente, devuelve la probabilidad de sufrir un ictus.

<details>
<summary>📋 Tabla de contenidos</summary>

1. [Primera parte — Idea de negocio](#primera-parte--idea-de-negocio)
   - [Tecnologías utilizadas](#tecnologías-utilizadas)
   - [Propósito, problema que soluciona y público objetivo](#propósito-problema-que-soluciona-y-público-objetivo)
   - [Estructura del código](#estructura-del-código)
   - [Repositorio](#repositorio)
   - [Flujo de usuario (userflow)](#flujo-de-usuario-userflow)
2. [Segunda parte — Parte técnica](#segunda-parte--parte-técnica)
   - [Implementaciones importantes](#implementaciones-importantes)
   - [Tecnologías utilizadas](#tecnologías-utilizadas-1)
   - [Diseño de la base de datos](#diseño-de-la-base-de-datos)
   - [Paradigma de desarrollo y estructura de carpetas](#paradigma-de-desarrollo-y-estructura-de-carpetas)
   - [Convenciones de nombres](#convenciones-de-nombres)
   - [Organización del repositorio, ramas y commits](#organización-del-repositorio-ramas-y-commits)
   - [Documentación general y de la API](#documentación-general-y-de-la-api)

</details>

---

# Primera parte — Idea de negocio

## Tecnologías utilizadas

| Ámbito | Tecnología |
|---|---|
| Lenguaje | **Python 3.13** |
| Análisis exploratorio (EDA) | **Pandas**, **Matplotlib**, **Seaborn**, **missingno** — notebooks Jupyter (gestionados con **Jupytext**) |
| Modelado / ML | **Scikit-learn**, **CatBoost**, **XGBoost**, **imbalanced-learn (SMOTE)** |
| Gestión de dependencias | **uv** (`pyproject.toml` + `uv.lock`) |
| Orquestación de tareas | **Makefile** |
| API / productivización | **FastAPI**, **Uvicorn**, **Pydantic** |
| Testing | **Pytest** |
| Documentación de decisiones | SDD (`SDD/`) |

## Propósito, problema que soluciona y público objetivo

### Propósito
Entregar un **modelo de Machine Learning funcional** que prediga si un paciente está en riesgo de sufrir un ictus, cumpliendo el requisito clave del curso: la **diferencia entre las métricas de entrenamiento y las de test debe ser inferior a 5 puntos porcentuales** (control de overfitting).

### Problema que soluciona
El ictus (accidente cerebrovascular) es una de las principales causas de mortalidad; detectar a tiempo a los pacientes de riesgo permite una intervención médica preventiva. Un clasificador desbalanceado "engaña" con una precisión alta (~95 %) que en realidad no detecta ningún ictus. Este proyecto construye un **sistema de cribado** que:
- **No deja escapar ictus** (recall alto: 0.86 en test).
- **Generaliza**: las métricas de train y test se mantienen a menos de 5 puntos (criterio de la asignatura).
- Es **consumible**: se puede interrogar por consola (CLI) y vía **API REST** (FastAPI).

### Público objetivo (user stories)

| User story | Roles |
|---|---|
| *Como profesional sanitario, quiero introducir los datos de un paciente y obtener su riesgo de ictus, para priorizar revisiones.* | Médicos de atención primaria, personal de cribado |
| *Como desarrollador/a, quiero invocar la predicción desde un servicio HTTP (API), para integrarla en una aplicación.* | Equipos de software sanitario |
| *Como alumno/profesor, quiero reproducir el entrenamiento y verificar el requisito de overfitting, para auditar el trabajo.* | Comisión evaluadora |

## Estructura del código

```
.
├── BACKEND/                 # API REST FastAPI (productivización)
│   └── main.py              #   Endpoints /predict y /health
├── data/                    # Datos: stroke_dataset.csv (n = 4981)
├── models/                  # Modelos serializados (.pkl) e informes
│   ├── catboost_final.pkl   #   MODELO FINAL (CatBoost regularizado)
│   ├── random_forest_baseline.pkl   #   Comparativa RF
│   ├── random_forest_smote.pkl      #   Comparativa RF + SMOTE
│   ├── xgboost_baseline.pkl         #   Comparativa XGBoost
│   └── INFORME_MODELOS.md           #   Informe de rendimiento del modelo
├── notebooks/               # EDA: 01–06 (gráficos) + 07.Informe.md
├── scripts/                 # Scripts de entrenamiento, CLI y verificación
│   ├── train_random_forest.py       #   Entrena RF baseline
│   ├── train_random_forest_smote.py #   Entrena RF + SMOTE
│   ├── train_xgboost.py             #   Entrena XGBoost
│   ├── train_catboost_regularized.py#   Entrena el modelo FINAL (CatBoost)
│   ├── compare_train_test.py        #   Comprueba requisito de overfitting
│   └── predict_cli.py               #   Aplicación de línea de comandos (CLI)
├── tests/                   # Suite de tests (pytest) + INFORME_TESTS.md
├── SDD/                     # Documento de diseño (Decisiones D1–D11)
├── makefile                 # Orquestación de tareas
├── pyproject.toml           # Dependencias del proyecto (uv)
├── uv.lock                  # Lock de dependencias
└── README.md                # Este documento
```

## Repositorio

- Repositorio remoto: GitHub — `Bootcamp-IA-MAD-P7/Proyecto-8-DataScience-Veru`.
- Organización: ramas *feature* (`eda`, `models`, `informe`, `fastapi`, `readme`) fusionadas en `dev`; `main` solo al cierre del proyecto.
- Commits con mensajes descriptivos en español y prefijo del tipo de cambio (`feat/`, `fix/`, `chore/`, `docs/`).

## Flujo de usuario (userflow)

**Vía CLI (consola):**

```
Usuario → introduce datos del paciente (--age, --gender, ...)
       → scripts/predict_cli.py carga catboost_final.pkl
       → devuelve: Probabilidad de ictus + veredicto RIEGO ALTO/BAJO
```

**Vía API (FastAPI):**

```
Cliente (curl/Postman/app) --> POST /predict  (JSON con datos del paciente)
                            --> {probabilidad_ictus, clase, riesgo}
```

**Ejemplo real (CLI):**

```bash
make predict ARGS="--age 75 --gender Male --hypertension 1 --heart_disease 1 \
  --ever_married Yes --work_type Private --residence_type Urban \
  --smoking_status 'never smoked' --bmi 30 --avg_glucose_level 200"
```

Salida: *Probabilidad de ictus = 0.8855 (88.6%) → RIESGO ALTO*.

**Ejemplo real (API):**

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"age":75,"avg_glucose_level":200,"bmi":30,"gender":"Male","ever_married":"Yes",\
       "work_type":"Private","residence_type":"Urban","smoking_status":"never smoked",\
       "hypertension":1,"heart_disease":1}'
```

Salida: `{"probabilidad_ictus":0.8855,"clase":1,"riesgo":"ALTO"}`

---

# Segunda parte — Parte técnica

## Implementaciones importantes

- **Pipeline de preprocesado + modelo** (scikit-learn `Pipeline`): one-hot encoding de categóricas, estandarización de numéricas y `passthrough` de binarias, con el modelo al final. Previene **data leakage** (D6).
- **Tratamiento del desbalance**: `scale_pos_weight` (CatBoost) y `class_weight` / SMOTE (comparativa). El desbalance (4.98 % de ictus) hace que la precisión sea estructuralmente baja (0.146) a cambio de un recall alto.
- **Control de overfitting**: reducción de iteraciones, profundidad (`depth`), `min_data_in_leaf`, `l2_leaf_reg` y `bagging_temperature`. Verificado con `scripts/compare_train_test.py` y con el test automático `test_final_model_meets_overfitting_requirement`.
- **Validación cruzada**: `StratifiedKFold(5)` sobre el train en el entrenamiento del modelo final (`train_catboost_regularized.py`), reportando media ± desv de recall/precisión/F1 por configuración (decisión D5.1).
- **Optimización de hiperparámetros**: `GridSearchCV` (scikit-learn) con rejilla de 6 hiperparámetros, scoring **F1** y CV estratificada dentro de la misma pipeline (sin data leakage) (decisión D5.2).
- **CLI** (`argparse`): `scripts/predict_cli.py`, validación de argumentos y veredicto legible.
- **API REST** (FastAPI + Pydantic): modelo de entrada tipado (`PatientData`), validación de dominios (género, trabajo, fumador…), respuestas tipadas (`PredictionResponse`) y error 422 ante entradas inválidas.
- **Naming**: ver [Convenciones de nombres](#convenciones-de-nombres).

## Tecnologías utilizadas

| Capa | Tecnología | Uso |
|---|---|---|
| EDA | Pandas, Seaborn, Matplotlib, missingno | notebooks 01–06, informe 07 |
| Modelos | Scikit-learn, CatBoost, XGBoost, imbalanced-learn | entrenamiento y comparativa |
| API | FastAPI, Uvicorn, Pydantic | `BACKEND/main.py` |
| CLI | argparse | `scripts/predict_cli.py` |
| Tests | Pytest | `tests/` |
| Tooling/entorno | uv, Makefile, Jupytext | dependencias y orquestación |

## Diseño de la base de datos

Hay **dos orígenes de datos**:

1. **`data/stroke_dataset.csv`** (datos de entrenamiento, solo lectura): 4981 filas × 11 columnas, cargado con pandas en scripts y modelo.
2. **Base de datos de predicciones** (historial de la API): tabla `predictions` donde se guarda cada resultado de `POST /predict`. Usa **PostgreSQL** (`DATABASE_URL`) en Docker/producción y **SQLite** (`data/predictions.db`) por defecto en local/tests (decisión D11). Se consulta con `GET /predictions`.

### Modelo entidad-relación (E-R)

Base de datos de **una sola entidad** (`predictions`): una predicción de ictus. No hay relaciones entre entidades ni claves foráneas (el modelo predice sobre un paciente puntual).

```
PREDICTION (tabla predictions)
 ┌───────────────────────────────────────────────────────────┐
 │ id (PK, autoincremental)                                  │
 │ ── Entrada (features del paciente) ──                     │
 │ age, gender, hypertension, heart_disease, ever_married,   │
 │ work_type, Residence_type, avg_glucose_level, bmi,        │
 │ smoking_status                                            │
 │ ── Salida (resultado del modelo) ──                       │
 │ probabilidad (0–1), clase (0/1), riesgo ("ALTO"/"BAJO"),  │
 │ created_at (fecha/hora)                                   │
 └───────────────────────────────────────────────────────────┘
 Relaciones: ninguna · PK: id · FK: no hay
```

Columnas de la tabla `predictions`:

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | serial/int **PK** | Identificador autogenerado |
| `age` | float | Edad en años |
| `gender` | str | Male / Female / Other |
| `hypertension` | int | 0/1 |
| `heart_disease` | int | 0/1 |
| `ever_married` | str | Yes / No |
| `work_type` | str | Private, Self-employed, Govt_job, children, Never_worked |
| `Residence_type` | str | Urban / Rural |
| `avg_glucose_level` | float | Glucosa media (mg/dL) |
| `bmi` | float | Índice de masa corporal |
| `smoking_status` | str | never smoked / formerly smoked / smokes / Unknown |
| `probabilidad` | float | Probabilidad de ictus devuelta por el modelo |
| `clase` | int | 0/1 (riesgo) |
| `riesgo` | str | `ALTO` / `BAJO` |
| `created_at` | timestamp | Fecha y hora de la predicción |

## Paradigma de desarrollo, estructura de carpetas y patrones de diseño

- **Paradigma**: desarrollo **funcional por script** (cada `scripts/*.py` es autónomo y ejecutable), con **programación orientada a objetos solo donde aporta** (pydantic `BaseModel`, `Pipeline` de scikit-learn). Se sigue el principio **DRY** en la definición de features (`NUMERIC`/`CATEGORICAL`/`BINARY` repetidos deliberadamente en cada script para mantenerlos autocontenidos).
- **Patrones**:
  - **Pipeline** de scikit-learn (preprocesado + modelo) → encapsula y evita data leakage.
  - **Bootstrap/entrypoint único** en la API: `BACKEND/main.py` carga el modelo al arrancar (`joblib.load`).
  - **Config por constantes de módulo** (`THRESHOLD`, `RANDOM_STATE`, `MAX_DIFF`, `FEATURES`).
- **Estructura de carpetas**: ver [Estructura del código](#estructura-del-código).

## Convenciones de nombres

| Elemento | Convención | Ejemplo |
|---|---|---|
| Scripts | `snake_case.py`, prefijo de acción (`train_`, `predict_`, `compare_`) | `train_catboost_regularized.py` |
| Variables | `snake_case` | `avg_glucose_level`, `X_train` |
| Constantes | `MAYÚSCULAS_SNAKE` | `MAX_DIFF`, `RANDOM_STATE`, `THRESHOLD` |
| Funciones | `snake_case`, verbos | `build_pipeline()`, `parse_args()`, `predict()` |
| Clases (pydantic) | `UpperCamelCase` | `PatientData`, `PredictionResponse` |
| Columnas | respetan el nombre original del CSV | `Residence_type` |
| Archivos de modelos | `make train-*` → `models/<modelo>.pkl` | `catboost_final.pkl` |

## Organización del repositorio, ramas y commits

**Workflow**: desarrollo por ramas *feature* fusionadas en `dev`; `main` solo al cierre.

```
main  ─────────────────────────── (solo al final del proyecto)
dev   ◄── models ◄── informe ◄── fastapi ◄── validacioncruzada ◄── readme
```

| Rama | Contenido |
|---|---|
| `main` | Entrega final |
| `dev` | Rama de integración (base de trabajo) |
| `eda` / `paths` | Análisis exploratorio y correcciones de rutas |
| `models` | Entrenamiento de modelos y tests base |
| `informe` | Informe + feature importance |
| `fastapi` | API de productivización y sus tests |
| `validacioncruzada` | Validación cruzada y optimización (GridSearchCV) del modelo final |
| `readme` | Documentación del proyecto |

**Commits**: mensajes **descriptivos en español**, estilo *convencional* (`feat/`, `fix/`, `chore/`, `docs/`):

```text
feat/optimizacion: GridSearchCV + validación cruzada (D5.1–D5.2)
chore/completar tarea: validación cruzada del modelo final
fix/readme con lo que sí está hecho
feat/tests: tests funcionales de la API FastAPI y limpieza de __pycache__
entrenamiento de modelos y tests pasados
```

## Documentación general y de la API

**Documentación general del proyecto**

| Documento | Ubicación |
|---|---|
| Este README | `README.md` |
| Decisiones de diseño (D1–D11) | `SDD/02.Scope_anchored.md` |
| Informe de rendimiento del modelo | `models/INFORME_MODELOS.md` |
| Informe de tests | `tests/INFORME_TESTS.md` |
| Informe EDA / conclusiones | `notebooks/07.Informe.md` |

**Documentación de la API**

- Arranque: `make api` → `http://127.0.0.1:8000`
- **Swagger interactivo**: `http://127.0.0.1:8000/docs` (generado por FastAPI).
- Endpoints:

| Método | Ruta | Descripción | Códigos |
|---|---|---|---|
| `GET` | `/health` | Estado y nombre del modelo cargado | 200 |
| `POST` | `/predict` | Predicción de riesgo de ictus | 200, 422 (dato inválido) |

**Cuerpo de `POST /predict`** (esquema `PatientData`):

```json
{
  "age": 75.0,
  "avg_glucose_level": 200.0,
  "bmi": 30.0,
  "gender": "Male",
  "ever_married": "Yes",
  "work_type": "Private",
  "residence_type": "Urban",
  "smoking_status": "never smoked",
  "hypertension": 1,
  "heart_disease": 1
}
```

**Respuesta** (esquema `PredictionResponse`):

```json
{ "probabilidad_ictus": 0.8855, "clase": 1, "riesgo": "ALTO" }
```

---

## Modelo final y requisitos cumplidos

**Modelo elegido: CatBoost regularizado** ("med": `iterations=100`, `depth=3`, `min_data_in_leaf=8`, `l2_leaf_reg=8`, `bagging_temperature=1`, `scale_pos_weight=19.12`).

| Métrica | Valor en test |
|---|---|
| Recall | **0.86** |
| Precision | 0.146 |
| F1 | 0.250 |
| \|train − test\| (todas) | ≤ 1.4 puntos ✅ |

La selección se hace con **validación cruzada (`StratifiedKFold`, 5 folds)** y **GridSearchCV (scoring F1)**: ambas cubren el requisito de CV y de optimización con herramienta de tuning. La mejor combinación del grid (F1 CV 0.235) sobreajusta (diff recall 10.9 pts > 5), así que prevalece el requisito del trabajo y gana **"med"** (mejor F1 medio de CV = 0.225 entre las que cumplen).

**Análisis de características** (feature importance): `age` concentra el **64.2 %** de la importancia; le siguen `bmi` (11.0 %) y `avg_glucose_level` (9.4 %).

### Requisitos de la asignatura

| Requisito | Estado |
|---|---|
| Modelo ML funcional que prediga riesgo de ictus | ✅ `catboost_final.pkl` |
| EDA con gráficos y estadísticas | ✅ notebooks 01–07 |
| Modelo de ML con técnicas de **ensemble** | ✅ CatBoost (y comparativa con RF/XGBoost) |
| Uso de **validación cruzada** | ✅ `StratifiedKFold(5)` en el entrenamiento del modelo final (D5.1) |
| **Mitigar el desbalance** (4.98 % positivos) | ✅ `scale_pos_weight` + comparativa con `class_weight`/SMOTE |
| **Optimización de hiperparámetros** con herramienta de tuning | ✅ GridSearchCV (D5.2) |
| Overfitting \|train − test\| ≤ 5 puntos | ✅ CatBoost final ≤ 1.4 pts, verificado por test |
| **Test unitarios** | ✅ 24 tests (`make test`) |
| Aplicación de línea de comandos | ✅ `scripts/predict_cli.py` |
| Solución que productivice el modelo | ✅ API FastAPI (`BACKEND/main.py`) |
| Informe con precisión/recall/F1/AUC-ROC + características | ✅ `models/INFORME_MODELOS.md` (secc. 4.1) |
| Repo Git con ramas organizadas y commits limpios | ✅ flujo feature → `dev` → `main` |
| Documentación y README | ✅ este documento |

---

## Cómo empezar

```bash
# 1. Instalar dependencias (uv)
uv sync

# 2. (Opcional) Reentrenar el modelo final (validación cruzada + GridSearchCV)
make train-cat     # o todos: make all

# 3. Verificar el requisito de overfitting
make compare

# 4. Ejecutar los tests
make test         # 24 passed

# 5a. Predecir por CLI
make predict ARGS="--age 75 --gender Male --hypertension 1 ..."

# 5b. Lanzar la API
make api          # → http://127.0.0.1:8000/docs
```

---

*Proyecto 8 · Data Science / AI Developer · Bootcamp IA·MAD · 2026*
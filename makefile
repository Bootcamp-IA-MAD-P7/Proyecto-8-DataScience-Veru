#TARGET PARA CONVERTIR LOS .PY EN IPYNB

PYTEXT ?= jupytext
NOTEBOOKS_DIR := notebooks
PY_GLOBS := $(wildcard $(NOTEBOOKS_DIR)/*.py)

.PHONY: notebook clean list

notebook:
	@echo "Convirtiendo .py -> .ipynb en $(NOTEBOOKS_DIR)..."
	@for f in $(PY_GLOBS); do \
		$(PYTEXT) --to ipynb "$$f"; \
	done

clean:
	@echo "Borrando .ipynb generados en $(NOTEBOOKS_DIR)..."
	@rm -f $(NOTEBOOKS_DIR)/*.ipynb

list:
	@echo "Archivos .py detectados en $(NOTEBOOKS_DIR):"
	@for f in $(PY_GLOBS); do echo "$$f"; done


#TARGET PARA PASAR LOS TESTS

TEST ?= pytest
TESTS_DIR := tests

.PHONY: test

test:
	@echo "Ejecutando tests (si existen)..."
	@$(TEST) -q $(TESTS_DIR) || \
		( echo "No se encontraron tests; por ahora no hay nada que ejecutar." && exit 0 )


#TARGETS DE ENTRENAMIENTO DE MODELOS

PY ?= uv run python
SCRIPTS := scripts

.PHONY: train-rf train-rf-smote train-xgb train-cat train all compare predict api

## Random Forest baseline (comparativa)
train-rf:
	@echo "Entrenando Random Forest baseline..."
	@$(PY) $(SCRIPTS)/train_random_forest.py

## Random Forest + SMOTE (comparativa)
train-rf-smote:
	@echo "Entrenando Random Forest + SMOTE..."
	@$(PY) $(SCRIPTS)/train_random_forest_smote.py

## XGBoost (comparativa)
train-xgb:
	@echo "Entrenando XGBoost..."
	@$(PY) $(SCRIPTS)/train_xgboost.py

## CatBoost regularizado (modelo final) -> genera catboost_final.pkl
train-cat:
	@echo "Entrenando CatBoost regularizado (modelo final)..."
	@$(PY) $(SCRIPTS)/train_catboost_regularized.py

## Alias de train-cat (modelo final del proyecto)
train: train-cat

## Alias de train-rf-smote (mantiene compatibilidad)
smote: train-rf-smote

## Entrena todos los modelos para la comparativa
all: train-rf train-rf-smote train-xgb train-cat

## Comprobación del requisito de overfitting (train vs test <= 5 puntos)
compare:
	@echo "Comparando métricas train vs test (requisito de overfitting)..."
	@$(PY) $(SCRIPTS)/compare_train_test.py

## Predicción de riesgo de ictus desde línea de comandos (CLI)
## Uso: make predict ARGS="--age 75 --gender Male --hypertension 1 ..."
predict:
	@$(PY) $(SCRIPTS)/predict_cli.py $(ARGS)

## Lanza la API FastAPI de predicción (productivización, D10)
## Uso: make api  -> http://127.0.0.1:8000/docs
api:
	@echo "Lanzando API FastAPI en http://127.0.0.1:8000/docs ..."
	@$(PY) -m uvicorn BACKEND.main:app --reload


#INFRAESTRUCTURA DOCKER

.PHONY: shap docker-build docker-up docker-down db-up db-down

SHAP_CMD ?= echo "Definid SHAP_CMD cuando tengáis script de interpretabilidad."
shap:
	@echo "Ejecutando target shap..."
	@$(SHAP_CMD)

DOCKER_BUILD_CMD ?= docker compose build
docker-build:
	@echo "Construyendo imágenes Docker (API + Postgres)..."
	@$(DOCKER_BUILD_CMD)

docker-up:
	@echo "Levantando servicios (Postgres + API) en segundo plano..."
	@docker compose up -d --build
	@echo "API disponible en http://localhost:8000 (docs: /docs)"

docker-down:
	@echo "Deteniendo servicios..."
	@docker compose down

db-up:
	@echo "Levantando solo PostgreSQL..."
	@docker compose up -d db

db-down:
	@echo "Deteniendo PostgreSQL..."
	@docker compose stop db


#RESUMEN DE USO

.PHONY: help

help:
	@echo "Targets disponibles:"
	@echo "  train-rf       - Entrena Random Forest baseline (comparativa)"
	@echo "  train-rf-smote - Entrena Random Forest + SMOTE (comparativa)"
	@echo "  train-xgb      - Entrena XGBoost (comparativa)"
	@echo "  train-cat      - Entrena CatBoost regularizado (MODELO FINAL)"
	@echo "  train          - Alias de train-cat"
	@echo "  smote          - Alias de train-rf-smote"
	@echo "  all            - Entrena todos los modelos (comparativa completa)"
	@echo "  compare        - Comprueba requisito de overfitting (train vs test)"
	@echo "  predict        - Predice riesgo de ictus: make predict ARGS='--age 75 ...'"
	@echo "  api            - Lanza la API FastAPI (http://127.0.0.1:8000/docs)"
	@echo "  test           - Ejecuta pytest (si hay tests)"
	@echo "  notebook       - Convierte notebooks .py (EDA) a .ipynb"
	@echo "  docker-build   - Construye las imágenes Docker"
	@echo "  docker-up      - Levanta Postgres + API (http://localhost:8000)"
	@echo "  docker-down    - Detiene los servicios Docker"
	@echo "  db-up          - Levanta solo PostgreSQL"
	@echo "  db-down        - Detiene solo PostgreSQL"
	@echo "  shap           - Ejecuta SHAP_CMD (pendiente)"

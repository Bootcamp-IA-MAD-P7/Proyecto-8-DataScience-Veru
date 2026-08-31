#TARGET PARA CONVERTIR LOS .PY EN IPYNB

PYTEXT ?= jupytext
NOTEBOOKS_DIR := notebooks
PY_GLOBS := $(wildcard $(NOTEBOOKS_DIR)/*.py)

.PHONY: notebook clean

notebook:
	@echo "Convirtiendo .py -> .ipynb en $(NOTEBOOKS_DIR)..."
	@for f in $(PY_GLOBS); do \
		$(PYTEXT) --to ipynb "$$f"; \
	done

clean:
	@echo "Borrando .ipynb generados en $(NOTEBOOKS_DIR)..."
	@rm -f $(NOTEBOOKS_DIR)/*.ipynb


.PHONY: list
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



.PHONY: train docker-build

train:
	@echo "TARGET train: pendiente (hoy solo estamos preparando Makefile)."

DOCKER_BUILD_CMD ?= docker compose build
docker-build:
	@echo "Ejecutando target docker-build..."
	@$(DOCKER_BUILD_CMD)


train-smote:
	@echo "TARGET train-smote: pendiente. Ejecutará train con SMOTE activado."

.PHONY: train

TRAIN_CMD ?= uv run python scripts/train_random_forest.py
train:
	@echo "Ejecutando target train..."
	@$(TRAIN_CMD)

.PHONY: explain shap smote train-smote

SHAP_CMD ?= echo "Definid SHAP_CMD cuando tengáis script de interpretabilidad."
SMOTE_CMD ?= uv run python scripts/train_random_forest_smote.py

shap:
	@echo "Ejecutando target shap..."
	@$(SHAP_CMD)

smote:
	@echo "Ejecutando target smote..."
	@$(SMOTE_CMD)

train-smote:
	@echo "Ejecutando target train-smote..."
	@$(SMOTE_CMD)

#RESUMEN DE USO

.PHONY: help

help:
	@echo "Targets disponibles:"
	@echo "  notebook     - Convierte notebooks .py (EDA) a .ipynb"
	@echo "  test         - Ejecuta pytest (si hay tests)"
	@echo "  train        - Ejecuta TRAIN_CMD (definir cuando el lunes sepáis el comando)"
	@echo "  docker-build - Construye imágenes Docker (pendiente)"
	@echo "  shap         - Ejecuta SHAP_CMD (pendiente)"
	@echo "  smote        - Ejecuta SMOTE_CMD (pendiente)"


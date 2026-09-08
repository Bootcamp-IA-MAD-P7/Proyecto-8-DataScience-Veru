# Informe de Tests

> Estado: suite **verde — 18 passed, 3 skipped** (ejecutado con `make test`)

Los tests automatizados verifican de forma objetiva que los **requisitos del trabajo** se cumplen. Se ejecutan con:

```bash
make test        # equivale a: uv run pytest -q tests/
```

## Resumen

| Resultado | Cantidad | Descripción |
|---|---|---|
| ✅ Passed | 18 | Requisitos implementados (dataset, preprocesado, modelo final, informe, **CLI**) |
| ⏭️ Skipped | 3 | Requisitos **pendientes** (API/productivización) — se activarán al implementarla |
| ❌ Failed | 0 | — |

## Cobertura de requisitos

| Requisito del trabajo | Test | Estado |
|---|---|---|
| Análisis exploratorio (EDA) con gráficos y descriptivos | — (documentado en notebooks/07.Informe.md) | ✅ hecho |
| **Control de overfitting** (\|train − test\| ≤ 5 puntos) | `test_final_model_meets_overfitting_requirement` | ✅ **PASSED** |
| Informe de rendimiento (precisión, recall, F1, AUC-ROC) | `test_report_exists`, `test_report_mentions_required_metrics[...]` | ✅ PASSED |
| Aplicación en línea de comandos (CLI) | `test_cli_script_exists`, `test_cli_returns_prediction` | ✅ **PASSED** (implementado) |
| Productivización (API / Streamlit / Gradio / Dash) | `test_api_module_exists`, `test_api_loads`, `test_api_imports_final_model` | ⏭️ **SKIPPED** (pendiente) |

## Detalle: tests implementados (`tests/test_model.py`)

### Dataset (`data/stroke_dataset.csv`)
| Test | Qué verifica |
|---|---|
| `test_dataset_shape` | 4981 filas × 11 columnas |
| `test_dataset_no_nulls` | Sin valores nulos |
| `test_dataset_no_duplicates` | Sin filas duplicadas |
| `test_target_in_dataset` | Existe la columna `stroke` |
| `test_stroke_proportion` | Target binario y proporción de ictus ~5 % (desbalanceado) |

### Preprocesado
| Test | Qué verifica |
|---|---|
| `test_preprocess_features` | Las features usadas son exactamente las definidas (3 numéricas + 5 categóricas + 2 binarias) |

### Modelo final (`models/catboost_final.pkl`)
| Test | Qué verifica |
|---|---|
| `test_final_model_exists` | El modelo final existe |
| `test_final_model_can_predict` | Carga y produce probabilidades sobre un paciente sintético |
| `test_final_model_meets_overfitting_requirement` | **Requisito clave**: \|train − test\| ≤ 0.05 en recall, precisión y F1 |

### Informe de rendimiento (`models/INFORME_MODELOS.md`)
| Test | Qué verifica |
|---|---|
| `test_report_exists` | El informe existe |
| `test_report_mentions_required_metrics[...]` | Menciona Recall, Precisión, F1, AUC-ROC, CatBoost y overfitting |

## Detalle: tests pendientes (`tests/test_pending.py`)

Cubren el requisito aún por implementar (la **API de productivización**).
Están marcados con `@pytest.mark.skip` para que la suite siga en verde mientras
se desarrolla; al crear la API solo hay que retirar el decorador.

| Test | Requisito al que pertenece |
|---|---|
| `test_cli_script_exists` | CLI — que exista `scripts/predict_cli.py` (**implementado, activo**) |
| `test_cli_returns_prediction` | CLI — que devuelva una predicción a partir de argumentos (**implementado, activo**) |
| `test_api_module_exists` | API — que exista `app/main.py` (FastAPI) |
| `test_api_loads` | API — que el módulo importe sin errores |
| `test_api_imports_final_model` | API — que cargue el modelo final |

## Cómo ejecutar

```bash
make test          # suite completa (pytest -q tests/)
uv run pytest tests/ -v   # con detalle de cada test
```

## Salida de referencia (última ejecución)

```
tests/test_model.py::test_dataset_shape PASSED
tests/test_model.py::test_dataset_no_nulls PASSED
tests/test_model.py::test_dataset_no_duplicates PASSED
tests/test_model.py::test_target_in_dataset PASSED
tests/test_model.py::test_stroke_proportion PASSED
tests/test_model.py::test_preprocess_features PASSED
tests/test_model.py::test_final_model_exists PASSED
tests/test_model.py::test_final_model_can_predict PASSED
tests/test_model.py::test_final_model_meets_overfitting_requirement PASSED
tests/test_model.py::test_report_exists PASSED
tests/test_model.py::test_report_mentions_required_metrics[...] PASSED
tests/test_pending.py::test_cli_* PASSED
tests/test_pending.py::test_api_* SKIPPED
==================== 18 passed, 3 skipped in ~6.4s ====================
```

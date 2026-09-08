# INFORME DE MODELOS DE PREDICCIÓN DE ICTUS — Modelo final: CatBoost

> Documento de resultados del proceso de modelado. Fecha: 08/09/2026. Dataset: `data/stroke_dataset.csv` (n = 4981, 248 ictus = 4.98 %).

---

## 1. Resumen ejecutivo

Se probaron varios algoritmos y técnicas de balanceo para predecir si un paciente está en riesgo de ictus. Todas las configuraciones usan el mismo preprocesado (one-hot para categóricas, estandarización para numéricas), partición train/test estratificada (80/20) y los mismos criterios de evaluación.

El **modelo final elegido es CatBoost con regularización** (configuración "med": `iterations=100`, `depth=3`, `min_data_in_leaf=8`, `l2_leaf_reg=8`, `bagging_temperature=1`, `scale_pos_weight=19.12`), porque es el que **cumple el requisito del trabajo: la diferencia entre métricas de entrenamiento y de test es inferior a 5 puntos porcentuales** (en todas las métricas de clasificación).

---

## 2. Recorrido de modelos (de menor a mayor rendimiento)

### 2.1 Random Forest (baseline y con SMOTE)

| Configuración | Recall | Precision | F1 | AUC-ROC |
|---|---|---|---|---|
| Random Forest baseline | 0.060 | 0.107 | 0.077 | 0.818 |
| Random Forest + SMOTE | 0.100 | 0.111 | 0.105 | 0.813 |

**Lectura:** con el umbral 0.5, el RF apenas detecta ictus (6–10 % de recall). El `class_weight` no basta y SMOTE apenas ayuda. Además, en la comprobación train-vs-test, todos los RF mostraban **fuerte sobreajuste** (recall 1.000 en train frente a 0.06–0.10 en test, diferencias de 40–90 puntos).

### 2.2 XGBoost (baseline y tunado)

| Configuración | Recall | Precision | F1 | AUC-ROC |
|---|---|---|---|---|
| XGBoost baseline (scale_pos_weight) | 0.160 | 0.211 | 0.182 | 0.808 |
| XGBoost tunado (min_child_weight=3, umbral 0.15) | 0.480 | 0.164 | 0.245 | — |

**Lectura:** XGBoost supera claramente al RF y, al **bajar el umbral** de 0.5 a 0.15, el recall sube de 0.16 a 0.48. Esto confirmó que el umbral 0.5 estaba mal situado en datos desbalanceados. No obstante, XGBoost también sobreajustaba con el umbral 0.5 (diferencias train-test > 5 puntos).

### 2.3 CatBoost (baseline)

| Umbral | Recall | Precision | F1 |
|---|---|---|---|
| 0.15 | 0.56 | 0.132 | 0.214 |
| 0.10 | 0.70 | 0.138 | 0.230 |

**Lectura:** CatBoost con `scale_pos_weight` detecta el 70 % de los ictus con umbral 0.10, el mejor recall hasta ese momento. Cumple el objetivo de criba en cuanto a recall, aunque con baja precisión (muchos falsos positivos), algo inherente al desbalance y a señales débiles.

### 2.4 CatBoost + SMOTE

| Umbral | Recall | Precision | F1 |
|---|---|---|---|
| 0.15 | 0.86 | 0.113 | 0.199 |
| 0.10 | 0.90 | 0.106 | 0.189 |

**Lectura:** añadir SMOTE dispara el recall (hasta 0.90) pero **reduce la precisión** (más falsos positivos), y además **empeora el control de overfitting**. Por eso se descartó como modelo final.

---

## 3. Control de overfitting (requisito del trabajo)

**Requisito:** la diferencia entre métricas de training y de test debe ser **inferior a 5 puntos porcentuales** (|train − test| ≤ 0.05) en las métricas de clasificación.

La comprobación con el umbral 0.5 mostró que **todos los modelos iniciales sobreajustaban** (diferencias de 40–90 puntos, con recall 1.000 en train). Para corregirlo se **reduce la complejidad y se añade regularización** (menos iteraciones, profundidad limitada, `min_data_in_leaf`, `l2_leaf_reg`, `bagging_temperature`).

### Resultado de regularización sobre CatBoost (umbral 0.5)

| Configuración | Recall train | Recall test | \|Diff recall\| | ¿Cumple? |
|---|---|---|---|---|
| light (100, depth 4) | 0.879 | 0.800 | 0.079 | ❌ |
| **med (100, depth 3)** | 0.874 | 0.860 | **0.014** | ✅ |
| strong (60, depth 2) | 0.864 | 0.880 | 0.016 | ✅ |
| very-strong (40, depth 2) | 0.869 | 0.880 | 0.011 | ✅ |

La configuración **"med"** cumple el requisito con holgura en **todas** las métricas:

| Métrica | Train | Test | \|Diff\| |
|---|---|---|---|
| Recall | 0.874 | 0.860 | **0.014** |
| Precision | 0.142 | 0.146 | 0.004 |
| F1 | 0.244 | 0.250 | 0.006 |

- Máxima diferencia: **1.4 puntos porcentuales** (muy por debajo del límite de 5).
- Además mantiene un **recall de test alto (0.86)**: detecta 86 de cada 100 ictus.

### Validación cruzada (StratifiedKFold, 5 folds)

Además de la partición train/test, cada configuración se evalúa con **validación cruzada estratificada de 5 folds sobre el train** (`scripts/train_catboost_regularized.py`, decisión D5.1). Resultados **media ± desviación**:

| Configuración | Recall (CV) | Precisión (CV) | F1 (CV) |
|---|---|---|---|
| light (iter=100, depth=4) | 0.753 ± 0.066 | 0.137 ± 0.006 | 0.232 ± 0.011 |
| **med (iter=100, depth=3)** | 0.793 ± 0.060 | 0.131 ± 0.010 | **0.225 ± 0.016** |
| strong (iter=60, depth=2) | 0.838 ± 0.075 | 0.121 ± 0.010 | 0.212 ± 0.017 |
| very-strong (iter=40, depth=2) | 0.859 ± 0.076 | 0.118 ± 0.010 | 0.208 ± 0.017 |

La selección del modelo final combina ambos criterios: cumple el requisito de overfitting (tabla anterior) y tiene el **mejor F1 medio de CV entre las configuraciones que lo cumplen** → **"med"** (F1 CV 0.225).

---

## 4. Modelo final y cómo interpretarlo

**Modelo elegido: CatBoost "med"** (guardado en `models/catboost_final.pkl`).

| Métrica | Valor en test |
|---|---|
| Recall (sensibilidad) | **0.86** |
| Precision | 0.146 |
| F1 | 0.250 |
| Diff train-test (todas) | ≤ 1.4 puntos ✅ |

### Cómo interpretar los valores

- **Recall 0.86** = de cada 100 pacientes que realmente tuvieron ictus, el modelo detecta **86**. Es la métrica clave para un cribado: no queremos dejar escapar ictus.
- **Precision 0.146** = de cada 100 pacientes marcados "en riesgo", **~15** tuvieron ictus real; el resto son falsos positivos (avisos a revisar). Es baja porque las señales del dataset apenas separan las clases y, al buscar alto recall, se asumen más falsos positivos (ver trade-off en sección 5).
- **F1 0.250** = equilibrio global low, dominado por la baja precisión.
- **Control de overfitting ✅** = el modelo generaliza: lo que aprende en train se mantiene en test, requisito del trabajo cumplido.

---

## 4.1 Análisis de las características más importantes

Importancia de características del modelo final (CatBoost, importancia _lossfunction_ sobre las 19 columnas tras el preprocesado, agrupadas por variable original):

| Variable | Importancia |
|---|---|
| **age** | **64.2 %** |
| bmi | 11.0 % |
| avg_glucose_level | 9.4 % |
| hypertension | 3.3 % |
| smoking_status | 3.6 % |
| work_type | 4.6 % |
| gender | 1.6 % |
| ever_married | 0.8 % |
| heart_disease | 0.7 % |
| Residence_type | 0.8 % |

**Lectura:** `age` concentra **casi dos tercios de la importancia** del modelo. Es coherente con el EDA (07.Informe.md), donde la edad era la variable con mayor separación entre grupos (corr. 0.246 con ictus). Le siguen, muy por debajo, **bmi** y **avg_glucose_level** (variables relacionadas con el perfil metabólico). El resto aporta poco: `hypertension`, `smoking_status` y `work_type` contribuyen marginalmente, y `gender`, `ever_married`, `heart_disease` y `Residence_type` son casi irrelevantes para el modelo.

**Implicación:** la señal de riesgo de ictus en estos datos proviene fundamentalmente de la **edad** (y en menor medida del perfil metabólico). La baja precisión (0.146) está ligada a que el resto de variables aportan apenas información: la vía de mejora real sería añadir variables más informativas (colesterol, HbA1c/diabetes, presión arterial), tal como se anota en las conclusiones.

---

## 5. Inconvenientes encontrados y soluciones

| Inconveniente | Qué pasa | Solución aplicada |
|---|---|---|
| **Fuerte desbalance** (4.98 % de positivos) | El modelo tiende a predecir "no ictus" y el recall cae | `scale_pos_weight` (ponderar clase minoritaria) |
| **Umbral 0.5 mal situado** | Con desbalance, cortar en 0.5 descarta casi todos los positivos | Ajustar umbral (bajar) cuando convenga |
| **Overfitting severo** | Todos los modelos iniciales daban recall 1.0 en train vs 0.06–0.16 en test | Reducir complejidad + regularización (CatBoost "med") |
| **SMOTE no aislado** | Con RF se combinó SMOTE + class_weight sin aislar su aporte; con CatBoost+SMOTE bajó precisión y empeoró overfitting | Se descarta SMOTE; CatBoost "med" sin SMOTE da mejor equilibrio |
| **Pocos positivos (248)** | Métricas de precisión muy ruidosas | Se prioriza recall (criba) y se valida con train-vs-test |
| **Señales débiles** (solo `age` con peso real · corr. 0.246) | Techo de separabilidad; la precisión no puede ser alta a recall alto | Se acepta la precisión baja inherente; la vía de mejora real sería más/m ejores variables |

---

## 6. Conclusiones

1. **El modelo final es CatBoost regularizado** (`models/catboost_final.pkl`), elegido por cumplir el **requisito de overfitting** (diferencias train-test ≤ 1.4 puntos, límite 5), ofrecer un **recall alto (0.86)** y validarse mediante **validación cruzada estratificada (StratifiedKFold, 5 folds)**.
2. **La precisión (0.146) es baja** por el desbalance y la debilidad de las señales; es un límite estructural, no un defecto del algoritmo (trade-off precisión-recall).
3. El proceso siguió un **plan por fases** documentado en el SDD (D3.1–D3.3): umbral → tuning → CatBoost → SMOTE → sobreajuste, descartando las opciones que no cumplían el requisito o que empeoraban la fiabilidad, y añadiendo **validación cruzada** en el entrenamiento del modelo final (D5.1).
4. **Vías de mejora futuras** (si se dispusiera de más tiempo/datos): incorporar variables más informativas (colesterol, HbA1c/diabetes, presión arterial, consumo de alcohol), o ingeniería de características, lo que subiría la precisión a igual recall.

---

## 7. Reproducibilidad (scripts y makefile)

Los entrenamientos están orquestados en el **makefile** para que la comparativa sea reproducible:

| Comando make | Qué ejecuta |
|---|---|
| `make train-rf` | Entrena Random Forest baseline → `models/random_forest_baseline.pkl` |
| `make train-rf-smote` | Entrena Random Forest + SMOTE → `models/random_forest_smote.pkl` |
| `make train-xgb` | Entrena XGBoost → `models/xgboost_baseline.pkl` |
| `make train-cat` | Entrena CatBoost regularizado con validación cruzada (`StratifiedKFold`, 5 folds) y guarda el modelo final → `models/catboost_final.pkl` |
| `make all` | Entrena los cuatro modelos (comparativa completa) |
| `make compare` | Comprueba el requisito de overfitting (train vs test) en todos |

Scripts asociados:
- `scripts/train_random_forest.py`, `scripts/train_random_forest_smote.py`, `scripts/train_xgboost.py` → comparativa
- `scripts/train_catboost_regularized.py` → modelo final (incluye validación cruzada, véase D5.1)
- `scripts/compare_train_test.py` → verificación del requisito de overfitting

> Nota: solo se persisten en `models/` el modelo final (`catboost_final.pkl`) y el informe; los `.pkl` de los modelos descartados se regeneran con su script (`make train-*`), evitando acumular artefactos pesados.

---

## 8. Referencias metodológicas

Las referencias del EDA (07.Informe.md) sustentan el enfoque: en datos desbalanceados no se usa accuracy y se priorizan precision/recall/F1/AUC; un modelo que no trate el desbalance puede dar accuracy alto (~95 %) con recall 0 en la clase minoritaria.

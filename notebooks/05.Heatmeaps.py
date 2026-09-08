# %% 
# import sys
import sys
from pathlib import Path
ROOT = Path.cwd().parent  # sube de notebooks/ a la raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
# %%
df = pd.read_csv(ROOT / "data" / "stroke_dataset.csv")

# [1] Calcular la matriz de correlación (solo numéricas)
correlation = df.corr(numeric_only=True)

# [2] Heatmap de correlaciones
plt.figure(figsize=(10, 8))
# annot=True muestra los números dentro de cada celda
# cmap='coolwarm' usa colores: rojo=positivo, azul=negativo
# center=0 centra la escala de colores en cero
# vmin/vmax fijan la escala entre -1 y 1
# linewidths=1 añade líneas entre celdas para mejor legibilidad
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
            vmin=-1, vmax=1,
            linewidths=1, fmt='.2f', square=True)
plt.title('Matriz de correlación - Stroke Dataset (variables numéricas)')
plt.tight_layout()
plt.show()

# %%
# VARIANTE CON VARIABLES CATEGÓRICAS CODIFICADAS
# Codifica las categóricas (ordinal/one-hot) para incluir las no numéricas
# en la matriz de correlación.
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

df_corr = df.copy()
for col in cat_cols:
    df_corr[col] = pd.factorize(df_corr[col])[0]  # codificación ordinal simple

plt.figure(figsize=(12, 10))
corr_all = df_corr.corr(numeric_only=True)
sns.heatmap(corr_all, annot=True, cmap='coolwarm', center=0,
            vmin=-1, vmax=1,
            linewidths=0.5, fmt='.2f', square=True)
plt.title('Matriz de correlación - Stroke Dataset (numéricas + categóricas codificadas)')
plt.tight_layout()
plt.show()
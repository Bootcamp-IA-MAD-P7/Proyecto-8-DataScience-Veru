# %% 
# import sys
import sys
from pathlib import Path

ROOT = Path.cwd().parent  # sube de notebooks/ a la raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib

# %%
df = pd.read_csv(ROOT / "data" / "stroke_dataset.csv")
df.head(10)
# %%
df.shape
# %%
df.shape
print(f"Columns ({len(df.columns)}): {df.columns}")
# %%
print(f"Types: {df.dtypes}")
print(f"Columns ({len(df.columns)}): {df.columns}")
# %%
df.describe
# %%
df.isnull().sum()
# %% Contar duplicados
df.duplicated()
# %% Eliminar duplicados
print(f"Duplicados: {df.drop_duplicates()}")
# %% 
df.columns.tolist()
# %% 
df.describe(include='object')
# %%
df.head()

# %%
import missingno as msngno
msngno.matrix(df)
plt.show()

# %%
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0])

print("Dataset después de eliminar columnas:")
df.head()

# %%
media = df['age'].mean()
print("Edad - media:", media)
mediana = df['age'].median()
print("Edad - mediana:", mediana)
desviacion_estandar = df['age'].std()  # ddof=1 (pandas), coherente con el informe
print("Edad - desviación estándar:", desviacion_estandar)
# %%
# Detección de valores extraños tipo "sin dato" que no son NaN
# (caso smoking_status = "Unknown"). Comparación insensible a mayúsculas.
sospechosos = ["?", "", "none", "n/a", "-", "unknown", "  ", "null", "nan"]
for col in df.columns:
    coincidencias = df[col].astype(str).str.lower().isin(sospechosos).sum()
    if coincidencias > 0:
        print(f"{col}: {coincidencias} valores raros")

# 2) Columnas constantes (no aportan información, caso veil-type).
constantes = [col for col in df.columns if df[col].nunique() == 1]
print("\nColumnas constantes:", constantes)
# %%

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
media = np.mean(df['age'])
print(media)
desviacion_estandar = np.std(df['age'])
print(desviacion_estandar)
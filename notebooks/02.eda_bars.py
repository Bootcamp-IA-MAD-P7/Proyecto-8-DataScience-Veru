# %% 
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
df = pd.read_csv(ROOT / "data" / "stroke_dataset.csv")
# %%
print(df['gender'].value_counts())
df['gender'].value_counts().plot(kind='bar', color=['purple', 'green'])
plt.title('Gender Distribution')  # Distribución por género
plt.xlabel('gender')  # Género (male/female)
plt.ylabel('Count')  # Cantidad de personas
plt.xticks(rotation=0)  # Etiquetas horizontales
plt.show()
# %%
print(df['ever_married'].value_counts())
df['ever_married'].value_counts().plot(kind='bar', color=['purple', 'green'])
plt.title('Matrimonio')  
plt.xlabel('ever_married')  
plt.ylabel('Count')  
plt.xticks(rotation=0) 
plt.show()
# %%
print(df['stroke'].value_counts())
df['stroke'].value_counts().plot(kind='bar', color=['purple', 'green'])
plt.title('Stroke')  
plt.xlabel('stroke')  
plt.ylabel('Count')  
plt.xticks(rotation=0) 
plt.show()
# %%
print(df['ever_married'].value_counts())
df['ever_married'].value_counts().plot(kind='bar', color=['purple', 'green'])
plt.title('Matrimonio')  
plt.xlabel('ever_married')  
plt.ylabel('Count')  
plt.xticks(rotation=0) 
plt.show()


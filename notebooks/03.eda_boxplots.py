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
# %%
df.boxplot(column='age', patch_artist=True, 
           boxprops=dict(facecolor='lightgreen'))
plt.title('Age Distribution')  
plt.ylabel('age')  
plt.show()
# %%
df.boxplot(column='avg_glucose_level', patch_artist=True, 
           boxprops=dict(facecolor='lightgreen'))
plt.title('Avg Glucose Level Distribution') 
plt.ylabel('avg_glucose_level')  
plt.show()
# %%
df.boxplot(column='bmi', patch_artist=True, 
           boxprops=dict(facecolor='lightgreen'))
plt.title('BMI Distribution') 
plt.ylabel('bmi')  
plt.show()

# %%
# Boxplots por grupo (ictus vs no ictus): más informativos que el univariante,
# muestran cómo se distribuye cada variable según el resultado.
sns.boxplot(x='stroke', y='age', hue='stroke', data=df, palette='Set2', legend=False)
plt.title('Age by Stroke')
plt.xlabel('stroke')
plt.ylabel('age')
plt.show()

# %%
sns.boxplot(x='stroke', y='avg_glucose_level', hue='stroke', data=df, palette='Set2', legend=False)
plt.title('Avg Glucose Level by Stroke')
plt.xlabel('stroke')
plt.ylabel('avg_glucose_level')
plt.show()

# %%
sns.boxplot(x='stroke', y='bmi', hue='stroke', data=df, palette='Set2', legend=False)
plt.title('BMI by Stroke')
plt.xlabel('stroke')
plt.ylabel('bmi')
plt.show()

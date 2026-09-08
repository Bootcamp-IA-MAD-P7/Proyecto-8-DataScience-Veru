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
df['age'].plot(kind='hist', bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.title('Age Distribution')  
plt.xlabel('age') 
plt.ylabel('Frequency')  
plt.show()

# %%
df['avg_glucose_level'].plot(kind='hist', bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.title('lucose Level Distribution')  
plt.xlabel('avg_glucose_level Distribution') 
plt.ylabel('Frequency')  
plt.show()

# %%
df['bmi'].plot(kind='hist', bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.title('Bmi Distribution')  
plt.xlabel('bmi') 
plt.ylabel('Frequency')  
plt.show()


# %%
sns.histplot(df['work_type'], bins=20, kde=True)
plt.title('Type work Distribution')
plt.show()


# %%
sns.histplot(df['avg_glucose_level'], bins=20, kde=True)

plt.title('Histograma de avg_glucose_level')

plt.show()


# %%
sns.histplot(df['bmi'], bins=20, kde=True)
plt.title('Bmi distribution')
plt.show()






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
# Pairplot solo sobre variables numéricas y coloreando por ictus (stroke)
g = sns.pairplot(
    df,
    vars=['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease'],
    hue='stroke',
    palette='Set1'
)
g.fig.suptitle("Variables numéricas según ictus", y=1.02)
plt.show()
# %%

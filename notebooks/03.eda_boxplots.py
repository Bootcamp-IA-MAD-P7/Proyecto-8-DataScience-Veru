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
plt.title('Hypertension Distribution')  
plt.ylabel('age')  
plt.show()
# %%
df.boxplot(column='avg_glucose_level', patch_artist=True, 
           boxprops=dict(facecolor='lightgreen'))
plt.title('Hypertension Distribution') 
plt.ylabel('avg_glucose_level')  
plt.show()
# %%
df.boxplot(column='bmi', patch_artist=True, 
           boxprops=dict(facecolor='lightgreen'))
plt.title('Hypertension Distribution') 
plt.ylabel('avbmig_glucose_level')  
plt.show()

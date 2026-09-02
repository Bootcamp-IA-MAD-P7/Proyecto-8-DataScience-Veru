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

# Visualizar relaciones entre todas las variables numéricas
# hue='Survived' colorea los puntos según si sobrevivieron (verde) o no (rojo)
# palette={0: 'red', 1: 'green'} define los colores específicos
# Este gráfico puede tardar un poco porque crea muchas visualizaciones
sns.pairplot(numeric_df, hue='Survived', palette={0: 'red', 1: 'green'}, 
             diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Pairplot of Numeric Variables by Survival', y=1.02)  # Título general
plt.show()
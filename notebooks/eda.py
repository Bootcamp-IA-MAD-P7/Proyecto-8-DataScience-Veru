# %% 
# import sys
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # notebooks/ -> raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")


df = pd.read_csv

df.head()


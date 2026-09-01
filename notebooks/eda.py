# %% 
# import sys
import sys
from pathlib import Path

ROOT = Path.cwd().parent  # sube de notebooks/ a la raíz del repo
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from utils.load_raw_data import RawData
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")

data_raw = RawData(year=2025)
df = data_raw.download()
df.head(10)
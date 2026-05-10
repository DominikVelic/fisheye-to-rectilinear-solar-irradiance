import pandas as pd
from pathlib import Path


METEO_DATA_CLEANED = "meteo_data_cleaned.csv"
DATA_DIR = Path("data/")

dfs = []
for split in ("test", "train", "val"):
    split_csv = DATA_DIR / split / METEO_DATA_CLEANED
    dfs.append(pd.read_csv(split_csv))

data = pd.concat(dfs)

print(data.describe())

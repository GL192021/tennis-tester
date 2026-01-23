import os
import re
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

years = []
combined_data = {}
s=0
for fname in os.listdir(RAW):
    if fname.lower().startswith('atp_matches'):
        if not fname.lower().endswith("toy_example.csv"):
            match = re.search(r"atp_matches_(\d{4})", fname)
            if match:
                year = int(match.group(1))
                years.append(year)
                atp_y_path = os.path.join(RAW, f"atp_matches_{year}.csv")
                atp_y = pd.read_csv(atp_y_path)
                s += len(atp_y)
                columns_lst = atp_y.columns
                for col in columns_lst:
                    lst = combined_data.get(col, [])
                    lst = lst + list(atp_y[col])
                    combined_data[col] = lst

start_year = min(years)
end_year = max(years)

pd.DataFrame(combined_data).to_csv(os.path.join(OUT, f"combined_{start_year}-{end_year}.csv"), index=False)

















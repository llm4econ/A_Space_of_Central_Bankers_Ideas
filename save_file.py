import os
import pandas as pd
from pandas_datareader import data as pdr

os.environ["FRED_API_KEY"] = "0480db599027440c5b79abd092014378"

fred_id = "CP0000EZ19M086NEST"
start_date = "2000-01-01"
end_date = "2025-01-01"

df = pdr.DataReader(fred_id, "fred", start_date, end_date)
df.rename(columns={fred_id: "Euro_Area_CPI"}, inplace=True)

file_path = "fred_data.csv"

df.to_csv(file_path, index=True)
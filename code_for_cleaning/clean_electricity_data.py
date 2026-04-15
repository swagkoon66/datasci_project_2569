import numpy as np
import pandas as pd

# Open file with relative path from 
# Source - https://stackoverflow.com/a/32470564
# Posted by Jared Mackey, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-13, License - CC BY-SA 3.0

import os

cur_path = os.path.dirname(__file__)

# filepath_1 = os.path.relpath('..\\raw_data\\dataset_11_35.csv', cur_path)
# filepath_2 = os.path.relpath('..\\raw_data\\dataset_11_36.csv', cur_path)
filepath_3 = os.path.relpath('..\\raw_data\\dataset_11_37.csv', cur_path)

# df1:pd.DataFrame
# df2:pd.DataFrame

# All province infromation
df3:pd.DataFrame 

def read_raw_csv (filepath:str):
    temp:pd.DataFrame
    with open(filepath, 'rb') as f:
        try:
            temp = pd.read_csv(f,encoding="utf-8")
            print(f"READ FILE: {f.name}\n")
            # print(df1.dtypes)
        except Exception as e:
            print(e.__traceback__)
        # finally:
        #     print("Read!")
    return temp

df3 = read_raw_csv(filepath_3)
# print(df3.describe())

# CLEANING

def clean_electricity_df(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    # ---- STEP 1: Create proper date ----
    df_clean["date"] = pd.to_datetime(
        df_clean["Year"].astype(str) + "-" + df_clean["Month"] + "-01"
    )

    # ---- STEP 2: Rename columns ----
    df_clean = df_clean.rename(columns={
        "Sector": "type",
        "Quantity": "electricity_consumption_GWh"
    })

    # ---- STEP 3: Convert GWh → kWh ----
    df_clean["electricity_consumption_kWh"] = (
        df_clean["electricity_consumption_GWh"] * 1_000_000
    )

    # ---- STEP 4: Format date ----
    df_clean["date"] = df_clean["date"].dt.strftime("%Y-%m-%d")

    # ---- STEP 5: Select final columns ----
    df_final = df_clean[[
        "date",
        "type",
        "electricity_consumption_kWh"
    ]]

    return df_final

df3_cleaned = clean_electricity_df(df3)

print(df3_cleaned.columns)
print(df3_cleaned.head())

# EXPORT FUNCTION

def export_clean_df(df: pd.DataFrame, filename: str):
    # Create target folder path relative to current file
    export_dir = os.path.relpath('..\\cleaned_data', cur_path)

    # Ensure directory exists
    os.makedirs(export_dir, exist_ok=True)

    # Full file path
    full_path = os.path.join(export_dir, filename)

    # Export CSV
    df.to_csv(full_path, index=False, encoding="utf-8")

    print(f"EXPORTED FILE TO: {full_path}")

# EXPORT df3_cleaned
export_clean_df(df3_cleaned, "df3_dataset_11_37_clean.csv")
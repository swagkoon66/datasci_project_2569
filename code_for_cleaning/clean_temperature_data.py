import os
import pandas as pd
import json
# READ JSON FUNCTION
cur_path = os.path.dirname(__file__)
filepath1 = os.path.relpath('..\\raw_data\\THA_1950_2014.json', cur_path)
filepath2 = os.path.relpath('..\\raw_data\\THA_2014_2100.json', cur_path)

# MERGE CLIMATE JSON FILES
def merge_climate_json(file1_path: str, file2_path: str, output_path: str):

    # Load both files
    with open(file1_path, "r", encoding="utf-8") as f1:
        data1 = json.load(f1)

    with open(file2_path, "r", encoding="utf-8") as f2:
        data2 = json.load(f2)

    # Start with first file as base
    merged = data1.copy()

    # Navigate into structure
    for variable in data2["data"]:  # e.g., "tas", "tasmax"

        if variable not in merged["data"]:
            merged["data"][variable] = data2["data"][variable]
            continue

        for country in data2["data"][variable]:  # e.g., "THA"

            if country not in merged["data"][variable]:
                merged["data"][variable][country] = data2["data"][variable][country]
                continue

            # Merge time series (this is the key part)
            merged["data"][variable][country].update(
                data2["data"][variable][country]
            )

    # Save merged JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4)

    print(f"MERGED JSON SAVED TO: {output_path}")

merge_climate_json(filepath1,filepath2,(os.path.relpath('..\\cleaned_data\\THA_1950_2100.json', cur_path)))

# JSON → DATAFRAME
def json_to_dataframe(filepath: str) -> pd.DataFrame:
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    # Navigate structure
    for variable, var_data in data["data"].items():  # tas, tasmax
        
        for country, time_series in var_data.items():  # THA
            
            for date, value in time_series.items():
                records.append({
                    "date": date,
                    "country": country,
                    "variable": variable,
                    "value": value
                })

    df = pd.DataFrame(records)

    # Convert date format
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return df

final_pd = json_to_dataframe((os.path.relpath('..\\cleaned_data\\THA_1950_2100.json', cur_path)))

def export_df_to_csv(df: pd.DataFrame, filename: str):
    export_dir = os.path.relpath('..\\cleaned_data')

    os.makedirs(export_dir, exist_ok=True)

    full_path = os.path.join(export_dir, filename)

    df.to_csv(full_path, index=False, encoding="utf-8")

    print(f"EXPORTED TO: {full_path}")

export_df_to_csv(final_pd,(os.path.relpath('..\\cleaned_data\\THA_1950_2100.csv', cur_path)))
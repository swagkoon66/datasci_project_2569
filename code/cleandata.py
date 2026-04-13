import numpy as np
import pandas as pd

# Open file with relative path from 
# Source - https://stackoverflow.com/a/32470564
# Posted by Jared Mackey, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-13, License - CC BY-SA 3.0

import os

cur_path = os.path.dirname(__file__)

filepath_1 = os.path.relpath('..\\raw_data\\dataset_11_35.csv', cur_path)
filepath_2 = os.path.relpath('..\\raw_data\\dataset_11_36.csv', cur_path)
filepath_3 = os.path.relpath('..\\raw_data\\dataset_11_37.csv', cur_path)

df1:pd.DataFrame
df2:pd.DataFrame
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

# use only this two files
df1 = read_raw_csv(filepath_1)
df2 = read_raw_csv(filepath_2)

print(df1.head())
print(df2.head())

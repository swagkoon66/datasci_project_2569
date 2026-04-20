import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import os

# 1. Load Datasets
curr_dir = os.path.dirname(__file__)
elec_df = pd.read_csv(os.path.relpath('..\\cleaned_data\\df3_dataset_11_37_clean.csv',curr_dir))
temp_df = pd.read_csv(os.path.relpath('..\\cleaned_data\\THA_1950_2100.csv',curr_dir))

# Preprocessing: Convert dates and filter for 'Residential'
elec_df['date'] = pd.to_datetime(elec_df['date'])
temp_df['date'] = pd.to_datetime(temp_df['date'])

sector_types = ["Residential", "Business","Industrial", "Government & Non-Profit", "Agriculture", "Other", "Free of Charge"]

elec_res = elec_df[elec_df['type'] == sector_types[0]].sort_values('date')
temp_tas = temp_df[temp_df['variable'] == 'tas'][['date', 'value']].rename(columns={'value': 'temperature'})

# Merge consumption and temperature data
df = pd.merge(elec_res, temp_tas, on='date', how='inner')
df = df[(df['date'] >= '2002-01-01') & (df['date'] <= '2025-12-31')]

# 2. Feature Engineering
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['lag_1'] = df['electricity_consumption_kWh'].shift(1) # Previous month's consumption
df = df.dropna() # Drop the first row due to lag

# 3. Model Training
# Define features and target
features = ['year', 'month', 'temperature', 'lag_1']
X = df[features]
y = df['electricity_consumption_kWh']

# Split: Train on data before 2024, Test on 2024-2025
train_mask = df['date'] < '2024-01-01'
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]

# Initialize and fit RandomForest
model = RandomForestRegressor(n_estimators=100, random_state=42, bootstrap=True)
model.fit(X_train, y_train)
# Generate predictions for the entire period to visualize trend
df['predicted_kWh'] = model.predict(X)

# 4. Visualization
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['electricity_consumption_kWh'], label='Actual Consumption', color='blue', alpha=0.6)
plt.plot(df['date'], df['predicted_kWh'], label='Model Prediction/Trend', color='red', linestyle='--')
plt.axvline(pd.to_datetime('2024-01-01'), color='black', linestyle=':', label='Forecast Start (2024)')

plt.title('Residential Electricity Consumption Trend & Prediction (2002-2025)')
plt.xlabel('Year')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
# Need to save first before show (if close == no save)
plt.tight_layout()
plt.savefig('randomforest.png')
plt.show()

# Print performance
mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
print(f"Model MAPE (2024-2025): {mape:.2%}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
import os

# 1. Load and Prepare Data
curr_dir = os.path.dirname(__file__)
elec_df = pd.read_csv(os.path.relpath('..\\cleaned_data\\df3_dataset_11_37_clean.csv',curr_dir))
temp_df = pd.read_csv(os.path.relpath('..\\cleaned_data\\THA_1950_2100.csv',curr_dir))

elec_df['date'] = pd.to_datetime(elec_df['date'])
temp_df['date'] = pd.to_datetime(temp_df['date'])

# Filter for Residential
elec_res = elec_df[elec_df['type'] == 'Residential'].sort_values('date')

# Pivot temperature to get average (tas) and max (tasmax) temperatures
temp_pivot = temp_df[temp_df['variable'].isin(['tas', 'tasmax'])].pivot(
    index='date', columns='variable', values='value').reset_index()

# Merge consumption and temperature
df = pd.merge(elec_res, temp_pivot, on='date', how='inner')
df = df[(df['date'] >= '2002-01-01') & (df['date'] <= '2025-12-31')]

# 2. Feature Engineering
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['lag_1'] = df['electricity_consumption_kWh'].shift(1)
df['lag_12'] = df['electricity_consumption_kWh'].shift(12)  # Seasonal lag
df['rolling_mean_3'] = df['electricity_consumption_kWh'].shift(1).rolling(window=3).mean()
df['CDD'] = df['tas'].apply(lambda x: max(0, x - 25))  # Cooling Degree Days proxy

df = df.dropna()

# 3. Gradient Boosting Model
features = ['year', 'month', 'tas', 'tasmax', 'CDD', 'lag_1', 'lag_12', 'rolling_mean_3']
X = df[features]
y = df['electricity_consumption_kWh']

# Train on data before 2024
train_mask = df['date'] < '2024-01-01'
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)
df['predicted_kWh'] = model.predict(X)

# 4. Visualization & Saving Plot
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['electricity_consumption_kWh'], label='Actual Consumption', color='blue', alpha=0.5, linewidth=2)
plt.plot(df['date'], df['predicted_kWh'], label='GBM predicted_kWh', color='green', linestyle='--', alpha=0.8)
plt.axvline(pd.to_datetime('2024-01-01'), color='red', linestyle=':', label='Forecast Start (2024)')

plt.title('Improved Residential Electricity Consumption: Gradient Boosting Model', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Consumption (kWh)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('improved_gradient_model.png')
plt.show()
# Print performance
mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
print(f"Model MAPE (2024-2025): {mape:.2%}")
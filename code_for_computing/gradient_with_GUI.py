import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

curr_dir = os.path.dirname(__file__)

class ElectricityForecastGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Electricity Consumption Forecasting - Gradient Boosting")
        self.root.geometry("1200x800")
        
        # Load and prepare data
        self.load_data()
        
        # Create GUI elements
        self.create_widgets()
        
    def load_data(self):
        """Load and prepare the dataset"""
        # Load data
        elec_df = pd.read_csv(os.path.relpath('..\\cleaned_data\\df3_dataset_11_37_clean.csv',curr_dir))
        temp_df = pd.read_csv(os.path.relpath('..\\cleaned_data\\THA_1950_2100.csv',curr_dir))
        
        elec_df['date'] = pd.to_datetime(elec_df['date'])
        temp_df['date'] = pd.to_datetime(temp_df['date'])
        
        # Filter for Residential
        elec_res = elec_df[elec_df['type'] == 'Residential'].sort_values('date')
        
        # Pivot temperature
        temp_pivot = temp_df[temp_df['variable'].isin(['tas', 'tasmax'])].pivot(
            index='date', columns='variable', values='value').reset_index()
        
        # Merge
        df = pd.merge(elec_res, temp_pivot, on='date', how='inner')
        df = df[(df['date'] >= '2002-01-01') & (df['date'] <= '2025-12-31')]
        
        # Feature Engineering
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['lag_1'] = df['electricity_consumption_kWh'].shift(1)
        df['lag_12'] = df['electricity_consumption_kWh'].shift(12)
        df['rolling_mean_3'] = df['electricity_consumption_kWh'].shift(1).rolling(window=3).mean()
        df['CDD'] = df['tas'].apply(lambda x: max(0, x - 25))
        
        self.df = df.dropna()
        self.features = ['year', 'month', 'tas', 'tasmax', 'CDD', 'lag_1', 'lag_12', 'rolling_mean_3']
        
        # Get date range
        self.min_date = self.df['date'].min()
        self.max_date = self.df['date'].max()
        
    def create_widgets(self):
        """Create GUI widgets"""
        # Control Panel Frame
        control_frame = ttk.LabelFrame(self.root, text="Training Configuration", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Method Selection
        ttk.Label(control_frame, text="Split Method:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        self.method_var = tk.StringVar(value="date")
        ttk.Radiobutton(control_frame, text="Date Range Split", variable=self.method_var, 
                       value="date", command=self.update_method).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Percentage Split", variable=self.method_var, 
                       value="percentage", command=self.update_method).pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="Time Series Cross-Validation", variable=self.method_var, 
                       value="tscv", command=self.update_method).pack(anchor=tk.W, pady=(0, 10))
        
        # Date Range Selection (for date method)
        self.date_frame = ttk.LabelFrame(control_frame, text="Date Range", padding=5)
        self.date_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.date_frame, text="Training Start:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.train_start_var = tk.StringVar(value=self.min_date.strftime('%Y-%m-%d'))
        self.train_start_entry = ttk.Entry(self.date_frame, textvariable=self.train_start_var, width=15)
        self.train_start_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(self.date_frame, text="Training End:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.train_end_var = tk.StringVar(value='2023-12-31')
        self.train_end_entry = ttk.Entry(self.date_frame, textvariable=self.train_end_var, width=15)
        self.train_end_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(self.date_frame, text="Test Start:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.test_start_var = tk.StringVar(value='2024-01-01')
        self.test_start_entry = ttk.Entry(self.date_frame, textvariable=self.test_start_var, width=15)
        self.test_start_entry.grid(row=2, column=1, padx=5, pady=2)
        
        ttk.Label(self.date_frame, text="Test End:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.test_end_var = tk.StringVar(value=self.max_date.strftime('%Y-%m-%d'))
        self.test_end_entry = ttk.Entry(self.date_frame, textvariable=self.test_end_var, width=15)
        self.test_end_entry.grid(row=3, column=1, padx=5, pady=2)
        
        # Percentage Split (for percentage method)
        self.pct_frame = ttk.LabelFrame(control_frame, text="Percentage Split", padding=5)
        
        ttk.Label(self.pct_frame, text="Training Size (%):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.train_pct_var = tk.DoubleVar(value=80.0)
        self.train_pct_scale = ttk.Scale(self.pct_frame, from_=50, to=95, 
                                        variable=self.train_pct_var, orient=tk.HORIZONTAL, length=150)
        self.train_pct_scale.grid(row=0, column=1, padx=5, pady=2)
        self.train_pct_label = ttk.Label(self.pct_frame, text="80.0%")
        self.train_pct_label.grid(row=0, column=2, pady=2)
        self.train_pct_var.trace('w', self.update_pct_label)
        
        # Time Series CV options
        self.tscv_frame = ttk.LabelFrame(control_frame, text="Time Series CV", padding=5)
        
        ttk.Label(self.tscv_frame, text="Number of Splits:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.n_splits_var = tk.IntVar(value=5)
        ttk.Spinbox(self.tscv_frame, from_=2, to=10, textvariable=self.n_splits_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        # Model Parameters
        params_frame = ttk.LabelFrame(control_frame, text="Model Parameters", padding=5)
        params_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(params_frame, text="N Estimators:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.n_estimators_var = tk.IntVar(value=200)
        ttk.Spinbox(params_frame, from_=50, to=500, increment=50, 
                   textvariable=self.n_estimators_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(params_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.DoubleVar(value=0.05)
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(params_frame, text="Max Depth:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.depth_var = tk.IntVar(value=5)
        ttk.Spinbox(params_frame, from_=3, to=10, textvariable=self.depth_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Train Model", command=self.train_model, 
                  style="Accent.TButton").pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Save Plot", command=self.save_plot).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Export Results", command=self.export_results).pack(fill=tk.X, pady=2)
        
        # Results Display
        results_frame = ttk.LabelFrame(control_frame, text="Results", padding=5)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.results_text = tk.Text(results_frame, height=10, width=35, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Plot Frame
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize with date method
        self.update_method()
        
    def update_method(self):
        """Update visible frames based on selected method"""
        if self.method_var.get() == "date":
            self.date_frame.pack(fill=tk.X, pady=5)
            self.pct_frame.pack_forget()
            self.tscv_frame.pack_forget()
        elif self.method_var.get() == "percentage":
            self.date_frame.pack_forget()
            self.pct_frame.pack(fill=tk.X, pady=5)
            self.tscv_frame.pack_forget()
        else:  # tscv
            self.date_frame.pack_forget()
            self.pct_frame.pack_forget()
            self.tscv_frame.pack(fill=tk.X, pady=5)
    
    def update_pct_label(self, *args):
        """Update percentage label"""
        self.train_pct_label.config(text=f"{self.train_pct_var.get():.1f}%")
    
    def train_model(self):
        """Train the model based on selected method"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Training model...\n")
        self.root.update()
        
        try:
            X = self.df[self.features]
            y = self.df['electricity_consumption_kWh']
            
            method = self.method_var.get()
            
            if method == "date":
                # Date range method
                train_start = pd.to_datetime(self.train_start_var.get())
                train_end = pd.to_datetime(self.train_end_var.get())
                test_start = pd.to_datetime(self.test_start_var.get())
                test_end = pd.to_datetime(self.test_end_var.get())
                
                train_mask = (self.df['date'] >= train_start) & (self.df['date'] <= train_end)
                test_mask = (self.df['date'] >= test_start) & (self.df['date'] <= test_end)
                
                X_train, y_train = X[train_mask], y[train_mask]
                X_test, y_test = X[test_mask], y[test_mask]
                
                self.results_text.insert(tk.END, f"\nMethod: Date Range Split\n")
                self.results_text.insert(tk.END, f"Training: {train_start.date()} to {train_end.date()}\n")
                self.results_text.insert(tk.END, f"Testing: {test_start.date()} to {test_end.date()}\n")
                self.results_text.insert(tk.END, f"Train samples: {len(X_train)}\n")
                self.results_text.insert(tk.END, f"Test samples: {len(X_test)}\n\n")
                
            elif method == "percentage":
                # Percentage split using sklearn
                train_size = self.train_pct_var.get() / 100.0
                
                # For time series, we don't shuffle
                split_idx = int(len(X) * train_size)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                train_mask = self.df.index < split_idx
                test_mask = self.df.index >= split_idx
                
                self.results_text.insert(tk.END, f"\nMethod: Percentage Split ({train_size*100:.1f}% train)\n")
                self.results_text.insert(tk.END, f"Train samples: {len(X_train)}\n")
                self.results_text.insert(tk.END, f"Test samples: {len(X_test)}\n\n")
                
            else:  # Time Series Cross-Validation
                n_splits = self.n_splits_var.get()
                tscv = TimeSeriesSplit(n_splits=n_splits)
                
                self.results_text.insert(tk.END, f"\nMethod: Time Series CV ({n_splits} splits)\n\n")
                
                mapes = []
                for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    model = GradientBoostingRegressor(
                        n_estimators=self.n_estimators_var.get(),
                        learning_rate=self.lr_var.get(),
                        max_depth=self.depth_var.get(),
                        random_state=42
                    )
                    
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    mape = mean_absolute_percentage_error(y_test, predictions)
                    mapes.append(mape)
                    
                    self.results_text.insert(tk.END, f"Fold {i+1}: MAPE = {mape:.2%}\n")
                
                self.results_text.insert(tk.END, f"\nAverage MAPE: {np.mean(mapes):.2%}\n")
                self.results_text.insert(tk.END, f"Std Dev: {np.std(mapes):.2%}\n")
                
                # Use last split for visualization
                train_mask = self.df.index.isin(train_idx)
                test_mask = self.df.index.isin(test_idx)
            
            # Train final model
            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators_var.get(),
                learning_rate=self.lr_var.get(),
                max_depth=self.depth_var.get(),
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions on all data for visualization
            self.df['prediction'] = model.predict(X)
            
            # Calculate metrics
            train_mape = mean_absolute_percentage_error(y_train, model.predict(X_train))
            test_mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
            
            if method != "tscv":
                self.results_text.insert(tk.END, f"Training MAPE: {train_mape:.2%}\n")
                self.results_text.insert(tk.END, f"Testing MAPE: {test_mape:.2%}\n")
            
            # Feature importance
            feature_imp = pd.DataFrame({
                'feature': self.features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.results_text.insert(tk.END, "\nTop 5 Features:\n")
            for idx, row in feature_imp.head().iterrows():
                self.results_text.insert(tk.END, f"  {row['feature']}: {row['importance']:.4f}\n")
            
            # Plot results
            self.plot_results(train_mask, test_mask)
            
            self.results_text.insert(tk.END, "\n✓ Training complete!")
            
        except Exception as e:
            self.results_text.insert(tk.END, f"\n✗ Error: {str(e)}")
    
    def plot_results(self, train_mask, test_mask):
        """Plot the results"""
        self.ax.clear()
        
        # Plot actual data
        self.ax.plot(self.df['date'], self.df['electricity_consumption_kWh'], 
                    label='Actual Consumption', color='blue', alpha=0.5, linewidth=2)
        
        # Plot predictions
        self.ax.plot(self.df['date'], self.df['prediction'], 
                    label='GBM Prediction', color='green', linestyle='--', alpha=0.8, linewidth=2)
        
        # Highlight train/test split
        if train_mask is not None:
            train_dates = self.df.loc[train_mask, 'date']
            test_dates = self.df.loc[test_mask, 'date']
            
            if len(train_dates) > 0 and len(test_dates) > 0:
                split_date = test_dates.min()
                self.ax.axvline(split_date, color='red', linestyle=':', 
                              label=f'Test Start: {split_date.date()}', linewidth=2)
        
        self.ax.set_title('Residential Electricity Consumption: Gradient Boosting Model', 
                         fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Date', fontsize=11)
        self.ax.set_ylabel('Consumption (kWh)', fontsize=11)
        self.ax.legend(loc='upper left')
        self.ax.grid(True, which='both', linestyle='--', alpha=0.3)
        
        # Format x-axis
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        self.ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_plot(self):
        """Save the current plot"""
        try:
            # filename = f"gradient_boost_GUI_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            filename = f"gradient_boost_GUI_plot.png"
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.results_text.insert(tk.END, f"\n✓ Plot saved as {filename}")
        except Exception as e:
            self.results_text.insert(tk.END, f"\n✗ Error saving plot: {str(e)}")
    
    def export_results(self):
        """Export predictions to CSV"""
        try:
            if 'prediction' not in self.df.columns:
                self.results_text.insert(tk.END, "\n✗ Please train model first!")
                return
            
            filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            export_df = self.df[['date', 'electricity_consumption_kWh', 'prediction']].copy()
            export_df.to_csv(filename, index=False)
            self.results_text.insert(tk.END, f"\n✓ Results exported to {filename}")
        except Exception as e:
            self.results_text.insert(tk.END, f"\n✗ Error exporting: {str(e)}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ElectricityForecastGUI(root)
    root.mainloop()
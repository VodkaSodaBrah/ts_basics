import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_loader import load_bnb_data

# Fit & evaluate a seasonal ARIMA model
def run_sarima(train_frac: float = 0.8, # 80% id data for training
               order: tuple = (1, 1, 1), # Non-seasonal ARIMA part: p = 1 > look back 1 step at the raw series (Auto-Regressive lag). d = 1 > difference the series once to remove trend. q = 1 > look back 1 step at the models past forecast errors (Moving-Average lag). Params (p, d, q).
               seasonal_order: tuple = (1, 1, 1, 96), # Seasonal ARIMA part: P = 1 > 1 seasonal AR term. D = 1 > seasonal differencing once. Q = 1 > seasonal MA term. s = 96 >  the season repeats every 96 observations 96 x 15-min bars = 1 day). Params (P, D, Q, s).
               resample_freq: str = '15T', # Groups the original 1-minute data into coarser bars. The time step for the model.
               maxiter: int = 50): # Upper limit on optimizer iterations when the model is fit; higher = more thorough but slower convergence. Caps how long the fitting routine is allowed to search for the best parameters.
    """
    Fits a SARIMA model on the 'close' price series and evaluates the forecast.
    - train_frac: fraction of data used for training
    - order: (p, d, q) non-seasonal ARIMA parameters
    - seasonal_order: (P, D, Q, s) seasonal ARIMA parameters
    - resample_freq: frequency string for downsampling (e.g. '15T')
    - maxiter: maximum optimizer iterations
    Returns (metrics, test_series, forecast_series, fitted_model).
    """
    # Load the price history into a table
    df = load_bnb_data()
    # Group data into chunks and fill missing spots
    series = df['close'].resample(resample_freq).last().ffill()

    # Decide how much data is for training vs. testing
    split_idx = int(len(series) * train_frac)
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]

    # Set up the ARIMA model rules
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    # Train the model, but stop early if it takes too long
    fit = model.fit(disp=False, maxiter=maxiter)

    # Make predictions for the test data
    forecast = fit.forecast(steps=len(test))

    # Measure how close predictions are to actual values
    metrics = {
        'mae': mean_absolute_error(test, forecast),
        'rmse': np.sqrt(mean_squared_error(test, forecast)),
        'aic': fit.aic,
        'bic': fit.bic
    }
    # Give back error numbers, true prices, predicted prices, and the model itself
    return metrics, test, forecast, fit

# If you run this file, do the analysis and print the results
if __name__ == '__main__':
    m, y_true, y_pred, model = run_sarima()
    print("SARIMA results:")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")
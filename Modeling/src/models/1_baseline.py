import os
import sys

# ensure src/ is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_bnb_data

def run_baseline(train_frac: float = 0.8):
    """
    Fits a random-walk ARIMA model (0,1,0) on the 'close' series and evaluates forecast.
    - train_frac: fraction of data to train
    Returns (metrics, test_series, naive_forecast, rw_forecast, fitted_model).
    """
    # 1. Load and prepare the data
    df = load_bnb_data()

    # Extract the 'close' price series for forecasting
    series = df['close']

    # Split the series into training and test sets based on train_frac
    split_idx = int(len(series) * train_frac)
    train, test = series.iloc[:split_idx], series.iloc[split_idx:]

    # Naïve forecast: repeat the last value from the training set
    naive_forecast = np.repeat(train.iloc[-1], len(test))

    # ARIMA(0,1,0) random-walk model: difference once, no AR/MA terms
    model = ARIMA(train, order=(0, 1, 0))
    model_fit = model.fit()

    # Forecast the next len(test) steps using the fitted ARIMA model
    rw_forecast = model_fit.forecast(steps=len(test))

    # Compute evaluation metrics and information criteria
    metrics = {
        'mae_naive': mean_absolute_error(test, naive_forecast),  # MAE of the naive forecast
        'rmse_naive': np.sqrt(mean_squared_error(test, naive_forecast)),  # RMSE of the naive forecast
        'mae_rw': mean_absolute_error(test, rw_forecast),  # MAE of the random-walk ARIMA forecast
        'rmse_rw': np.sqrt(mean_squared_error(test, rw_forecast)),  # RMSE of the random-walk ARIMA forecast
        'aic_rw': model_fit.aic,  # AIC of the fitted ARIMA model
        'bic_rw': model_fit.bic  # BIC of the fitted ARIMA model
    }
    return metrics, test, naive_forecast, rw_forecast, model_fit

# When run as a script, execute the baseline and print the results
if __name__ == '__main__':
    m, y_true, y_naive, y_rw, model = run_baseline()
    print("Baseline ARIMA(0,1,0) results:")
    for k, v in m.items():
        print(f"{k}: {v:.4f}")
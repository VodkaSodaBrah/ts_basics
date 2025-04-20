import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib
import mplfinance as mpf
import os
# Ensure the plots directory exists
plots_dir = '/Users/mchildress/ts_basics/plots'
os.makedirs(plots_dir, exist_ok=True)
# Force a GUI backend on macOS so that plt.show() opens windows
matplotlib.use('MacOSX')
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the dataset
data_path = '/Users/mchildress/ts_basics/data/bnbusdt_1m.csv'

# Read CSV, parse 'open_time' as datetime index
df = pd.read_csv(
    data_path,
    parse_dates=['open_time'],
    index_col='open_time'
)
# Use the 'close' column as the price series
price = df['close']
# --- Subset for heavy computations to avoid OOM ---
# Use only the last 10080 minutes (~1 week) for decomposition, stationarity, and ACF/PACF
subset = price.tail(10080)
# Debug: confirm data loaded correctly
print("DataFrame shape:", df.shape)
print("Price series head:\n", price.head())


# Plot candlestick chart (OHLC + volume)
ohlc = df[['open', 'high', 'low', 'close', 'volume']]
# Limit to last 1440 rows (one day's worth of 1m candles) to avoid plotting excessive data
ohlc = ohlc.tail(1440)
mpf.plot(
    ohlc,
    type='candle',
    mav=(5, 10),
    volume=True,
    title='BNB/USDT 1m Candlestick',
    ylabel='Price',
    ylabel_lower='Volume',
    figratio=(16, 9),
    datetime_format='%H:%M',
    warn_too_much_data=2000,
    savefig=os.path.join(plots_dir, 'candlestick.png')
)

# Compute & plot returns
# pct_change() gives simple returns; drop the NA at first row
returns = price.pct_change().dropna()

plt.figure()
plt.plot(returns, label='Returns')
plt.title('1-Minute Returns')
plt.xlabel('Time')
plt.ylabel('Return')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'returns.png'))
plt.close()

# Compute & plot rolling volatility
# Use a 20‑period rolling window to estimate volatility (std)
volatility = returns.rolling(window=20).std()

plt.figure()
plt.plot(volatility, label='Rolling Volatility (20-min)')
plt.title('20-Period Volatility')
plt.xlabel('Time')
plt.ylabel('Volatility')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'volatility.png'))
plt.close()

# Decompose time series
# Choose period=1440 for daily seasonality (if 1‑min data)
decomp = seasonal_decompose(subset, model='additive', period=1440)

# This creates a 4‑panel figure: observed, trend, seasonal, resid
decomp.plot()
plt.suptitle('Seasonal Decomposition (Additive)', y=1.02)
plt.savefig(os.path.join(plots_dir, 'decomposition.png'))
plt.close()

# Stationarity tests
# ---- ADF Test ----
adf_stat, adf_p, _, _, adf_crit, _ = adfuller(subset.dropna())
print(f"ADF  Statistic: {adf_stat:.4f}")
print(f"ADF  p‑Value : {adf_p:.4f}")
print("ADF Critical Values:")
for sig, val in adf_crit.items():
    print(f"  {sig}: {val:.4f}")

# ---- KPSS Test ----
kpss_stat, kpss_p, _, kpss_crit = kpss(subset.dropna(), regression='c', nlags='auto')
print(f"\nKPSS Statistic: {kpss_stat:.4f}")
print(f"KPSS p‑Value : {kpss_p:.4f}")
print("KPSS Critical Values:")
for sig, val in kpss_crit.items():
    print(f"  {sig}: {val:.4f}")

# Plot ACF & PACF
plot_acf(subset.dropna(), lags=40, title='ACF (up to 40 lags)')
plt.savefig(os.path.join(plots_dir, 'acf.png'))
plt.close()

plot_pacf(subset.dropna(), lags=40, title='PACF (up to 40 lags)')
plt.savefig(os.path.join(plots_dir, 'pacf.png'))
plt.close()
# File: Modeling/src/data_loader.py
import pandas as pd

def load_bnb_data(path: str = None) -> pd.DataFrame:
    """
    Load and preprocess the BNB-USDT 1-minute time-series data.
    - path: optional override for CSV location
    Returns a DataFrame indexed by timestamp with regular 1-minute frequency.
    """
    # Default CSV path
    default_path = "/Users/mchildress/Active Code/ts_basics/data/bnbusdt_1m.csv"
    csv_path = path or default_path

    # Read CSV, parse timestamp column, and set as index
    df = pd.read_csv(
        csv_path,
        parse_dates=['open_time'],
        index_col='open_time'
    )
    # Keep only necessary columns
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Ensure a regular 1-minute frequency
    df = df.asfreq('T')

    # Forward-fill any missing values
    df = df.fillna(method='ffill')

    return df
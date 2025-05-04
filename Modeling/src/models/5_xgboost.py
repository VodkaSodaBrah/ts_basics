

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class BnbXGBModel:
    """
    Wrapper for training and evaluating an XGBoost regressor
    on BNB/USDT 1-minute log-returns.
    """

    def __init__(self,
                 seq_len: int = 60,
                 params: dict = None,
                 num_round: int = 200,
                 test_size: float = 0.2,
                 random_state: int = 42):
        self.seq_len = seq_len
        self.params = params or {
            "objective": "reg:squarederror",
            "tree_method": "gpu_hist"
        }
        self.num_round = num_round
        self.test_size = test_size
        self.random_state = random_state
        self.model = None

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load CSV and compute log-returns."""
        df = pd.read_csv(csv_path, parse_dates=['open_time'], index_col='open_time')
        df['log_ret'] = np.log(df['close']).diff() * 100
        df = df.dropna(subset=['log_ret'])
        return df

    def create_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Generate rolling and lag features and sliding windows."""
        df['lag1']  = df['log_ret'].shift(1)
        df['roll5'] = df['log_ret'].rolling(5).mean()
        df['roll10']= df['log_ret'].rolling(10).std()
        df_feat = df[['log_ret','lag1','roll5','roll10']].dropna()

        arr = df_feat.values.astype(np.float32)
        X, Y = [], []
        for i in range(len(arr) - self.seq_len):
            X.append(arr[i:i + self.seq_len])
            Y.append(arr[i + self.seq_len, 0])
        X = np.stack(X)
        Y = np.stack(Y)
        return X, Y

    def train_test_split(self, X: np.ndarray, Y: np.ndarray):
        """Split into train/test sets."""
        return train_test_split(
            X, Y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=False
        )

    def fit(self, csv_path: str):
        """Load data, engineer features, train XGBoost model."""
        df = self.load_data(csv_path)
        X, Y = self.create_features(df)
        X_train, X_test, Y_train, Y_test = self.train_test_split(X, Y)

        # Flatten for XGBoost
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat  = X_test.reshape(X_test.shape[0], -1)

        dtrain = xgb.DMatrix(X_train_flat, label=Y_train)
        dtest  = xgb.DMatrix(X_test_flat,  label=Y_test)
        watchlist = [(dtrain, 'train'), (dtest, 'eval')]

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_round,
            evals=watchlist,
            verbose_eval=True
        )
        self.X_test_flat = X_test_flat
        self.Y_test = Y_test
        return self.model

    def evaluate(self) -> float:
        """Compute RMSE on test set."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        preds = self.model.predict(xgb.DMatrix(self.X_test_flat))
        rmse = mean_squared_error(self.Y_test, preds, squared=False)
        return rmse

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict from raw sliding-window data."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        flat = X.reshape(X.shape[0], -1).astype(np.float32)
        return self.model.predict(xgb.DMatrix(flat))
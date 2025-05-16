#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def simulate_payout(K, n_sim=200_000, seed=42):
    np.random.seed(seed + K)
    trials = np.random.geometric(p=0.5, size=n_sim) - 1
    heads = np.minimum(trials, K)
    payout = np.where(heads > 0, 2**(heads + 1) - 2, 0)
    return payout.mean()

def main():
    # 1) Generate stable data
    Ks = np.arange(5, 105, 5)  # more points up to K=100
    seeds = [0, 1, 2, 3, 4]
    mean_payouts = []
    for K in Ks:
        # average over multiple seeds
        vals = [simulate_payout(K, n_sim=200_000, seed=s) for s in seeds]
        sorted_vals = sorted(vals)
        # drop the smallest and largest before averaging
        trimmed = sorted_vals[1:-1]
        mean = np.mean(trimmed)
        mean_payouts.append(mean)
        print(f"  Cap={K:3d} → Mean payout ≈ {mean:.4f}")
    df = pd.DataFrame({'K': Ks, 'mean_payout': mean_payouts})
    df['analytic'] = 2 * df['K'] - 2
    df['logK'] = np.log(df['K'])
    df['tail_corr'] = 2.0 ** (1 - df['K'])

    X = df[['K', 'analytic', 'logK', 'tail_corr']]
    y = df['mean_payout']
    # Split out a small test set to monitor overfitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # 2) Simple ridge regression on engineered features
    from sklearn.linear_model import RidgeCV
    alphas = [0.1, 1, 10, 100]
    ridge = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train)
    ridge_score = ridge.score(X_test, y_test)
    print(f"Ridge test R² = {ridge_score:.4f}, chosen alpha = {ridge.alpha_}")
    best_model = ridge
    ridge_coef = ridge.coef_
    ridge_inter = ridge.intercept_
    print("Ridge intercept:", ridge_inter)
    print("Ridge coefficients:", ridge_coef)

    # 3) Train MLP with scaling
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=(32, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=1e-3,
        max_iter=5000,
        tol=1e-5,
        random_state=0
    )
    mlp.fit(X_train_scaled, y_train)
    mlp_score = mlp.score(X_test_scaled, y_test)
    print("\nMLP test R²:", mlp_score)

    # 5) Robust regression with HuberRegressor
    from sklearn.linear_model import HuberRegressor
    huber = HuberRegressor().fit(X_train, y_train)
    huber_score = huber.score(X_test, y_test)
    print("HuberRegressor test R²:", huber_score)

    # 6) RANSAC with Ridge base estimator
    from sklearn.linear_model import RANSACRegressor
    from sklearn.linear_model import Ridge
    ransac = RANSACRegressor(estimator=Ridge(alpha=ridge.alpha_), min_samples=0.5, random_state=0)
    ransac.fit(X_train, y_train)
    ransac_score = ransac.score(X_test, y_test)
    print("RANSACRegressor test R²:", ransac_score)

    # 7) Modeling log-transformed target
    from sklearn.metrics import r2_score
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # Ridge on log-target
    ridge_log = RidgeCV(alphas=alphas, cv=5).fit(X_train, y_train_log)
    log_preds = ridge_log.predict(X_test)
    ridge_log_score = r2_score(y_test_log, log_preds)
    print("Ridge (log-target) R² on log-scale:", ridge_log_score)

    # MLP on log-target
    mlp_log = MLPRegressor(
        hidden_layer_sizes=(32,32),
        activation='relu',
        solver='adam',
        learning_rate_init=1e-3,
        max_iter=5000,
        tol=1e-5,
        random_state=0
    )
    mlp_log.fit(X_train_scaled, y_train_log)
    log_preds_mlp = mlp_log.predict(X_test_scaled)
    mlp_log_score = r2_score(y_test_log, log_preds_mlp)
    print("MLP (log-target) R² on log-scale:", mlp_log_score)

    # 4) Report full table
    df['poly_pred'] = best_model.predict(df[['K', 'analytic', 'logK', 'tail_corr']])
    df['mlp_pred']  = mlp.predict(scaler.transform(df[['K', 'analytic', 'logK', 'tail_corr']]))
    print("\n K |  Mean  | PolyPred | MLPPred")
    print("---+--------+----------+--------")
    for _,r in df.iterrows():
        print(f"{int(r.K):>3d} | {r.mean_payout:6.2f} |"
              f" {r.poly_pred:8.2f} | {r.mlp_pred:6.2f}")

if __name__ == "__main__":
    main()
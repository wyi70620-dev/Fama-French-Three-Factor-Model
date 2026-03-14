# === * Library Imports ===
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import Parallel, delayed
from itertools import product
import matplotlib.pyplot as plt


# === 1. Define an auxiliary function ===
def build_H(X):
    X = np.asarray(X, dtype=float)
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])

def build_F(k, state_model="RW", phi=None):
    if state_model == "RW":
        F = np.eye(k)
    elif state_model == "AR1":
        if phi is None:
            raise ValueError("state_model='AR1' must provide phi")
        if abs(float(phi)) >= 1.0:
            raise ValueError("state_model='AR1' must have abs(phi) < 1.0")
        F = float(phi) * np.eye(k)
    else:
        raise ValueError("Undefined state_model")
    return F

def initialization(k, state_model, Q_mat, phi, x0, P0):
    if x0 is None:
        x0_vec = np.zeros((k, 1))
    else:
        x0_vec = np.asarray(x0, dtype=float).reshape(k, 1)

    if P0 is None:
        if state_model == "RW":
            P0_mat = np.eye(k) * 1000.0
        elif state_model == "AR1":
            if phi is None:
                raise ValueError("state_model='AR1' must provide phi")
            if abs(float(phi)) >= 1.0:
                raise ValueError("state_model='AR1' must have abs(phi) < 1.0")          
            P0_mat = np.asarray(Q_mat, dtype=float) / (1.0 - phi**2)
        else:
            raise ValueError("Undefined state_model")
    else:
        P0_mat = np.asarray(P0, dtype=float)
        assert P0_mat.shape == (k, k), "P0 must be a (k, k) matrix"

    return x0_vec, P0_mat


# === 2. Data Loading and preprocessing ===
def preprocess_stock_data(stock_filename, factors_df):
    stock_df = pd.read_csv(stock_filename)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], format='%Y-%m')
    merged_df = pd.merge(stock_df, factors_df, on='Date', how='inner')
    merged_df['Excess Return'] = merged_df['Monthly Return'] - merged_df['RF']
    merged_df = merged_df.rename(columns={
        'Mkt-RF': 'x1',
        'SMB': 'x2',
        'HML': 'x3'
    })
    return merged_df[['Date', 'Excess Return', 'x1', 'x2', 'x3']]

def train_test_split_time(df, test_ratio=0.2, date_col='Date'):
    df = df.sort_values(date_col).reset_index(drop=True)
    split_index = int(len(df) * (1 - test_ratio))
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = np.asarray(train_df[['x1', 'x2', 'x3']])
    y_train = np.asarray(train_df['Excess Return'])
    X_test = np.asarray(test_df[['x1', 'x2', 'x3']])
    y_test = np.asarray(test_df['Excess Return'])

    return X_train, y_train, X_test, y_test

factors_df = pd.read_csv("~/Desktop/Fama_French_3_Factors_Monthly.csv")
factors_df['Date'] = pd.to_datetime(factors_df['Date'], format='%Y-%m')
amzn_df = preprocess_stock_data("~/Desktop/AMZN.csv", factors_df)
aapl_df = preprocess_stock_data("~/Desktop/AAPL.csv", factors_df)
m_df    = preprocess_stock_data("~/Desktop/M.csv", factors_df)

amzn_X_train, amzn_y_train, amzn_X_test, amzn_y_test = train_test_split_time(amzn_df)
aapl_X_train, aapl_y_train, aapl_X_test, aapl_y_test = train_test_split_time(aapl_df)
m_X_train, m_y_train, m_X_test, m_y_test = train_test_split_time(m_df)


# === 3. Performance Metrics Calculation Function ===
def prediction_metrics(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)

    # Error Class
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + eps)))

    # Directional Accuracy
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)
    da = np.mean(sign_true == sign_pred)

    # Pearson correlation coefficient
    if np.std(y_true) < eps or np.std(y_pred) < eps:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(y_pred, y_true)[0, 1])

    # Spearman’s rank correlation coefficient
    y_true_rank = pd.Series(y_true).rank(method='average').to_numpy()
    y_pred_rank = pd.Series(y_pred).rank(method='average').to_numpy()
    if np.std(y_true_rank) < eps or np.std(y_pred_rank) < eps:
        spearman = np.nan
    else:
        spearman = float(np.corrcoef(y_pred_rank, y_true_rank)[0, 1])

    # Theil’s U
    rmse_num = np.sqrt(np.mean((y_pred - y_true) ** 2))
    denom = np.sqrt(np.mean(y_pred ** 2)) + np.sqrt(np.mean(y_true ** 2))
    theil_u = float(rmse_num / (denom + eps))

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'DA': da,
        'Pearson': pearson,
        'Spearman': spearman,
        'TheilU': theil_u
    }
    

# === 4. Hyperparameter Tuning ===
def get_kf_loglik(y, H_rows, q, r, state_model="RW", phi=None, x0=None, P0=None, eps=1e-9):
    y = np.asarray(y, dtype=float).ravel()
    H_rows = np.asarray(H_rows, dtype=float)
    T, k = H_rows.shape
    F = build_F(k, state_model=state_model, phi=phi)
    Q = q * np.eye(k)
    R = float(r)
    assert y.shape[0] == T, "y and H_rows must have the same number of periods T"
    assert np.isscalar(q) and np.isscalar(r), "q and r must have to be scalar"
    assert r > 0 and q >= 0, "r and q must have to be: r > 0 and q >= 0"

    x0_vec, P0_mat = initialization(k, state_model=state_model, Q_mat=Q, phi=phi, x0=x0, P0=P0)
        
    kf = KalmanFilter(dim_x=k, dim_z=1)
    kf.F = F
    kf.Q = Q
    kf.R = R
    kf.x = x0_vec
    kf.P = P0_mat
        
    loglik = 0.0
    log2pi = np.log(2.0 * np.pi)

    for t in range(T):
        kf.predict()
        H_t = H_rows[t].reshape(1, k)
        z_t = float(y[t])
        kf.update(z_t, H=H_t)
        v_t = float(kf.y.item())
        S_t = float(kf.S.item())

        if S_t <= 0:
            S_t = eps

        loglik += -0.5 * (log2pi + np.log(S_t) + (v_t * v_t) / S_t)

    return float(loglik)

def run_kf_grid_search(y, H_rows, param_ranges, state_model="RW", x0=None, P0=None, n_jobs=-1):
    keys = list(param_ranges.keys())
    combinations = list(product(*[param_ranges[k] for k in keys]))
    param_grid = pd.DataFrame(combinations, columns=keys)

    def evaluate(row):
        params = row.to_dict()
        q = float(params['q'])
        r = float(params['r'])
        phi = params.get('phi', None)
        ll = get_kf_loglik(y, H_rows, q, r, state_model=state_model, phi=phi, x0=x0, P0=P0)
        return ll

    loglik_list = Parallel(n_jobs=n_jobs)(
        delayed(evaluate)(row) for _, row in param_grid.iterrows()
    )
    param_grid['LOG_LIK'] = loglik_list
    best_row = param_grid.loc[param_grid['LOG_LIK'].idxmax()]
    return best_row.to_dict()


# === 5. Model Training and Prediction ===
def kalman_methods(y, H_rows, Q, R, state_model="RW", phi=None, x0=None, P0=None):
    # Normativity and Examination
    y = np.asarray(y, dtype=float).ravel()
    H_rows = np.asarray(H_rows, dtype=float)
    T, k = H_rows.shape
    F = build_F(k, state_model=state_model, phi=phi)
    assert y.shape[0] == T, "y and H_rows must have the same number of periods T"
    
    Q = np.asarray(Q, dtype=float)
    if Q.ndim == 0:
        Q_mat = float(Q) * np.eye(k)
    elif Q.ndim == 1:
        assert Q.shape[0] == k, "Length of Q’s diagonal vector must equal the state dimension k"
        Q_mat = np.diag(Q)
    else:
        assert Q.shape == (k, k), "Q must be a (k, k) matrix"
        Q_mat = Q

    if np.isscalar(R):
        R_mat = np.array([[float(R)]], dtype=float)
    else:
        R_mat = np.asarray(R, dtype=float)
        assert R_mat.shape == (1, 1), "R must be a scalar or a (1, 1) matrix"
    
    # Initialization
    x0_vec, P0_mat = initialization(k, state_model=state_model, Q_mat=Q_mat, phi=phi, x0=x0, P0=P0)

    kf = KalmanFilter(dim_x=k, dim_z=1)
    kf.F = F
    kf.Q = Q_mat
    kf.R = R_mat
    kf.x = x0_vec
    kf.P = P0_mat
    xs_filt = np.zeros((T, k))
    Ps_filt = np.zeros((T, k, k))
    xs_pred = np.zeros((T, k))
    Ps_pred = np.zeros((T, k, k))
    
    # Kalman Filter
    for t in range(T):
        # Prediction
        kf.predict()
        xs_pred[t] = kf.x.ravel()
        Ps_pred[t] = kf.P
        
        # Update
        H_t = H_rows[t][None, :]
        z_t = y[t]
        kf.update(z_t, H=H_t)
        xs_filt[t] = kf.x.ravel()
        Ps_filt[t] = kf.P
    
    # Kalman Smoother
    xs_smooth = np.zeros_like(xs_filt)
    Ps_smooth = np.zeros_like(Ps_filt)
    xs_smooth[-1] = xs_filt[-1]
    Ps_smooth[-1] = Ps_filt[-1]
    F = kf.F
    
    for t in range(T-2, -1, -1):
        Pt_f = Ps_filt[t]
        Pt1_p = Ps_pred[t+1]
        C_t = Pt_f @ F.T @ np.linalg.inv(Pt1_p)
        x_update = xs_smooth[t+1] - xs_pred[t+1]
        xs_smooth[t] = xs_filt[t] + (C_t @ x_update)
        P_update = Ps_smooth[t+1] - Pt1_p
        Ps_smooth[t] = Pt_f + C_t @ P_update @ C_t.T
    
    # The last time point (as the prediction result)
    x_last = xs_smooth[-1].copy()
    P_last = Ps_smooth[-1].copy()
    
    return {
        'x_last': x_last,
        'P_last': P_last,
        'F': F
    }

def kf_train(X_train, y_train, param_ranges, state_model="RW", x0=None, P0=None):
    H_train = build_H(X_train)
    best_param = run_kf_grid_search(y_train, H_train, param_ranges, state_model=state_model, x0=x0, P0=P0, n_jobs=-1)
    Q, R = float(best_param['q']), float(best_param['r'])
    phi = best_param.get('phi', None)

    result = kalman_methods(y_train, H_train, Q, R, state_model=state_model, phi=phi, x0=x0, P0=P0)
    model = {
        "k": H_train.shape[1],
        "Q": Q, "R": R,
        "F": result["F"].copy(),
        "coefficient": result["x_last"].copy(),
        "covariance": result["P_last"].copy()
    }
    return model

def kf_predict(X_test, y_test, model, update_with_observed=True):
    H_test = build_H(X_test)
    k = model["k"]
    q, r = model["Q"], model["R"]
    F = model["F"]
    x = model["coefficient"].reshape(-1, 1).copy()
    P = model["covariance"].copy()
    y_pred = np.zeros(H_test.shape[0])

    if not update_with_observed:
        y_pred = (H_test @ x).ravel()
        return y_pred
    else:
        kf = KalmanFilter(dim_x=k, dim_z=1)
        kf.F = F
        kf.Q = q * np.eye(k)
        kf.R = float(r)
        kf.x = x
        kf.P = P

        for t in range(H_test.shape[0]):
            H_t = H_test[t:t+1, :]
            if y_test is None:
                raise ValueError("update_with_observed=True must provide y_test")
            if t % 2 == 0:  
                kf.update(float(y_test[t]), H=H_t)
            y_pred[t] = float((H_t @ kf.x).item())
            if t < H_test.shape[0] - 1:
                kf.predict()
        return y_pred

# Random-Walk
RW_param_ranges = {
    "q": np.logspace(-6, -2, 13),
    "r": np.logspace(-6, -2, 13)
}

RW_amzn_model = kf_train(amzn_X_train, amzn_y_train, RW_param_ranges, state_model="RW")
RW_amzn_pred = kf_predict(amzn_X_test, amzn_y_test, RW_amzn_model)

RW_aapl_model = kf_train(aapl_X_train, aapl_y_train, RW_param_ranges, state_model="RW")
RW_aapl_pred = kf_predict(aapl_X_test, aapl_y_test, RW_aapl_model)

RW_m_model = kf_train(m_X_train, m_y_train, RW_param_ranges, state_model="RW")
RW_m_pred = kf_predict(m_X_test, m_y_test, RW_m_model)

# Auto-Regressive(order 1)
AR_param_ranges = {
    "q": np.logspace(-6, -2, 13),
    "r": np.logspace(-6, -2, 13),
    "phi": list(np.arange(0.01, 1, 0.01))
}

AR_amzn_model = kf_train(amzn_X_train, amzn_y_train, AR_param_ranges, state_model="AR1")
AR_amzn_pred = kf_predict(amzn_X_test, amzn_y_test, AR_amzn_model)

AR_aapl_model = kf_train(aapl_X_train, aapl_y_train, AR_param_ranges, state_model="AR1")
AR_aapl_pred = kf_predict(aapl_X_test, aapl_y_test, AR_aapl_model)

AR_m_model = kf_train(m_X_train, m_y_train, AR_param_ranges, state_model="AR1")
AR_m_pred = kf_predict(m_X_test, m_y_test, AR_m_model)


# === 6. Results: Statistical Metrics ===
results = {
    'KF-RW': {
        'AMZN': prediction_metrics(amzn_y_test, RW_amzn_pred),
        'AAPL': prediction_metrics(aapl_y_test, RW_aapl_pred),
        'M':    prediction_metrics(m_y_test,    RW_m_pred)
    },
    'KF-AR1': {
        'AMZN': prediction_metrics(amzn_y_test, AR_amzn_pred),
        'AAPL': prediction_metrics(aapl_y_test, AR_aapl_pred),
        'M':    prediction_metrics(m_y_test,    AR_m_pred)
    }
}

results_df = pd.DataFrame({
    (model, stock): results[model][stock]
    for model in results.keys()
    for stock in results[model].keys()
}).T

results_df.index.names = ['Model', 'Stock']
results_df = results_df.reset_index()
print(results_df)


# === 7. Results: Visualization ===
stock_list = ['AMZN', 'AAPL', 'M']
y_true_dict = {
    'AMZN': amzn_y_test,
    'AAPL': aapl_y_test,
    'M':    m_y_test
}
pred_dicts = {
    'AMZN': {
        'KF-RW':  RW_amzn_pred,
        'KF-AR1': AR_amzn_pred
    },
    'AAPL': {
        'KF-RW':  RW_aapl_pred,
        'KF-AR1': AR_aapl_pred
    },
    'M': {
        'KF-RW':  RW_m_pred,
        'KF-AR1': AR_m_pred
    }
}

def get_test_dates(df, test_ratio=0.2, date_col='Date'):
    df = df.sort_values(date_col).reset_index(drop=True)
    split_index = int(len(df) * (1 - test_ratio))
    return df.loc[split_index:, date_col].values

test_dates = {
    'AMZN': get_test_dates(amzn_df),
    'AAPL': get_test_dates(aapl_df),
    'M':    get_test_dates(m_df)
}

# Residual Plot
def plot_residuals(stock, y_true, preds_dict):
    plt.figure(figsize=(8, 5))
    for name, y_pred in preds_dict.items():
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, s=18, alpha=0.7, label=name)
    plt.axhline(0, linestyle='--', linewidth=1)
    plt.xlabel('Predicted Excess Return')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title(f'{stock}: Residual Plot')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Actual vs Predicted
def plot_actual_vs_pred(stock, y_true, preds_dict):
    plt.figure(figsize=(8, 5))
    min_val = min(np.min(y_true), *(np.min(p) for p in preds_dict.values()))
    max_val = max(np.max(y_true), *(np.max(p) for p in preds_dict.values()))
    padding = 0.02 * (max_val - min_val if max_val > min_val else 1.0)
    line_min, line_max = min_val - padding, max_val + padding
    plt.plot([line_min, line_max], [line_min, line_max],
             linestyle='--', linewidth=1, label='45° line')

    for name, y_pred in preds_dict.items():
        plt.scatter(y_true, y_pred, s=18, alpha=0.7, label=name)

    plt.xlabel('Actual Excess Return')
    plt.ylabel('Predicted Excess Return')
    plt.title(f'{stock}: Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Time series
def plot_time_series(stock, dates, y_true, preds_dict):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_true, linewidth=2, label='Actual', alpha=0.9)
    for name, y_pred in preds_dict.items():
        plt.plot(dates, y_pred, linewidth=1.5, alpha=0.9, label=name)
    plt.xlabel('Date')
    plt.ylabel('Excess Return')
    plt.title(f'{stock}: Time Series')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Quantile-Quantile Plot
def plot_qq(stock, y_true, preds_dict, q=200):
    qs = np.linspace(0, 1, q)
    actual_q = np.quantile(y_true, qs)
    plt.figure(figsize=(8, 5))
    min_val = min(np.min(actual_q), *(np.min(np.quantile(p, qs)) for p in preds_dict.values()))
    max_val = max(np.max(actual_q), *(np.max(np.quantile(p, qs)) for p in preds_dict.values()))
    padding = 0.02 * (max_val - min_val if max_val > min_val else 1.0)
    line_min, line_max = min_val - padding, max_val + padding
    plt.plot([line_min, line_max], [line_min, line_max], linestyle='--', linewidth=1, label='45° line')

    for name, y_pred in preds_dict.items():
        pred_q = np.quantile(y_pred, qs)
        plt.scatter(actual_q, pred_q, s=14, alpha=0.7, label=name)

    plt.xlabel('Actual Quantiles')
    plt.ylabel('Predicted Quantiles')
    plt.title(f'{stock}: QQ Plot')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot
for stock in stock_list:
    y_true = y_true_dict[stock]
    preds  = pred_dicts[stock]
    dates  = test_dates[stock]

    plot_residuals(stock, y_true, preds)
    plot_actual_vs_pred(stock, y_true, preds)
    plot_time_series(stock, dates, y_true, preds)
    plot_qq(stock, y_true, preds, q=200)
